"""Главный конвейер: кадр -> движение -> детекция -> трекинг -> БД -> уведомления.

Новое:
  - мультиразрешение: есть движение — полный кадр, нет — мелкий (экономия CPU);
  - усреднение ночных кадров (меньше шума — номер читается в темноте);
  - трекинг людей: один проход/стояние = ОДНО уведомление (+ «вошёл/вышел»);
  - трекинг машин: номер собирается с нескольких кадров «голосованием»;
  - свои/чужие номера (car.known_plates).

Если трекинг выключен (tracking.enabled: false) — работает старый путь
(каждое появление с кулдауном по имени/номеру).
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera import Camera
from src.car_analyzer import CarAnalyzer
from src.config import Config
from src.database import Database
from src.detector import YoloDetector
from src.face_recognizer import FaceRecognizer
from src.motion import MotionDetector
from src.night_vision import ensure_night_quality, is_low_light
from src.notifier import Notifier
from src.plate_recognizer import PlateRecognizer
from src.recorder import ClipRecorder
from src.tracker import CentroidTracker, majority_vote

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config, camera_cfg, db: Database, notifier: Notifier):
        self.cfg = cfg
        self.camera_cfg = camera_cfg
        self.db = db
        self.notifier = notifier
        self.name = str(camera_cfg.get("name", "cam"))

        self.cam = Camera(
            camera_cfg["source"],
            width=int(cfg.camera.width),
            height=int(cfg.camera.height),
        )
        self.detector: Optional[YoloDetector] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.car_analyzer: Optional[CarAnalyzer] = None
        self.plate_recognizer: Optional[PlateRecognizer] = None

        self.motion = MotionDetector(
            threshold=int(cfg.motion.get("threshold", 1500)),
            scale=float(cfg.motion.get("scale", 0.25)),
        ) if bool(cfg.motion.get("enabled", True)) else None
        self.recorder = ClipRecorder(cfg, self.name) if bool(cfg.recording.get("enabled", False)) else None

        self.tracking_enabled = bool(self.cfg.tracking.get("enabled", True))
        self.face_tracker = CentroidTracker(
            max_disappeared=int(self.cfg.tracking.get("max_disappeared", 20)),
            max_distance=float(self.cfg.tracking.get("max_distance", 90)),
        ) if self.tracking_enabled else None
        self.car_tracker = CentroidTracker(
            max_disappeared=int(self.cfg.tracking.get("car_max_disappeared", 6)),
            max_distance=float(self.cfg.tracking.get("car_max_distance", 120)),
        ) if self.tracking_enabled else None
        self.known_plates = [str(p) for p in (self.cfg.car.get("known_plates", []))]

        self._events_dir = cfg.storage.resolve(cfg.storage.events_dir)
        self._frame_count = 0
        self._auto_capture_dir = cfg.resolve(cfg.face.get("auto_capture_dir",
                                                           "data/datasets/auto_unknown"))

    # ---------- имя для уведомлений ----------
    def _display_name(self, key: str) -> str:
        names = self.cfg.people.get("names", {})
        return names.get(key, key.capitalize())

    # ---------- ленивая загрузка моделей ----------
    def _ensure_models(self) -> None:
        if self.detector is None:
            self.detector = YoloDetector(self.cfg, conf_threshold=float(self.cfg.car.conf_threshold))
        if self.face_recognizer is None and bool(self.cfg.face.get("enabled", True)):
            self.face_recognizer = FaceRecognizer(self.cfg)
        if self.car_analyzer is None and bool(self.cfg.car.get("enabled", True)):
            self.car_analyzer = CarAnalyzer(self.cfg)
        if self.plate_recognizer is None and bool(self.cfg.car.get("plate_enabled", True)):
            self.plate_recognizer = PlateRecognizer(self.cfg)

    # ---------- сохранение ----------
    def _save_image(self, frame, prefix: str) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir = self._events_dir / self.name / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def _save_image_bytes(self, jpg_bytes, prefix: str) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir = self._events_dir / self.name / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        Path(path).write_bytes(jpg_bytes)
        return str(path)

    def _save_auto_capture(self, crop) -> Optional[str]:
        """Лицо незнакомца -> папка авто-обучения."""
        if not bool(self.cfg.face.get("auto_capture_unknown", True)) or crop is None:
            return None
        out_dir = self._auto_capture_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    # ---------- ночное усреднение кадров ----------
    def _read_averaged(self) -> Optional[np.ndarray]:
        """Если кадр тёмный и включено усреднение — берём N кадров, усредняем."""
        first = self.cam.read()
        if first is None:
            return None
        if not bool(self.cfg.camera.get("frame_averaging_night", True)):
            return first
        if not is_low_light(first, int(self.cfg.camera.night_brightness_threshold)):
            return first
        count = max(2, int(self.cfg.camera.get("frame_averaging_count", 3)))
        frames = [first]
        for _ in range(count - 1):
            f = self.cam.read()
            if f is None or f.shape != first.shape:
                break
            frames.append(f)
        if len(frames) < 2:
            return first
        stacked = np.stack([f.astype(np.float32) for f in frames])
        avg = np.median(stacked, axis=0).astype(np.uint8)
        avg = cv2.bilateralFilter(avg, 5, 15, 15)
        return avg

    # ---------- ЛИЦА ----------
    def _maybe_line_cross(self, track) -> None:
        """Определяем «вошёл/вышел» при пересечении линии entry_line_y."""
        if "action" in track.data:
            return
        prev, cur = track.prev_centroid, track.centroid
        if prev is None or cur is None:
            return
        line = float(self.cfg.tracking.get("entry_line_y", 0.5))
        if prev[1] < line <= cur[1]:
            track.data["action"] = "вошёл"
        elif prev[1] >= line > cur[1]:
            track.data["action"] = "вышел"

    def _announce_face(self, frame, track, best_name: str) -> None:
        d = track.data
        action = d.get("action")
        if best_name == "unknown":
            if not bool(self.cfg.face.get("notify_unknown", True)):
                d["notified"] = True
                return
            if self.db.is_in_cooldown("face", "unknown", float(self.cfg.face.cooldown_minutes)):
                d["notified"] = True
                return
            img = str(self._save_image(frame, "face_unknown")) if bool(self.cfg.face.get("save_unknown", True)) else ""
            x1, y1, x2, y2 = d.get("bbox", (0, 0, 0, 0))
            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
            self._save_auto_capture(crop)
            self.db.save_face_event("unknown", d.get("sim"), self.name, img)
            label = str(self.cfg.people.get("unknown_label", "Незнакомец"))
            msg = f"Камера {self.name}. Схожесть: {d.get('sim', 0):.2f}"
            self.notifier.send_event(label, msg, img)
            self.recorder.trigger("unknown") if self.recorder else None
            d["notified"] = True
            return
        if self.db.is_in_cooldown("face", best_name, float(self.cfg.face.cooldown_minutes)):
            d["notified"] = True
            return
        img = str(self._save_image(frame, f"face_{best_name}"))
        self.db.save_face_event(best_name, d.get("sim", 0), self.name, img)
        label = self._display_name(best_name)
        title = f"{label} {action}" if action else f"{label} у камеры"
        self.notifier.send_event(title, f"Камера {self.name}. Схожесть: {d.get('sim', 0):.2f}", img)
        self.recorder.trigger("face") if self.recorder else None
        d["notified"] = True

    def _process_faces(self, frame) -> None:
        if self.face_recognizer is None:
            return
        matches, _ = self.face_recognizer.recognize(frame)
        boxes = [m.bbox for m in matches]
        if self.face_tracker is None:
            # старый путь: каждое появление с кулдауном
            for m in matches:
                fake = Track_stub()
                fake.data.update({"sim": m.similarity, "bbox": m.bbox})
                self._announce_face(frame, fake, m.name)
            return
        tracks = self.face_tracker.update(boxes)
        min_frames = int(self.cfg.tracking.get("min_frames_to_notify", 3))
        for track, m in zip(tracks, matches):
            if track is None:
                continue
            d = track.data
            d.setdefault("votes", Counter())
            d.setdefault("bbox", m.bbox)
            d["bbox"] = m.bbox
            d["sim"] = max(d.get("sim", 0.0), m.similarity)
            d["votes"][m.name] += 1
            self._maybe_line_cross(track)
            if d.get("notified"):
                continue
            if track.seen < min_frames:
                continue
            best = majority_vote(list(d["votes"].elements())) or "unknown"
            self._announce_face(frame, track, best)

    # ---------- АВТОМОБИЛИ ----------
    def _finalize_car(self, frame, track) -> None:
        d = track.data
        plate = majority_vote(d.get("plate_votes", []))
        if plate and self.db.is_in_cooldown("car", plate, float(self.cfg.car.cooldown_minutes)):
            d["notified"] = True
            return
        brand, color = d.get("brand", "unknown"), d.get("color", "unknown")
        img = str(self._save_image_bytes(d.get("last_jpg", b""), "car")) if d.get("last_jpg") else ""
        self.db.save_car_event(self.name, plate or "", brand, color, img, "")
        if plate:
            known = plate in self.known_plates
            title = "Своя машина" if known else f"Чужая машина {plate}"
        else:
            title = "Автомобиль (номер не распознан)"
        self.notifier.send_event(title,
                                 f"Камера {self.name}\nМарка: {brand}\nЦвет: {color}\nНомер: {plate or '—'}",
                                 img)
        self.recorder.trigger("car") if self.recorder else None
        d["notified"] = True

    def _process_vehicles(self, frame, dets) -> None:
        if not dets or self.car_analyzer is None:
            return
        boxes = [d.bbox for d in dets]
        vote_frames = int(self.cfg.car.get("plate_vote_frames", 5))
        if self.car_tracker is None:
            # старый путь: один кадр, без трекинга
            zone = None
            for det in dets:
                crop = det.crop(frame)
                info = self.car_analyzer.analyze(crop) if crop is not None else None
                if info is None:
                    continue
                plate = ""
                if self.plate_recognizer is not None:
                    zone = PlateRecognizer.plate_crop_from_car(frame, det.bbox)
                    if zone is not None:
                        plate, _ = self.plate_recognizer.read(zone)
                if plate and self.db.is_in_cooldown("car", plate, float(self.cfg.car.cooldown_minutes)):
                    continue
                full = str(self._save_image(frame, "car"))
                crop_img = str(self._save_image(zone, "plate")) if zone is not None else ""
                self.db.save_car_event(self.name, plate, info.brand, info.color, full, crop_img)
                parts = [f"Камера {self.name}", f"Марка: {info.brand}", f"Цвет: {info.color}"]
                parts.append(f"Номер: {plate}" if plate else "Номер: не распознан")
                self.notifier.send_event("Автомобиль", "\n".join(parts), full, extra_images=[crop_img] if crop_img else ())
                self.recorder.trigger("car") if self.recorder else None
            return

        tracks = self.car_tracker.update(boxes)
        for track, det in zip(tracks, dets):
            if track is None:
                continue
            d = track.data
            d.setdefault("plate_votes", [])
            crop = det.crop(frame)
            info = self.car_analyzer.analyze(crop) if crop is not None else None
            if info is not None:
                d["brand"] = info.brand
                d["color"] = info.color
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                d["last_jpg"] = buf.tobytes()
            if self.plate_recognizer is not None:
                zone = PlateRecognizer.plate_crop_from_car(frame, det.bbox)
                if zone is not None:
                    plate, _ = self.plate_recognizer.read(zone)
                    if plate and plate not in d["plate_votes"][-5:]:
                        d["plate_votes"].append(plate)
            if d.get("notified"):
                continue
            # финализируем: трек пропал ИЛИ набрали достаточно кадров
            if track.disappeared >= 2 or track.seen >= vote_frames:
                self._finalize_car(frame, track)

    # ---------- отрисовка рамок (превью) ----------
    @staticmethod
    def _draw_boxes(frame, dets) -> None:
        for d in dets:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.label} {d.conf:.2f}", (d.x1, max(0, d.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- главный цикл ----------
    def run(self, max_frames: Optional[int] = None) -> None:
        self._ensure_models()
        force_interval = int(self.cfg.motion.get("force_interval_frames", 300))
        detect_every = max(1, int(self.cfg.analysis.detect_every_n_frames))
        frame_scale = float(self.cfg.analysis.frame_scale)
        small_scale = float(self.cfg.analysis.get("small_frame_scale", 0.5))
        multi_res = bool(self.cfg.analysis.get("multi_resolution", True))
        preview = bool(self.cfg.ui.get("preview_enabled", False))
        win = str(self.cfg.ui.get("window_title", "MLCamera — {cam}")).format(cam=self.name)

        log.info("Поток камеры '%s' запущен (трекинг=%s)", self.name, self.tracking_enabled)
        frames = 0
        while True:
            # ночное усреднение кадров (шуменьше → номер/лицо читается в темноте)
            frame = self._read_averaged()
            if frame is None:
                if max_frames is not None:
                    break
                time.sleep(1.0)
                continue
            frames += 1
            self._frame_count += 1

            if bool(self.cfg.camera.night_enabled):
                _, frame = ensure_night_quality(frame, int(self.cfg.camera.night_brightness_threshold))

            if self.recorder is not None:
                self.recorder.push(frame)

            motion_hit = self.motion.detect(frame) if self.motion is not None else True
            every_hit = (self._frame_count % detect_every == 0)
            force_hit = (self._frame_count % force_interval == 0)

            analyze = False
            work = frame
            if motion_hit and (every_hit or force_hit):
                analyze = True
                if frame_scale < 1.0:
                    h, w = frame.shape[:2]
                    work = cv2.resize(frame, (int(w * frame_scale), int(h * frame_scale)))
            elif force_hit and multi_res:
                # нет движения, но периодически «прочёсываем» мелким кадром
                analyze = True
                if small_scale < 1.0:
                    h, w = frame.shape[:2]
                    work = cv2.resize(frame, (int(w * small_scale), int(h * small_scale)))

            if analyze:
                dets = []
                if bool(self.cfg.face.get("enabled", True)):
                    self._process_faces(work)
                if bool(self.cfg.car.get("enabled", True)) and self.detector:
                    dets = self.detector.predict_vehicles(work)
                    self._process_vehicles(work, dets)
                if preview:
                    self._draw_boxes(frame, dets)

            if preview:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            delay = float(self.cfg.analysis.get("debug_delay", 0.0))
            if delay > 0:
                time.sleep(delay)
            if max_frames is not None and frames >= max_frames:
                break
        if self.recorder is not None:
            self.recorder.close()
        self.cam.close()

    def close(self) -> None:
        try:
            self.cam.close()
        except Exception:
            pass


class Track_stub:
    """Минимальный объект с атрибутами Track для старого пути (без трекинга)."""
    def __init__(self):
        self.data: dict = {}
        self.seen = 1
        self.prev_centroid = None
        self.centroid = None