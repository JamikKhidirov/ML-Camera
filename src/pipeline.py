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
from src.plate_recognizer import PlateRecognizer, normalize_plate
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
        self.recorder = (ClipRecorder(cfg, self.name, source_fps=self.cam.fps)
                         if bool(cfg.recording.get("enabled", False)) else None)

        self.tracking_enabled = bool(self.cfg.tracking.get("enabled", True))
        self.face_tracker = CentroidTracker(
            max_disappeared=int(self.cfg.tracking.get("max_disappeared", 20)),
            max_distance=float(self.cfg.tracking.get("max_distance", 90)),
        ) if self.tracking_enabled else None
        self.car_tracker = CentroidTracker(
            max_disappeared=int(self.cfg.tracking.get("car_max_disappeared", 6)),
            max_distance=float(self.cfg.tracking.get("car_max_distance", 120)),
        ) if self.tracking_enabled else None
        # Номера в конфиге могут быть написаны кириллицей/с пробелами —
        # приводим к тому же виду, что и результат OCR (A123BC 777).
        self.known_plates = {
            normalize_plate(str(p)) or str(p).strip().upper()
            for p in (self.cfg.car.get("known_plates") or [])
        }

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
        car_enabled = bool(self.cfg.car.get("enabled", True))
        if self.detector is None and car_enabled:
            self.detector = YoloDetector(self.cfg, conf_threshold=float(self.cfg.car.conf_threshold))
        if self.face_recognizer is None and bool(self.cfg.face.get("enabled", True)):
            self.face_recognizer = FaceRecognizer(self.cfg)
        if self.car_analyzer is None and car_enabled:
            self.car_analyzer = CarAnalyzer(self.cfg)
        if (self.plate_recognizer is None and car_enabled
                and bool(self.cfg.car.get("plate_enabled", True))):
            try:
                self.plate_recognizer = PlateRecognizer(self.cfg)
            except Exception as exc:
                # Без tesseract/easyocr камера всё равно должна работать:
                # просто не читаем номера, а не падаем на старте.
                log.error("Распознавание номеров отключено: %s", exc)
                self.plate_recognizer = None

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
    def _maybe_line_cross(self, track, frame_height: int) -> None:
        """Определяем «вошёл/вышел» при пересечении линии entry_line_y.

        entry_line_y — доля высоты кадра (0.0 сверху ... 1.0 снизу).
        Если в конфиге случайно указали пиксели (> 1), тоже поддерживаем.
        """
        if "action" in track.data or frame_height <= 0:
            return
        prev, cur = track.prev_centroid, track.centroid
        if prev is None or cur is None:
            return
        line = float(self.cfg.tracking.get("entry_line_y", 0.5))
        line_y = line * frame_height if 0.0 < line <= 1.0 else line
        if prev[1] < line_y <= cur[1]:
            track.data["action"] = "вошёл"
        elif prev[1] >= line_y > cur[1]:
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

    def _process_faces(self, work: np.ndarray, frame: np.ndarray,
                       scale: float = 1.0) -> None:
        """Лица ищутся на рабочем (возможно уменьшенном) кадре, а фото
        сохраняется из полного кадра — поэтому bbox пересчитываем обратно."""
        if self.face_recognizer is None:
            return
        matches, _ = self.face_recognizer.recognize(work)
        boxes = [m.bbox for m in matches]
        inv = 1.0 / scale if scale > 0 else 1.0

        def full_bbox(bbox: tuple) -> tuple:
            x1, y1, x2, y2 = bbox
            return (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))

        if self.face_tracker is None:
            # старый путь: каждое появление с кулдауном
            for m in matches:
                fake = Track_stub()
                fake.data.update({"sim": m.similarity, "bbox": full_bbox(m.bbox)})
                self._announce_face(frame, fake, m.name)
            return
        tracks = self.face_tracker.update(boxes)
        # лицам «потерянные» треки не нужны — просто не копим их в памяти
        self.face_tracker.pop_removed()
        min_frames = int(self.cfg.tracking.get("min_frames_to_notify", 3))
        for track, m in zip(tracks, matches):
            if track is None:
                continue
            d = track.data
            d.setdefault("votes", Counter())
            d.setdefault("bbox", full_bbox(m.bbox))
            d["bbox"] = full_bbox(m.bbox)
            d["sim"] = max(d.get("sim", 0.0), m.similarity)
            d["votes"][m.name] += 1
            self._maybe_line_cross(track, work.shape[0])
            if d.get("notified"):
                continue
            if track.seen < min_frames:
                continue
            best = majority_vote(list(d["votes"].elements())) or "unknown"
            self._announce_face(frame, track, best)

    # ---------- АВТОМОБИЛИ ----------
    @staticmethod
    def _remember_best_crop(track_data: dict, crop, full_frame, conf: float) -> bool:
        """Запоминает лучшую (самую крупную и уверенную) вырезку машины за трек.

        Марка определяется один раз при финализации — по лучшему кадру, а не
        на каждом кадре. Это и точнее (машина ближе/резче), и дешевле по CPU.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        score = float(h * w) * (0.5 + max(0.0, min(1.0, conf)))
        if score <= float(track_data.get("best_crop_score", 0.0)):
            return False
        # не держим в памяти полноразмерные вырезки
        max_side = 640
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        track_data["best_crop"] = crop.copy()
        track_data["best_crop_score"] = score
        # полный кадр нужен только для фото события; жмём редко, а не каждый кадр
        ok, buf = cv2.imencode(".jpg", full_frame)
        if ok:
            track_data["last_jpg"] = buf.tobytes()
        return True

    def _finalize_car(self, frame, track) -> None:
        d = track.data
        plate = majority_vote(d.get("plate_votes", []))
        if plate and self.db.is_in_cooldown("car", plate, float(self.cfg.car.cooldown_minutes)):
            d["notified"] = True
            return
        info = None
        crop = d.get("best_crop")
        if crop is not None and self.car_analyzer is not None:
            info = self.car_analyzer.analyze(crop)
        brand = info.brand if info is not None else d.get("brand", "unknown")
        color = info.color if info is not None else d.get("color", "unknown")
        img = str(self._save_image_bytes(d.get("last_jpg", b""), "car")) if d.get("last_jpg") else ""
        plate_img = (str(self._save_image_bytes(d["plate_crop_jpg"], "plate"))
                     if d.get("plate_crop_jpg") else "")
        self.db.save_car_event(self.name, plate or "", brand, color, img, plate_img)
        if plate:
            known = plate in self.known_plates
            title = "Своя машина" if known else f"Чужая машина {plate}"
        else:
            title = "Автомобиль (номер не распознан)"
        self.notifier.send_event(title,
                                 f"Камера {self.name}\nМарка: {brand}\nЦвет: {color}\nНомер: {plate or '—'}",
                                 img,
                                 extra_images=[plate_img] if plate_img else ())
        self.recorder.trigger("car") if self.recorder else None
        d["notified"] = True

    def _read_plate(self, zone: Optional[np.ndarray]) -> str:
        """OCR номера с защитой: если движок сломался — отключаем, но камеру не роняем."""
        if self.plate_recognizer is None or zone is None:
            return ""
        try:
            plate, _ = self.plate_recognizer.read(zone)
            return plate
        except Exception as exc:
            log.error("Распознавание номеров отключено из-за ошибки: %s", exc)
            self.plate_recognizer = None
            return ""

    def _plate_from_car(self, full_frame: np.ndarray,
                        bbox: tuple) -> tuple[str, Optional[np.ndarray]]:
        """Номер и картинка номера для машины с боксом в координатах полного кадра.

        Если установлен детектор номеров (FastALPR) — ищем номер во всей области
        машины (точнее, чем эвристическая нижняя полоса). Иначе — старая схема:
        нижняя центральная полоса + tesseract/easyocr.
        """
        if self.plate_recognizer is None:
            return "", None
        try:
            if getattr(self.plate_recognizer, "uses_frame", False):
                region = PlateRecognizer.car_crop(full_frame, bbox)
                plate, _ = self.plate_recognizer.read_frame(region)
                return plate, getattr(self.plate_recognizer, "last_crop", None)
            zone = PlateRecognizer.plate_crop_from_car(full_frame, bbox)
            return self._read_plate(zone), zone
        except Exception as exc:
            log.error("Распознавание номеров отключено из-за ошибки: %s", exc)
            self.plate_recognizer = None
            return "", None

    def _process_vehicles(self, work: np.ndarray, dets, frame: Optional[np.ndarray] = None,
                          scale: float = 1.0) -> None:
        """Машины ищутся на рабочем кадре, а события сохраняются из полного."""
        if self.car_analyzer is None:
            return
        dets = list(dets or [])
        full_frame = frame if frame is not None else work
        inv = 1.0 / scale if scale > 0 else 1.0

        def full_bbox(bbox: tuple) -> tuple:
            x1, y1, x2, y2 = bbox
            return (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))

        boxes = [d.bbox for d in dets]
        vote_frames = int(self.cfg.car.get("plate_vote_frames", 5))
        if self.car_tracker is None:
            # старый путь: один кадр, без трекинга
            for det in dets:
                crop = det.crop(work)
                if crop is None or crop.size == 0:
                    continue
                info = self.car_analyzer.analyze(crop)
                bbox_full = full_bbox(det.bbox)
                plate, plate_crop = self._plate_from_car(full_frame, bbox_full)
                if plate and self.db.is_in_cooldown("car", plate, float(self.cfg.car.cooldown_minutes)):
                    continue
                full = str(self._save_image(full_frame, "car"))
                crop_img = str(self._save_image(plate_crop, "plate")) if plate_crop is not None else ""
                self.db.save_car_event(self.name, plate, info.brand, info.color, full, crop_img)
                parts = [f"Камера {self.name}", f"Марка: {info.brand}", f"Цвет: {info.color}"]
                parts.append(f"Номер: {plate}" if plate else "Номер: не распознан")
                self.notifier.send_event("Автомобиль", "\n".join(parts), full,
                                          extra_images=[crop_img] if crop_img else ())
                self.recorder.trigger("car") if self.recorder else None
            return

        tracks = self.car_tracker.update(boxes)
        for track, det in zip(tracks, dets):
            if track is None:
                continue
            d = track.data
            if d.get("notified"):
                # событие уже отправлено — не тратим CPU на OCR и вырезки
                continue
            d.setdefault("plate_votes", [])
            crop = det.crop(work)
            self._remember_best_crop(d, crop, full_frame, det.conf)
            bbox_full = full_bbox(det.bbox)
            plate, plate_crop = self._plate_from_car(full_frame, bbox_full)
            if plate and plate not in d["plate_votes"][-5:]:
                d["plate_votes"].append(plate)
                if plate_crop is not None and plate_crop.size:
                    ok_p, buf_p = cv2.imencode(".jpg", plate_crop)
                    if ok_p:
                        d["plate_crop_jpg"] = buf_p.tobytes()
            # Финализируем, когда набрали достаточно кадров...
            if track.seen >= vote_frames:
                self._finalize_car(full_frame, track)

        # ...или когда машина уехала, так и не набрав vote_frames.
        # Трекер отдаёт такие «потерянные» треки только при удалении.
        for track in self.car_tracker.pop_removed():
            if not track.data.get("notified"):
                self._finalize_car(full_frame, track)

    # ---------- отрисовка рамок (превью) ----------
    @staticmethod
    def _draw_boxes(frame, dets, scale: float = 1.0) -> None:
        inv = 1.0 / scale if scale > 0 else 1.0
        for d in dets:
            x1, y1 = int(d.x1 * inv), int(d.y1 * inv)
            x2, y2 = int(d.x2 * inv), int(d.y2 * inv)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.label} {d.conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- главный цикл ----------
    def run(self, max_frames: Optional[int] = None) -> None:
        self._ensure_models()
        force_interval = max(1, int(self.cfg.motion.get("force_interval_frames", 300)))
        detect_every = max(1, int(self.cfg.analysis.detect_every_n_frames))
        frame_scale = float(self.cfg.analysis.frame_scale)
        small_scale = float(self.cfg.analysis.get("small_frame_scale", 0.5))
        multi_res = bool(self.cfg.analysis.get("multi_resolution", True))
        preview = bool(self.cfg.ui.get("preview_enabled", False))
        win = str(self.cfg.ui.get("window_title", "MLCamera — {cam}")).format(cam=self.name)

        log.info("Поток камеры '%s' запущен (трекинг=%s)", self.name, self.tracking_enabled)
        frames = 0
        try:
            while True:
                # ночное усреднение кадров (шума меньше → номер/лицо читается в темноте)
                frame = self._read_averaged()
                if frame is None:
                    if self.cam.is_file:
                        log.info("Видеофайл закончился — останавливаю камеру '%s'", self.name)
                        break
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
                    # координаты пересчитываются обратно в полный кадр
                    scale = work.shape[1] / frame.shape[1] if frame.shape[1] else 1.0
                    dets = []
                    if bool(self.cfg.face.get("enabled", True)):
                        self._process_faces(work, frame, scale)
                    if bool(self.cfg.car.get("enabled", True)) and self.detector:
                        dets = self.detector.predict_vehicles(work)
                        self._process_vehicles(work, dets, frame, scale)
                    if preview:
                        self._draw_boxes(frame, dets, scale)

                if preview:
                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                delay = float(self.cfg.analysis.get("debug_delay", 0.0))
                if delay > 0:
                    time.sleep(delay)
                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            if self.recorder is not None:
                self.recorder.close()
            self.cam.close()
            log.info("Поток камеры '%s' остановлен (кадров: %d)", self.name, frames)

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