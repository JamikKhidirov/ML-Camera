"""Главный конвейер: кадр -> движение -> детекция -> трекинг -> БД -> уведомления.

Новое:
  - мультиразрешение: есть движение — полный кадр, нет — мелкий (экономия CPU);
  - усреднение ночных кадров (меньше шума — номер читается в темноте);
  - трекинг людей: один проход/стояние = ОДНО уведомление (+ «вошёл/вышел»);
  - трекинг машин: номер собирается с нескольких кадров «голосованием»;
  - свои/чужие номера (car.known_plates);
  - животные (dog/cat/bird) и оставленные вещи (backpack/suitcase/handbag)
    на тех же весах YOLO: вещь считается оставленной, если трек долго стоит
    на месте (objects.* в конфиге).

Если трекинг выключен (tracking.enabled: false) — работает старый путь
(каждое появление с кулдауном по имени/номеру).
"""
from __future__ import annotations

import logging
import math
import threading
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

# Русские подписи для COCO-классов. Список можно расширить в конфиге.
ANIMAL_NAMES = {
    "dog": "Собака", "cat": "Кошка", "bird": "Птица", "horse": "Лошадь",
    "sheep": "Овца", "cow": "Корова", "elephant": "Слон", "bear": "Медведь",
    "zebra": "Зебра", "giraffe": "Жираф",
}
PACKAGE_NAMES = {
    "backpack": "Рюкзак", "suitcase": "Чемодан", "handbag": "Сумка",
}
PACKAGE_ALERTS = {
    "backpack": "Оставлен рюкзак", "suitcase": "Оставлен чемодан",
    "handbag": "Оставлена сумка",
}


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
            fps=int(cfg.camera.get("fps", 0) or 0),
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

        # ---------- животные и оставленные вещи ----------
        objects = cfg.get("objects", None)
        self.objects_cfg = objects if isinstance(objects, dict) else {}
        self.animals_cfg = self.objects_cfg.get("animals") or {}
        self.packages_cfg = self.objects_cfg.get("packages") or {}
        self.objects_enabled = bool(self.objects_cfg.get("enabled", False))
        self.animals_enabled = (self.objects_enabled
                                and bool(self.animals_cfg.get("enabled", True)))
        self.packages_enabled = (self.objects_enabled
                                 and bool(self.packages_cfg.get("enabled", True)))
        self.animal_classes = [str(x) for x in
                               (self.animals_cfg.get("classes") or ["dog", "cat", "bird"])]
        self.package_classes = [str(x) for x in
                                (self.packages_cfg.get("classes")
                                 or ["backpack", "suitcase", "handbag"])]
        self.animal_conf = float(self.animals_cfg.get("conf_threshold", 0.35))
        self.package_conf = float(self.packages_cfg.get("conf_threshold", 0.40))
        self.animal_cooldown = float(self.animals_cfg.get("cooldown_minutes", 30))
        self.package_cooldown = float(self.packages_cfg.get("cooldown_minutes", 120))
        self.animal_min_frames = int(self.animals_cfg.get(
            "min_frames", self.cfg.tracking.get("min_frames_to_notify", 3)))
        self.package_min_frames = int(self.packages_cfg.get(
            "min_frames", self.cfg.tracking.get("min_frames_to_notify", 3)))
        # Через сколько секунд неподвижности вещь считаем оставленной.
        self.stationary_seconds = float(self.packages_cfg.get("stationary_seconds", 45))
        self.stationary_move_ratio = float(
            self.packages_cfg.get("stationary_move_ratio", 0.15))
        self.stationary_min_pixels = float(
            self.packages_cfg.get("stationary_min_pixels", 8))
        self.animal_tracker = CentroidTracker(
            max_disappeared=int(self.animals_cfg.get(
                "max_disappeared", self.objects_cfg.get("max_disappeared", 20))),
            max_distance=float(self.animals_cfg.get(
                "max_distance", self.objects_cfg.get("max_distance", 100)),
        )) if self.animals_enabled else None
        self.package_tracker = CentroidTracker(
            max_disappeared=int(self.packages_cfg.get(
                "max_disappeared", self.objects_cfg.get("max_disappeared", 30))),
            max_distance=float(self.packages_cfg.get(
                "max_distance", self.objects_cfg.get("max_distance", 120)),
        )) if self.packages_enabled else None

        self._events_dir = cfg.storage.resolve(cfg.storage.events_dir)
        self._frame_count = 0
        self._auto_capture_dir = cfg.resolve(cfg.face.get("auto_capture_dir",
                                                           "data/datasets/auto_unknown"))
        # Управление из веб-панели: мягкая остановка и последний кадр для снимка.
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._last_frame_ts = 0.0
        self._last_frame_size = (0, 0)
        # Оверлей для панели: рамки объектов + «чертёж лица» по 106 точкам.
        web_cfg = cfg.get("web", None)
        web_cfg = web_cfg if isinstance(web_cfg, dict) else {}
        self.overlay_enabled = bool(web_cfg.get("overlay", True))
        self.face_mesh_enabled = bool(web_cfg.get("face_mesh", True))
        self._overlay_lock = threading.Lock()
        self._overlays: list[dict] = []
        self._object_overlays: list[dict] = []
        self._overlays_ts = 0.0
        self._face_overlays: list[dict] = []

    # ---------- оверлей ----------
    def _set_overlays(self, items: list[dict]) -> None:
        with self._overlay_lock:
            self._overlays = items
            self._overlays_ts = time.time()

    def _clear_overlays(self) -> None:
        with self._overlay_lock:
            self._overlays = []
            self._overlays_ts = 0.0

    def _draw_overlays(self, frame) -> None:
        if not self.overlay_enabled:
            return
        with self._overlay_lock:
            overlays = list(self._overlays)
            ts = self._overlays_ts
        # старые рамки не должны «прилипать» к кадру, если анализ давно не шёл
        if not overlays or time.time() - ts > 3.0:
            return
        for item in overlays:
            x1, y1, x2, y2 = (int(v) for v in item["bbox"])
            color = item.get("color", (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = str(item.get("label", ""))
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(0, y1 - th - 8)
                cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 7),
                              color, -1)
                cv2.putText(frame, label, (x1 + 4, top + th + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 12, 20), 1,
                            cv2.LINE_AA)
            if self.face_mesh_enabled:
                self._draw_face_mesh(frame, item, color)
        return

    @staticmethod
    def _draw_face_mesh(frame, item: dict, color) -> None:
        """Точки 106-точечной модели + контур лица — «чертёж лица»."""
        points = item.get("landmarks")
        if points is None:
            return
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        if not len(pts):
            return
        try:
            hull = cv2.convexHull(pts)
            cv2.polylines(frame, [hull], True, color, 1, cv2.LINE_AA)
        except cv2.error:
            pass
        for px, py in pts:
            cv2.circle(frame, (int(px), int(py)), 1, color, -1, cv2.LINE_AA)
        kps = item.get("kps")
        if kps is not None:
            for px, py in np.asarray(kps, dtype=np.int32).reshape(-1, 2):
                cv2.circle(frame, (int(px), int(py)), 3, (255, 255, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(frame, (int(px), int(py)), 4, color, 1, cv2.LINE_AA)

    # ---------- управление из веб-панели ----------
    @property
    def frames_processed(self) -> int:
        return self._frame_count

    @property
    def last_frame_ts(self) -> float:
        return self._last_frame_ts

    @property
    def frame_size(self) -> tuple[int, int]:
        """(ширина, высота) последнего кадра — для карточки камеры."""
        return self._last_frame_size

    def stop(self) -> None:
        """Попросить главный цикл завершиться (безопасно из другого потока)."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def prepare_start(self) -> None:
        """Переоткрыть источник после stop(): модели и настройки остаются в памяти."""
        self._stop.clear()
        try:
            self.cam.close()
        except Exception:
            pass
        self.cam = Camera(
            self.camera_cfg["source"],
            width=int(self.cfg.camera.width),
            height=int(self.cfg.camera.height),
            fps=int(self.cfg.camera.get("fps", 0) or 0),
        )
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:
                pass
            self.recorder = ClipRecorder(self.cfg, self.name, source_fps=self.cam.fps)

    def snapshot_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """JPEG последнего кадра (или None, если камера ещё не дала кадров)."""
        with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    # ---------- имя для уведомлений ----------
    def _display_name(self, key: str) -> str:
        names = self.cfg.people.get("names", {})
        return names.get(key, key.capitalize())

    # ---------- ленивая загрузка моделей ----------
    def _ensure_models(self) -> None:
        car_enabled = bool(self.cfg.car.get("enabled", True))
        need_detector = car_enabled or self.objects_enabled
        if self.detector is None and need_detector:
            conf = float(self.cfg.car.get("conf_threshold", 0.35))
            if self.objects_enabled:
                conf = min(conf, self.animal_conf if self.animals_enabled else conf,
                           self.package_conf if self.packages_enabled else conf)
            self.detector = YoloDetector(self.cfg, conf_threshold=conf)
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

        def scaled(points):
            if points is None:
                return None
            return np.asarray(points, dtype=np.float32).reshape(-1, 2) * inv

        unknown_label = str(self.cfg.people.get("unknown_label", "Незнакомец"))
        self._face_overlays = []
        for m in matches:
            if m.unknown:
                label = unknown_label
                color = (80, 150, 255)  # BGR: оранжевый
            else:
                label = f"{m.name} {m.similarity:.2f}"
                color = (80, 255, 170)  # BGR: зелёный
            self._face_overlays.append({
                "bbox": full_bbox(m.bbox),
                "label": label,
                "color": color,
                "landmarks": scaled(m.landmarks),
                "kps": scaled(m.kps),
            })

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

    # ---------- ЖИВОТНЫЕ И ОСТАВЛЕННЫЕ ВЕЩИ ----------
    def _object_conf_floor(self) -> float:
        values = []
        if self.animals_enabled:
            values.append(self.animal_conf)
        if self.packages_enabled:
            values.append(self.package_conf)
        return min(values) if values else 1.0

    def _image_from_track(self, frame, data: dict, prefix: str) -> str:
        """Фото события: готовый JPEG трека или текущий полный кадр."""
        jpg = data.get("last_jpg")
        if jpg:
            return self._save_image_bytes(jpg, prefix)
        return self._save_image(frame, prefix)

    def _crop_from_track(self, data: dict, prefix: str) -> str:
        crop = data.get("best_crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            return ""
        ok, buf = cv2.imencode(".jpg", crop)
        return self._save_image_bytes(buf.tobytes(), prefix) if ok else ""

    @staticmethod
    def _full_bbox(bbox: tuple, scale: float) -> tuple:
        inv = 1.0 / scale if scale > 0 else 1.0
        x1, y1, x2, y2 = bbox
        return (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))

    def _update_stationary(self, data: dict, bbox: tuple, now: float) -> float:
        """Сколько секунд объект стоит на месте.

        Порог движения зависит от размера объекта: мелкая сумка не должна
        «дрожать» на каждом пикселе, крупный чемодан может сместиться сильнее.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = max(1.0, float(max(x2 - x1, y2 - y1)))
        threshold = max(self.stationary_min_pixels, size * self.stationary_move_ratio)
        ref = data.get("still_ref")
        if ref is None or math.hypot(cx - ref[0], cy - ref[1]) > threshold:
            data["still_ref"] = (cx, cy)
            data["still_since"] = now
        return max(0.0, now - float(data.get("still_since", now)))

    def _announce_animal(self, frame, track) -> None:
        d = track.data
        label = majority_vote(list(d.get("label_votes", []))) or d.get("label", "animal")
        if self.db.is_object_in_cooldown("animal", label, self.animal_cooldown):
            d["notified"] = True
            return
        display = ANIMAL_NAMES.get(label, label)
        conf = float(d.get("conf", 0.0))
        img = self._image_from_track(frame, d, f"animal_{label}")
        crop_img = self._crop_from_track(d, f"animal_{label}_crop")
        title = f"{display} в кадре"
        self.db.save_object_event(
            self.name, "animal", label, display, conf, 0.0, img, crop_img)
        self.notifier.send_event(
            title,
            f"Камера {self.name}\nКласс: {label}\nУверенность: {conf:.2f}",
            img, extra_images=[crop_img] if crop_img else (),
        )
        if self.recorder is not None:
            self.recorder.trigger("animal")
        d["notified"] = True

    def _announce_package(self, frame, track) -> None:
        d = track.data
        label = majority_vote(list(d.get("label_votes", []))) or d.get("label", "package")
        if self.db.is_object_in_cooldown("package", label, self.package_cooldown):
            d["notified"] = True
            return
        display = PACKAGE_NAMES.get(label, "Вещь")
        still = float(d.get("still_seconds", 0.0))
        conf = float(d.get("conf", 0.0))
        img = self._image_from_track(frame, d, f"package_{label}")
        crop_img = self._crop_from_track(d, f"package_{label}_crop")
        title = PACKAGE_ALERTS.get(label, "Оставленная вещь")
        self.db.save_object_event(
            self.name, "package", label, display, conf, still, img, crop_img)
        self.notifier.send_event(
            title,
            (f"Камера {self.name}\nОбъект: {display} ({label})\n"
             f"Без движения: {int(still)} с\nУверенность: {conf:.2f}"),
            img, extra_images=[crop_img] if crop_img else (),
        )
        if self.recorder is not None:
            self.recorder.trigger("package")
        d["notified"] = True

    def _process_objects(self, work: np.ndarray, frame: Optional[np.ndarray] = None,
                         scale: float = 1.0, dets=None) -> list:
        """Животные и вещи на том же кадре YOLO.

        Возвращает найденные детекции — для отрисовки в превью.
        """
        if not self.objects_enabled:
            return []
        full_frame = frame if frame is not None else work
        if dets is None:
            if self.detector is None:
                return []
            dets = self.detector.predict(work, conf_threshold=self._object_conf_floor())
        dets = list(dets or [])
        animals = [d for d in dets
                   if d.label in self.animal_classes and d.conf >= self.animal_conf]
        packages = [d for d in dets
                    if d.label in self.package_classes and d.conf >= self.package_conf]
        found = animals + packages

        # ----- животные -----
        if self.animals_enabled:
            if self.animal_tracker is None:
                for det in animals:
                    stub = Track_stub()
                    stub.data.update({"label_votes": [det.label], "conf": det.conf,
                                      "bbox": self._full_bbox(det.bbox, scale)})
                    self._announce_animal(full_frame, stub)
            else:
                tracks = self.animal_tracker.update([d.bbox for d in animals])
                for track, det in zip(tracks, animals):
                    if track is None:
                        continue
                    d = track.data
                    d.setdefault("label_votes", []).append(det.label)
                    d["conf"] = max(float(d.get("conf", 0.0)), det.conf)
                    d["bbox"] = self._full_bbox(det.bbox, scale)
                    self._remember_best_crop(d, det.crop(work), full_frame, det.conf)
                    if d.get("notified"):
                        continue
                    if track.seen >= self.animal_min_frames:
                        self._announce_animal(full_frame, track)
                for track in self.animal_tracker.pop_removed():
                    if not track.data.get("notified") and track.seen >= self.animal_min_frames:
                        self._announce_animal(full_frame, track)

        # ----- оставленные вещи -----
        if self.packages_enabled:
            if self.package_tracker is None:
                for det in packages:
                    stub = Track_stub()
                    stub.data.update({"label_votes": [det.label], "conf": det.conf,
                                      "bbox": self._full_bbox(det.bbox, scale)})
                    self._announce_package(full_frame, stub)
            else:
                now = time.time()
                tracks = self.package_tracker.update([d.bbox for d in packages])
                for track, det in zip(tracks, packages):
                    if track is None:
                        continue
                    d = track.data
                    d.setdefault("label_votes", []).append(det.label)
                    d["conf"] = max(float(d.get("conf", 0.0)), det.conf)
                    bbox_full = self._full_bbox(det.bbox, scale)
                    d["bbox"] = bbox_full
                    self._remember_best_crop(d, det.crop(work), full_frame, det.conf)
                    d["still_seconds"] = self._update_stationary(d, bbox_full, now)
                    if d.get("notified"):
                        continue
                    if (track.seen >= self.package_min_frames
                            and d["still_seconds"] >= self.stationary_seconds):
                        self._announce_package(full_frame, track)
                for track in self.package_tracker.pop_removed():
                    d = track.data
                    if (not d.get("notified")
                            and track.seen >= self.package_min_frames
                            and float(d.get("still_seconds", 0.0)) >= self.stationary_seconds):
                        self._announce_package(full_frame, track)

        return found

    # ---------- отрисовка рамок (превью) ----------
    _BOX_COLORS = {
        "car": (0, 255, 0), "truck": (0, 255, 0), "bus": (0, 255, 0),
        "motorcycle": (0, 255, 0),
        "dog": (0, 210, 255), "cat": (0, 210, 255), "bird": (0, 210, 255),
        "backpack": (255, 150, 0), "suitcase": (255, 150, 0), "handbag": (255, 150, 0),
    }

    @classmethod
    def _draw_boxes(cls, frame, dets, scale: float = 1.0) -> None:
        inv = 1.0 / scale if scale > 0 else 1.0
        for d in dets:
            x1, y1 = int(d.x1 * inv), int(d.y1 * inv)
            x2, y2 = int(d.x2 * inv), int(d.y2 * inv)
            color = cls._BOX_COLORS.get(d.label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d.label} {d.conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
            while not self._stop.is_set():
                # ночное усреднение кадров (шума меньше → номер/лицо читается в темноте)
                frame = self._read_averaged()
                if frame is None:
                    if self.cam.is_file:
                        log.info("Видеофайл закончился — останавливаю камеру '%s'", self.name)
                        break
                    if max_frames is not None:
                        break
                    self._stop.wait(1.0)
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
                face_on = bool(self.cfg.face.get("enabled", True))
                face_every = max(1, int(self.cfg.face.get("mesh_every_n_frames", 8)))
                face_hit = (face_on and self.overlay_enabled and self.face_mesh_enabled
                            and self._frame_count % face_every == 0)

                analyze = False
                face_only = False
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
                elif face_hit:
                    # «Чертёж лица» обновляем и без движения — человек сидит
                    # неподвижно, а точки на лице должны оставаться на месте.
                    analyze = True
                    face_only = True
                    if frame_scale < 1.0:
                        h, w = frame.shape[:2]
                        work = cv2.resize(frame, (int(w * frame_scale), int(h * frame_scale)))

                if analyze:
                    # координаты пересчитываются обратно в полный кадр
                    scale = work.shape[1] / frame.shape[1] if frame.shape[1] else 1.0
                    dets: list = []
                    obj_dets: list = []
                    car_enabled = bool(self.cfg.car.get("enabled", True))
                    self._face_overlays = []
                    if face_on:
                        self._process_faces(work, frame, scale)
                    if face_only:
                        # без YOLO: только лица, рамки объектов оставляем с прошлого прохода
                        self._set_overlays(self._object_overlays + self._face_overlays)
                    else:
                        # Один прогон YOLO на кадр: машины, животные и вещи
                        # фильтруются из общего результата (без повторной инференции).
                        if self.detector is not None and (car_enabled or self.objects_enabled):
                            conf = (float(self.cfg.car.get("conf_threshold", 0.35))
                                    if car_enabled else self._object_conf_floor())
                            if self.objects_enabled:
                                conf = min(conf, self._object_conf_floor())
                            all_dets = self.detector.predict(work, conf_threshold=conf)
                            if car_enabled:
                                allowed = self.cfg.car.get(
                                    "detect_classes", ["car", "truck", "bus", "motorcycle"])
                                dets = [d for d in all_dets if d.label in allowed]
                                self._process_vehicles(work, dets, frame, scale)
                            if self.objects_enabled:
                                obj_dets = self._process_objects(work, frame, scale, all_dets)
                        # Рамки и «чертёж лица» — и для окна, и для веб-панели.
                        inv = 1.0 / scale if scale > 0 else 1.0
                        overlays = []
                        for d in dets + obj_dets:
                            overlays.append({
                                "bbox": (int(d.x1 * inv), int(d.y1 * inv),
                                         int(d.x2 * inv), int(d.y2 * inv)),
                                "label": f"{d.label} {d.conf:.2f}",
                                "color": self._BOX_COLORS.get(d.label, (0, 255, 0)),
                            })
                        self._object_overlays = overlays
                        overlays.extend(self._face_overlays)
                        self._set_overlays(overlays)

                self._draw_overlays(frame)

                # последний кадр — для кнопки «Снимок» в веб-панели
                with self._frame_lock:
                    self._last_frame = frame
                    self._last_frame_ts = time.time()
                    self._last_frame_size = (int(frame.shape[1]), int(frame.shape[0]))

                if preview:
                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                delay = float(self.cfg.analysis.get("debug_delay", 0.0))
                if delay > 0:
                    self._stop.wait(delay)
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