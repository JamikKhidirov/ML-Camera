"""Скелет тела по 17 точкам COCO (YOLO pose).

Зачем отдельный модуль:
  - скелет лица показывает только лицо, а человеку нужны ещё плечи, руки,
    ноги и стопы;
  - YOLOv8-pose за один проход находит человека и 17 ключевых точек, поэтому
    не нужно подключать тяжёлый MediaPipe и ещё один формат модели;
  - отрисовка отделена от модели: её можно использовать в Tkinter, веб-панели
    и тестах без загрузки весов.

Индексы COCO-17 (как в ultralytics):
  0 нос, 1 левый глаз, 2 правый глаз, 3 левое ухо, 4 правое ухо,
  5 левое плечо, 6 правое плечо, 7 левый локоть, 8 правый локоть,
  9 левое запястье, 10 правое запястье, 11 левый тазобедренный сустав,
  12 правый тазобедренный, 13 левое колено, 14 правое колено,
  15 левая стопа, 16 правая стопа.

«Лево/право» — со стороны человека, а не зрителя. Для зеркального кадра
координаты просто отражаются по X, менять индексы не нужно.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "data/models/yolov8n-pose.pt"
POSE_COLOR = (255, 210, 90)      # BGR: светло-голубой, чтобы не путать с лицом
POSE_SPINE_COLOR = (190, 140, 60)
KEYPOINT_COUNT = 17

# Связи стандартного скелета COCO (0-индексация).
POSE_EDGES: tuple[tuple[int, int], ...] = (
    # голова
    (0, 1), (0, 2), (1, 3), (2, 4), (1, 2),
    # плечи и руки
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (3, 5), (4, 6),
    # корпус
    (5, 11), (6, 12), (11, 12),
    # ноги
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass
class PoseOverlay:
    """Один человек в координатах полного кадра."""

    bbox: tuple[int, int, int, int]
    keypoints: np.ndarray                 # (17, 2) float32
    confidence: np.ndarray                # (17,) float32
    score: float = 0.0


def _to_numpy(value) -> Optional[np.ndarray]:
    """torch.Tensor / список / numpy → float32-массив (или None)."""
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class PoseEstimator:
    """Ленивая обёртка над YOLOv8-pose.

    Модель грузится при первом ``detect``: окно успевает показать видео,
    пока веса читаются с диска. Если ultralytics или веса недоступны,
    ``detect`` возвращает пустой список, а причина лежит в ``error``.
    """

    def __init__(self, cfg, model_path: str | Path | None = None,
                 conf: float | None = None, imgsz: int | None = None):
        self.cfg = cfg
        preview = cfg.get("preview", {}) or {}
        models = cfg.get("models", {}) or {}
        if conf is None:
            conf = preview.get("pose_conf", 0.35)
        if imgsz is None:
            imgsz = preview.get("pose_imgsz", 640)
        raw = str(model_path or models.get("pose", DEFAULT_MODEL)
                  or DEFAULT_MODEL)
        if model_path is None:
            try:
                self.path = cfg.resolve(raw)
            except Exception:
                self.path = Path(raw)
        else:
            self.path = Path(raw)
        self.conf = _clamp(conf, 0.05, 0.95)
        self.imgsz = max(160, int(imgsz))
        self._model = None
        self._error = ""

    @property
    def error(self) -> str:
        return self._error

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self):
        """Загрузить веса. Бросает исключение, если модель недоступна."""
        if self._model is not None:
            return self._model
        from ultralytics import YOLO
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(self.path))
        log.info("YOLO pose готова: %s", self.path)
        return self._model

    def detect(self, frame) -> list[PoseOverlay]:
        """Найти людей и их 17 точек. Никогда не роняет поток камеры."""
        if frame is None:
            return []
        try:
            height, width = frame.shape[:2]
        except (AttributeError, ValueError):
            return []
        if height <= 0 or width <= 0:
            return []
        try:
            model = self.load()
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if message != self._error:
                log.warning("Модель позы недоступна: %s", message)
            self._error = message
            return []
        try:
            results = model(frame, conf=self.conf, imgsz=self.imgsz,
                            verbose=False)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if message != self._error:
                log.warning("Ошибка инференса позы: %s", message)
            self._error = message
            return []
        self._error = ""
        if not results:
            return []
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return []
        xy = _to_numpy(getattr(keypoints, "xy", None))
        if xy is None or xy.size == 0:
            return []
        if xy.ndim == 2:
            xy = xy[None, ...]
        conf = _to_numpy(getattr(keypoints, "conf", None))
        if conf is None:
            conf = np.ones(xy.shape[:2], dtype=np.float32)
        if conf.ndim == 1:
            conf = conf[None, ...]

        boxes = getattr(result, "boxes", None)
        bboxes = _to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else None
        scores = _to_numpy(getattr(boxes, "conf", None)) if boxes is not None else None

        overlays: list[PoseOverlay] = []
        for index in range(xy.shape[0]):
            points = xy[index]
            if points.shape[0] < KEYPOINT_COUNT:
                continue
            points = points[:KEYPOINT_COUNT].astype(np.float32)
            point_conf = conf[index][:KEYPOINT_COUNT].astype(np.float32)
            if bboxes is not None and index < len(bboxes):
                x1, y1, x2, y2 = (int(v) for v in bboxes[index][:4])
            else:
                visible = points[point_conf >= self.conf]
                if len(visible) == 0:
                    continue
                x1, y1 = (int(v) for v in visible.min(axis=0))
                x2, y2 = (int(v) for v in visible.max(axis=0))
            score = (float(scores[index])
                     if scores is not None and index < len(scores) else 0.0)
            overlays.append(PoseOverlay(
                bbox=(x1, y1, x2, y2),
                keypoints=points,
                confidence=point_conf,
                score=score,
            ))
        return overlays


def _visible_points(item: dict, min_conf: float):
    points = _to_numpy(item.get("keypoints"))
    conf = _to_numpy(item.get("confidence"))
    if points is None:
        return None, None
    points = points.reshape(-1, 2)
    if conf is None:
        conf = np.ones(len(points), dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    if len(conf) < len(points):
        conf = np.pad(conf, (0, len(points) - len(conf)), constant_values=1.0)
    return points, conf[:len(points)]


def draw_pose(frame: np.ndarray, item: dict,
              color: tuple[int, int, int] = POSE_COLOR,
              thickness: int = 1,
              min_conf: float = 0.3) -> int:
    """Нарисовать скелет тела. Возвращает число линий (для тестов)."""
    points, conf = _visible_points(item or {}, min_conf)
    if points is None:
        return 0
    count = min(len(points), len(conf))
    points, conf = points[:count], conf[:count]

    bbox = (item or {}).get("bbox")
    radius = 2
    if bbox and len(bbox) == 4:
        height = max(1, int(bbox[3]) - int(bbox[1]))
        radius = max(2, min(5, int(height * 0.012)))
    radius = max(1, radius if thickness <= 1 else radius + 1)

    ok = conf >= float(min_conf)
    ok &= np.isfinite(points).all(axis=1)
    ok &= points[:, 0] > 0
    ok &= points[:, 1] > 0

    drawn = 0
    line_color = tuple(int(c * 0.85) for c in color)
    for i, j in POSE_EDGES:
        if i >= count or j >= count or not (ok[i] and ok[j]):
            continue
        p1 = (int(points[i][0]), int(points[i][1]))
        p2 = (int(points[j][0]), int(points[j][1]))
        cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)
        drawn += 1

    # Позвоночник: середина плеч → середина таза и шея → нос.
    if count > 12 and ok[5] and ok[6]:
        neck = (points[5] + points[6]) / 2.0
        if ok[11] and ok[12]:
            hip = (points[11] + points[12]) / 2.0
            cv2.line(frame, (int(neck[0]), int(neck[1])),
                     (int(hip[0]), int(hip[1])),
                     POSE_SPINE_COLOR, thickness, cv2.LINE_AA)
            drawn += 1
        if ok[0]:
            cv2.line(frame, (int(neck[0]), int(neck[1])),
                     (int(points[0][0]), int(points[0][1])),
                     POSE_SPINE_COLOR, thickness, cv2.LINE_AA)
            drawn += 1

    for index in range(count):
        if not ok[index]:
            continue
        center = (int(points[index][0]), int(points[index][1]))
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius + 1, (20, 30, 45), 1, cv2.LINE_AA)
    return drawn


__all__ = [
    "DEFAULT_MODEL", "POSE_COLOR", "POSE_SPINE_COLOR", "POSE_EDGES",
    "KEYPOINT_COUNT", "PoseOverlay", "PoseEstimator", "draw_pose",
]
