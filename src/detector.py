"""Детекция объектов через YOLOv8 (Ultralytics).

Находит людей и автомобили в кадре. Классы YOLO (по умолчанию):
  0 person | 2 car | 3 motorcycle | 5 bus | 7 truck
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import Config

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str          # person / car / truck / ...
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    bbox: tuple = field(init=False)

    def __post_init__(self):
        self.bbox = (self.x1, self.y1, self.x2, self.y2)

    def crop(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, self.x1), max(0, self.y1), min(w, self.x2), min(h, self.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]


class YoloDetector:
    """Обёртка над YOLOv8n. Модель скачивается автоматически при первом запуске."""

    def __init__(self, cfg: Config, conf_threshold: float = 0.35):
        from ultralytics import YOLO  # импорт здесь, чтобы не замедлять старт остального
        self.cfg = cfg
        self.conf_threshold = conf_threshold
        model_path = cfg.models.resolve(cfg.models.yolov8)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            log.info("Модель YOLO не найдена, скачиваю программе... %s", model_path)
        self._model = YOLO(str(model_path))
        self.names = self._model.names
        log.info("YOLO загружена: %d классов", len(self.names))

    def predict(self, frame, conf_threshold: float | None = None) -> list[Detection]:
        conf = conf_threshold or self.conf_threshold
        results = self._model.predict(
            frame, conf=conf, verbose=False, device="cpu"
        )
        dets: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                label = self.names[int(box.cls[0])]
                dets.append(Detection(
                    label=label,
                    conf=float(box.conf[0]),
                    x1=int(box.xyxy[0][0]), y1=int(box.xyxy[0][1]),
                    x2=int(box.xyxy[0][2]), y2=int(box.xyxy[0][3]),
                ))
        return dets

    def predict_persons(self, frame, conf_threshold: float | None = None) -> list[Detection]:
        return [d for d in self.predict(frame, conf_threshold) if d.label == "person"]

    def predict_vehicles(self, frame, allowed: list[str] | None = None,
                         conf_threshold: float | None = None) -> list[Detection]:
        allowed = allowed or self.cfg.car.detect_classes
        return [d for d in self.predict(frame, conf_threshold) if d.label in allowed]