"""Анализ автомобиля: цвет и (опционально) марка.

Цвет — по гистограмме HSV вырезки кузова (без колёс/тени).
Марка — лёгкая CNN (MobileNetV3), обученная scripts/train_brand_classifier.py
на наборе Stanford Cars. Пока модель не обучена — возвращается "unknown".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.config import Config

log = logging.getLogger(__name__)


@dataclass
class CarInfo:
    color: str = "unknown"
    brand: str = "unknown"
    plate: str = ""


# цветовые правила в HSV: (имя, нижняя граница, верхняя граница)
_COLOR_RULES = [
    ("red",             (0, 90, 60),    (8, 255, 255)),
    ("red",             (170, 90, 60),  (180, 255, 255)),
    ("orange",          (9, 90, 60),    (20, 255, 255)),
    ("yellow",          (21, 90, 60),   (34, 255, 255)),
    ("green",           (35, 60, 50),   (85, 255, 255)),
    ("cyan",            (86, 60, 50),   (97, 255, 255)),
    ("blue",            (98, 60, 50),   (130, 255, 255)),
    ("purple",          (131, 60, 50),  (155, 255, 255)),
    ("pink",            (156, 60, 80),  (169, 255, 255)),
    ("brown",           (10, 40, 30),   (25, 220, 140)),
    ("white",           (0, 0, 180),    (180, 60, 255)),
    ("gray",            (0, 0, 60),     (180, 60, 180)),
    ("black",           (0, 0, 0),      (180, 60, 60)),
]


class CarAnalyzer:
    """Классификация цвета и марки автомобиля."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ort_session = None
        self._labels: list[str] = []
        self._input_name: Optional[str] = None
        self._input_size = 224
        self._load_brand_model()

    # ---------- модель марки ----------
    def _load_brand_model(self) -> None:
        onnx_path = self.cfg.models.resolve(self.cfg.models.brand_onnx)
        labels_path = self.cfg.models.resolve(self.cfg.models.brand_labels)
        if not (onnx_path.exists() and labels_path.exists()):
            log.info("Модель марки не найдена (%s). Марка будет 'unknown'. "
                     "Обучи: python scripts/train_brand_classifier.py", onnx_path)
            return
        try:
            self._ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._input_name = self._ort_session.get_inputs()[0].name
            shape = self._ort_session.get_inputs()[0].shape
            self._input_size = int(shape[2]) if len(shape) == 4 and shape[2] else 224
            with open(labels_path, "r", encoding="utf-8") as f:
                self._labels = json.load(f)
            log.info("Модель марки готова, %d марок", len(self._labels))
        except Exception as exc:
            log.error("Не удалось загрузить модель марки: %s", exc)

    # ---------- цвет ----------
    @staticmethod
    def color_of(car_crop: np.ndarray) -> str:
        hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        # берём центральную зону кузова (без колёс снизу)
        hh, ww = hsv.shape[:2]
        zone = hsv[int(hh * 0.25):int(hh * 0.85), int(ww * 0.2):int(ww * 0.8)]
        # gather pixel mask to avoid scroll copies issues
        safe = zone.reshape(-1, 3)
        best, best_count = "unknown", -1
        for name, lo, hi in _COLOR_RULES:
            mask = cv2.inRange(zone, np.array(lo), np.array(hi))
            cnt = int(mask.sum() / 255)
            if cnt > best_count:
                best, best_count = name, cnt
        return best

    # ---------- марка ----------
    def brand_of(self, car_crop: np.ndarray) -> str:
        if self._ort_session is None or not self._labels:
            return "unknown"
        try:
            img = cv2.resize(car_crop, (self._input_size, self._input_size))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = np.clip((rgb - 0.5) / 0.5, -1, 1)
            x = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0).astype(np.float32)
            out = self._ort_session.run(None, {self._input_name: x})[0]
            idx = int(np.argmax(out[0]))
            return self._labels[idx] if 0 <= idx < len(self._labels) else "unknown"
        except Exception as exc:
            log.warning("Ошибка марки: %s", exc)
            return "unknown"

    def analyze(self, car_crop: np.ndarray) -> CarInfo:
        info = CarInfo(color="unknown", brand="unknown")
        if car_crop is None or car_crop.size == 0:
            return info
        if self.cfg.car.color_enabled:
            info.color = self.color_of(car_crop)
        if self.cfg.car.brand_enabled:
            info.brand = self.brand_of(car_crop)
        return info