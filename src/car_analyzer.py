"""Анализ автомобиля: цвет и (опционально) марка.

Цвет — по гистограмме HSV вырезки кузова (без колёс/тени).
Марка — готовая CNN (ConvNeXt-Tiny, обученная на Stanford Cars, 49 марок).

Модель и подписи классов лежат в data/models/:
  car_brand.onnx        — ONNX-модель (скачать: python scripts/download_car_brand_model.py)
  car_brand_labels.json — список марок в порядке выходов модели
  car_brand_meta.json   — (необязательно) размер входа и нормализация

Если метаданных нет, используется старый формат (224x224, нормализация в [-1, 1]) —
он совместим с моделями, обученными scripts/train_brand_classifier.py.
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

# нормализация ImageNet (так обучался ConvNeXt на Stanford Cars)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CarInfo:
    color: str = "unknown"
    brand: str = "unknown"
    plate: str = ""
    brand_confidence: float = 0.0


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
        self._mean = np.array((0.5, 0.5, 0.5), dtype=np.float32)
        self._std = np.array((0.5, 0.5, 0.5), dtype=np.float32)
        self._center_crop = False
        self._resize_ratio = 1.0
        self._tta = bool(cfg.car.get("brand_tta", True))
        self._min_conf = float(cfg.car.get("brand_min_conf", 0.0))
        self._load_brand_model()

    # ---------- модель марки ----------
    def _load_brand_model(self) -> None:
        onnx_path = self.cfg.models.resolve(self.cfg.models.brand_onnx)
        labels_path = self.cfg.models.resolve(self.cfg.models.brand_labels)
        if not (onnx_path.exists() and labels_path.exists()):
            log.info("Модель марки не найдена (%s). Марка будет 'unknown'. "
                     "Скачай готовую: python scripts/download_car_brand_model.py", onnx_path)
            return
        try:
            self._ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._input_name = self._ort_session.get_inputs()[0].name
            shape = self._ort_session.get_inputs()[0].shape
            self._input_size = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 224
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, dict):
                labels = labels.get("classes", labels.get("labels", []))
            self._labels = [str(x) for x in labels]

            meta_path = onnx_path.parent / "car_brand_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._input_size = int(meta.get("input_size", self._input_size))
                self._mean = np.array(meta.get("mean", _IMAGENET_MEAN), dtype=np.float32)
                self._std = np.array(meta.get("std", _IMAGENET_STD), dtype=np.float32)
                self._center_crop = meta.get("resize", "shortest_center_crop") == "shortest_center_crop"
                # при обучении было Resize(size*1.14) -> CenterCrop(size)
                self._resize_ratio = float(meta.get("resize_ratio",
                                                    1.14 if self._center_crop else 1.0))
            log.info("Модель марки готова: %d марок, вход %dx%d%s",
                     len(self._labels), self._input_size, self._input_size,
                     " (ImageNet-нормализация)" if self._center_crop else "")
        except Exception as exc:
            log.error("Не удалось загрузить модель марки: %s", exc)
            self._ort_session = None
            self._labels = []

    # ---------- цвет ----------
    @staticmethod
    def color_of(car_crop: np.ndarray) -> str:
        hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        # берём центральную зону кузова (без колёс снизу)
        hh, ww = hsv.shape[:2]
        zone = hsv[int(hh * 0.25):int(hh * 0.85), int(ww * 0.2):int(ww * 0.8)]
        if zone.size == 0:
            zone = hsv
        best, best_count = "unknown", -1
        for name, lo, hi in _COLOR_RULES:
            mask = cv2.inRange(zone, np.array(lo), np.array(hi))
            cnt = int(mask.sum() / 255)
            if cnt > best_count:
                best, best_count = name, cnt
        return best

    # ---------- марка ----------
    def _preprocess(self, car_crop: np.ndarray) -> np.ndarray:
        size = self._input_size
        img = car_crop
        if self._center_crop:
            h, w = img.shape[:2]
            if min(h, w) < 2:
                raise ValueError("слишком маленькая вырезка")
            base = max(size, int(round(size * self._resize_ratio)))
            scale = base / min(h, w)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            img = cv2.resize(img, (max(size, int(round(w * scale))), max(size, int(round(h * scale)))),
                             interpolation=interp)
            h, w = img.shape[:2]
            top, left = (h - size) // 2, (w - size) // 2
            img = img[top:top + size, left:left + size]
        else:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    def _run(self, x: np.ndarray) -> np.ndarray:
        out = self._ort_session.run(None, {self._input_name: x})[0]
        return np.asarray(out).reshape(-1)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        e = np.exp(x)
        return e / max(float(e.sum()), 1e-12)

    def classify_brand(self, car_crop: np.ndarray) -> tuple[str, float]:
        """Возвращает (марка, уверенность). Марка "unknown", если модель не уверена."""
        if self._ort_session is None or not self._labels or car_crop is None or car_crop.size == 0:
            return "unknown", 0.0
        try:
            x = self._preprocess(car_crop)
            probs = self._softmax(self._run(x))
            if self._tta:
                # горизонтальное отражение усредняет предсказание (как TTA при обучении)
                probs = (probs + self._softmax(self._run(x[:, :, :, ::-1].copy()))) / 2.0
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf < self._min_conf:
                return "unknown", conf
            return (self._labels[idx], conf) if 0 <= idx < len(self._labels) else ("unknown", conf)
        except Exception as exc:
            log.warning("Ошибка определения марки: %s", exc)
            return "unknown", 0.0

    def brand_of(self, car_crop: np.ndarray) -> str:
        return self.classify_brand(car_crop)[0]

    def analyze(self, car_crop: np.ndarray) -> CarInfo:
        info = CarInfo(color="unknown", brand="unknown")
        if car_crop is None or car_crop.size == 0:
            return info
        if self.cfg.car.color_enabled:
            info.color = self.color_of(car_crop)
        if self.cfg.car.brand_enabled:
            info.brand, info.brand_confidence = self.classify_brand(car_crop)
        return info
