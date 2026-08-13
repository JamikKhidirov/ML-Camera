"""Простая детекция движения по разнице кадров.

Позволяет НЕ гонять тяжёлые модели (YOLO, лица) когда ничего не движется.
Экономит CPU на VPS и даёт больше свободы на ПК.
"""
from __future__ import annotations

import cv2


class MotionDetector:
    def __init__(self, threshold: int = 1500, scale: float = 0.25, warmup: int = 10):
        self.threshold = threshold
        self.scale = scale
        self._prev: cv2.Mat | None = None
        self._warmup = warmup
        self._count = 0

    def detect(self, frame) -> bool:
        """True, если в кадре есть движение. Всегда True первые warmup кадров."""
        self._count += 1
        small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev is None:
            self._prev = gray
            return self._count <= self._warmup
        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        changed = int(cv2.countNonZero(thresh))
        return changed > self.threshold or self._count <= self._warmup

    def reset(self) -> None:
        self._prev = None