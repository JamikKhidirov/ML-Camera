"""Улучшение ночных кадров (яркость, контраст, шум).

Камера с ИК-подсветкой ночью отдаёт почти чёрно-белый тёмный кадр.
Выравнивание гистограммы (CLAHE) и подавление шума сильно повышают
шансы распознать лицо и прочитать номер в темноте.
"""
from __future__ import annotations

import cv2


def is_low_light(frame, threshold: int = 60) -> bool:
    """True, если средняя яркость кадра ниже порога (значит ночь/темно)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def enhance_low_light(frame):
    """Усиление тёмного кадра: CLAHE + авто-гамма + шумоподавление."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    # мягкое повышение яркости
    l = cv2.convertScaleAbs(l, alpha=1.15, beta=10)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # сохраняем цвет хотя бы частично: чуть повышаем насыщенность (если есть цвет)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=1.2)
    out = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    # шум ИК-съёмки
    out = cv2.bilateralFilter(out, d=5, sigmaColor=20, sigmaSpace=20)
    return out


def ensure_night_quality(frame, threshold: int = 60):
    """Простой интерфейс: если кадр тёмный — усилить и вернуть (True, кадр)."""
    if is_low_light(frame, threshold):
        return True, enhance_low_light(frame)
    return False, frame