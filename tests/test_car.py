"""Тесты определения цвета кузова."""
import cv2
import numpy as np
import pytest

from src.car_analyzer import CarAnalyzer


def _solid(color_bgr):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


@pytest.mark.parametrize("bgr,expected", [
    ((0, 0, 255), "red"),        # красный BGR=(0,0,255)
    ((255, 0, 0), "blue"),       # синий
    ((0, 255, 0), "green"),      # зелёный
    ((255, 255, 255), "white"),  # белый
    ((0, 0, 0), "black"),        # чёрный
])
def test_color_of(bgr, expected):
    result = CarAnalyzer.color_of(_solid(bgr))
    assert result == expected, f"для {bgr} получили {result}"


def test_color_unknown_on_empty():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert CarAnalyzer.color_of(img) in ("black", "gray")