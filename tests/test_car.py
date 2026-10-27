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


def test_ready_brand_model_loads_if_present():
    """Готовая модель марки (если скачана) грузится и даёт валидный класс."""
    from src.config import load_config

    cfg = load_config("config/config.local.yaml")
    analyzer = CarAnalyzer(cfg)
    if analyzer._ort_session is None:
        pytest.skip("модель марки не скачана (scripts/download_car_brand_model.py)")

    img = np.full((288, 480, 3), 128, dtype=np.uint8)
    brand, conf = analyzer.classify_brand(img)
    # на однотонном кадре модель может честно ответить "unknown" (низкая уверенность)
    assert brand == "unknown" or brand in analyzer._labels
    assert 0.0 <= conf <= 1.0
    # вход модели — 288x288, как обещано в training_info.json
    assert analyzer._input_size == 288
    assert analyzer._preprocess(img).shape == (1, 3, 288, 288)
