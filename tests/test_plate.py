"""Тесты распознавания номеров (без камеры)."""
import numpy as np
import pytest

from src.plate_recognizer import normalize_plate


@pytest.mark.parametrize("raw,expected", [
    ("А123ВС777", "A123BC 777"),
    ("a 1 2 3 bc 7 7 7", "A123BC 777"),
    ("Н456ОК178", "H456OK 178"),
    ("Т777МР02", "T777MP 02"),
    ("АB1234", ""),
    ("123", ""),
    ("М005КХ77", "M005KX 77"),
])
def test_normalize_plate(raw, expected):
    assert normalize_plate(raw) == expected


def test_normalize_strips_garbage():
    assert normalize_plate("!@#$%") == ""
    assert normalize_plate("") == ""


@pytest.mark.parametrize("raw,expected", [
    # европейская OCR-модель путает O/0 и 1/T — правим по позициям формата РФ
    ("CO51K119", "C051KT 19"),
    ("C051KO199", "C051KO 199"),
    ("c 051 ko 199", "C051KO 199"),
])
def test_normalize_fixes_confusable_chars(raw, expected):
    assert normalize_plate(raw) == expected


def test_car_crop_clamps_and_upscales():
    from src.plate_recognizer import PlateRecognizer

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = PlateRecognizer.car_crop(frame, (-20, -10, 150, 90))
    assert crop is not None
    assert crop.shape[1] == 320, "мелкую область увеличиваем для детектора номеров"
    assert crop.shape[0] > 0
    assert PlateRecognizer.car_crop(frame, (300, 300, 400, 400)) is None
