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