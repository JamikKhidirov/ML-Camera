"""Тесты базы данных (используют временный SQLite)."""
import time

import pytest

from src.config import Config
from src.database import Database


@pytest.fixture
def db(tmp_path):
    cfg = Config({"database": {"path": str(tmp_path / "test.db")}})
    return Database(cfg)


def test_save_face_event(db):
    _id = db.save_face_event("мама", 0.87, "cam1", "img.jpg")
    assert _id > 0
    rows = db.recent_events()
    assert rows and rows[0]["title"] == "мама"


def test_cooldown(db):
    db.save_face_event("папа", 0.9, "cam1", "")
    assert db.is_in_cooldown("face", "папа", cooldown_min=60) is True
    assert db.is_in_cooldown("face", "мама", cooldown_min=60) is False


def test_save_car_event(db):
    db.save_car_event("cam1", "A123BC 777", "Toyota", "white", "img.jpg", "crop.jpg")
    rows = db.recent_events()
    assert rows and rows[0]["title"] == "A123BC 777"
    assert db.is_in_cooldown("car", "A123BC 777", cooldown_min=60) is True