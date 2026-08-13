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


def test_full_event_flow(db):
    """Лица + машины + номера с разных камер пишутся и достаются целиком."""
    db.save_face_event("Аня", 0.91, "cam1", "data/events/cam1/1.jpg")
    db.save_face_event("unknown", 0.3, "cam1", "")
    db.save_face_event("Аня", 0.89, "cam2", "")
    db.save_car_event("cam2", "Т777МР 777", "Kia", "red", "car.jpg", "plate.jpg")
    db.save_car_event("cam1", "Н456ОК 78", "Lada", "silver", "car2.jpg", "plate2.jpg")

    rows = db.recent_events(10)
    titles = {r["title"] for r in rows}
    assert "Аня" in titles, "лицо человека должно быть в событиях"
    assert "Т777МР 777" in titles, "номер машины должен быть в событиях"
    assert "Н456ОК 78" in titles, "второй номер тоже"
    cams = {r["camera"] for r in rows}
    assert {"cam1", "cam2"} <= cams, "должны быть события с обеих камер"

    cars = [r for r in rows if "Т777МР 777" == r["title"]]
    assert cars and cars[0]["extra"] == "red Kia", f"не заполнены цвет/марка: {cars}"

    dig = db.digest_data(hours=24)
    faces = {r["person"]: r["cnt"] for r in dig["faces"]}
    assert faces.get("Аня") == 2, f"в дайджесте не 2 визита Ани: {faces}"
    assert dig["unknown"] == 1, f"незнакомец не учтён: {dig['unknown']}"
    plates = {r["plate"] for r in dig["cars"]}
    assert {"Т777МР 777", "Н456ОК 78"} == plates, f"номера не собраны в дайджесте: {plates}"