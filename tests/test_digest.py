"""Тесты ежедневного дайджеста (сводки из БД)."""
import time

import pytest

from src.config import Config
from src.database import Database


@pytest.fixture
def db(tmp_path):
    cfg = Config({"database": {"path": str(tmp_path / "digest.db")}})
    d = Database(cfg)
    for name in ("Алексей", "Алексей", "Ольга"):
        d.save_face_event(name, 0.9, "cam", "")
    d.save_face_event("unknown", 0.5, "cam", "")
    d.save_car_event("cam", "A123BC 777", "Toyota", "white", "", "")
    d.save_car_event("cam", "A123BC 777", "Toyota", "white", "", "")
    return d


def test_digest_data(db):
    data = db.digest_data(hours=24)
    assert data["unknown"] == 1
    faces = {r["person"]: r["cnt"] for r in data["faces"]}
    assert faces.get("Алексей") == 2
    assert faces.get("Ольга") == 1
    assert data["cars"][0]["plate"] == "A123BC 777"
    assert data["cars"][0]["cnt"] == 2
    assert data["cars"][0]["color"] == "white"