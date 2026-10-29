"""События животных/посылок и выборки для веб-панели."""
import time

import pytest

from src.config import Config
from src.database import Database


@pytest.fixture
def db(tmp_path):
    cfg = Config({"database": {"path": str(tmp_path / "test.db")}})
    database = Database(cfg)
    yield database
    database.close()


def test_save_object_event_and_cooldown(db):
    event_id = db.save_object_event("cam1", "animal", "dog", "Собака",
                                    confidence=0.88, image_path="dog.jpg")
    assert event_id > 0
    assert db.is_object_in_cooldown("animal", "dog", 30) is True
    assert db.is_object_in_cooldown("animal", "cat", 30) is False
    assert db.is_object_in_cooldown("package", "dog", 30) is False

    rows = db.recent_events()
    assert rows and rows[0]["title"] == "Собака"


def test_save_object_event_rejects_unknown_kind(db):
    with pytest.raises(ValueError):
        db.save_object_event("cam", "person", "x")


def test_query_events_filters_and_pagination(db):
    for i in range(5):
        db.save_face_event("Аня" if i % 2 else "unknown", 0.9, "cam1", f"f{i}.jpg")
    db.save_car_event("cam2", "A123BC 777", "Toyota", "white", "car.jpg", "plate.jpg")
    db.save_object_event("cam2", "animal", "dog", "Собака", 0.9, 0, "dog.jpg")
    db.save_object_event("cam2", "package", "backpack", "Рюкзак", 0.9, 65, "bag.jpg")
    time.sleep(0.01)

    animals, more = db.query_events(kind="animal", limit=10)
    assert [e["type"] for e in animals] == ["animal"]
    assert animals[0]["label"] == "dog"
    assert more is False

    packages, _ = db.query_events(kind="package")
    assert packages[0]["title"] == "Рюкзак"
    assert "без движения 65" in packages[0]["subtitle"]

    objects, _ = db.query_events(kind="object")
    assert {e["type"] for e in objects} == {"animal", "package"}

    cams, _ = db.query_events(camera="cam2", limit=50)
    assert {e["camera"] for e in cams} == {"cam2"}
    assert len(cams) == 3

    found, _ = db.query_events(q="A123")
    assert found and found[0]["type"] == "car"

    page1, more1 = db.query_events(limit=2, offset=0)
    page2, more2 = db.query_events(limit=2, offset=2)
    assert len(page1) == 2 and more1 is True
    assert len(page2) == 2 and more2 is True
    assert {(e["type"], e["id"]) for e in page1} != {(e["type"], e["id"]) for e in page2}


def test_stats_counts(db):
    db.save_face_event("Аня", 0.9, "cam1", "")
    db.save_face_event("unknown", 0.4, "cam1", "")
    db.save_car_event("cam1", "A123BC 777", "Toyota", "white")
    db.save_object_event("cam2", "animal", "cat", "Кошка")
    db.save_object_event("cam2", "package", "suitcase", "Чемодан", stationary_seconds=50)

    stats = db.stats(hours=24)
    assert stats["total"] == 5
    assert stats["faces"] == 2 and stats["unknown"] == 1
    assert stats["cars"] == 1
    assert stats["animals"] == 1 and stats["packages"] == 1
    assert {c["camera"] for c in stats["by_camera"]} == {"cam1", "cam2"}
    assert stats["last_ts"] is not None


def test_media_path_by_event_id(db, tmp_path):
    img = tmp_path / "dog.jpg"
    img.write_bytes(b"jpeg")
    crop = tmp_path / "plate.jpg"
    crop.write_bytes(b"crop")
    event_id = db.save_object_event("cam", "animal", "dog", "Собака", image_path=str(img))
    car_id = db.save_car_event("cam", "A123BC 777", "Toyota", "white", str(img), str(crop))

    assert db.get_media_path("animal", event_id) == str(img)
    assert db.get_media_path("car", car_id) == str(img)
    assert db.get_media_path("car", car_id, crop=True) == str(crop)
    assert db.get_media_path("animal", 99999) is None
    assert db.get_media_path("nope", event_id) is None
