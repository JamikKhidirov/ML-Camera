"""Животные и оставленные вещи в Pipeline (без загрузки YOLO)."""
import numpy as np
import pytest

from src.config import load_config
from src.detector import Detection
from src.pipeline import Pipeline


class FakeCamera:
    fps = 25.0
    is_file = True

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.objects = []
        self.cars = []
        self.faces = []

    def is_in_cooldown(self, *args, **kwargs):
        return False

    def is_object_in_cooldown(self, *args, **kwargs):
        return False

    def save_object_event(self, *args):
        self.objects.append(args)

    def save_car_event(self, *args):
        self.cars.append(args)

    def save_face_event(self, *args):
        self.faces.append(args)


class FakeNotifier:
    def __init__(self):
        self.events = []

    def send_event(self, title, message, image_path=None, extra_images=()):
        self.events.append((title, message, image_path))
        return True


class FakeAnalyzer:
    def analyze(self, crop):
        from src.car_analyzer import CarInfo
        return CarInfo(color="white", brand="unknown")


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr("src.pipeline.Camera", FakeCamera)
    cfg = load_config("config/config.yaml")
    cfg.car._data["plate_enabled"] = False
    cfg.car._data["plate_vote_frames"] = 99
    cfg.objects._data["packages"]["stationary_seconds"] = 2
    p = Pipeline(cfg, {"name": "cam", "source": "test.mp4"}, FakeDB(), FakeNotifier())
    p.car_analyzer = FakeAnalyzer()
    monkeypatch.setattr(p, "_save_image_bytes", lambda *a, **k: "")
    monkeypatch.setattr(p, "_save_image", lambda *a, **k: "")
    return p


def _frame():
    return np.zeros((160, 240, 3), dtype=np.uint8)


def test_dog_notified_once_per_track(pipe):
    frame = _frame()
    dog = Detection("dog", 0.9, 20, 20, 90, 110)

    for _ in range(3):                      # min_frames=2 -> на третьем кадре
        pipe._process_objects(frame, frame, 1.0, [dog])

    assert len(pipe.db.objects) == 1, "собака должна дать одно событие"
    kind, label, title = pipe.db.objects[0][1], pipe.db.objects[0][2], pipe.db.objects[0][3]
    assert (kind, label, title) == ("animal", "dog", "Собака")
    assert len(pipe.notifier.events) == 1
    assert "Собака" in pipe.notifier.events[0][0]

    # трек уже уведомлён — повторов нет
    for _ in range(5):
        pipe._process_objects(frame, frame, 1.0, [dog])
    assert len(pipe.db.objects) == 1
    assert len(pipe.notifier.events) == 1


def test_package_must_stand_still(pipe, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("src.pipeline.time.time", lambda: now[0])
    frame = _frame()
    # вещь «едет» по кадру — это не оставленная посылка
    for i in range(6):
        now[0] += 1.0
        bag = Detection("backpack", 0.9, 10 + i * 30, 40, 60 + i * 30, 100)
        pipe._process_objects(frame, frame, 1.0, [bag])
    assert pipe.db.objects == [], "движущийся рюкзак не должен считаться оставленным"

    # остановилась — 3 секунды без движения при пороге 2 с -> событие
    still = Detection("backpack", 0.9, 190, 40, 235, 100)
    for _ in range(4):
        now[0] += 1.0
        pipe._process_objects(frame, frame, 1.0, [still])
    assert len(pipe.db.objects) == 1
    kind, label, title = pipe.db.objects[0][1], pipe.db.objects[0][2], pipe.db.objects[0][3]
    assert (kind, label, title) == ("package", "backpack", "Рюкзак")
    assert "Оставлен рюкзак" in pipe.notifier.events[0][0]


def test_objects_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr("src.pipeline.Camera", FakeCamera)
    cfg = load_config("config/config.yaml")
    cfg.objects._data["enabled"] = False
    p = Pipeline(cfg, {"name": "cam", "source": "test.mp4"}, FakeDB(), FakeNotifier())
    frame = _frame()
    assert p._process_objects(frame, frame, 1.0, [Detection("dog", 0.9, 1, 1, 20, 20)]) == []
    assert p.db.objects == []
