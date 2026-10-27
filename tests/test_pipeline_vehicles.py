"""Проверка финализации автомобилей в Pipeline без загрузки моделей."""
import numpy as np
import pytest

from src.car_analyzer import CarInfo
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
        self.cars = []
        self.faces = []

    def is_in_cooldown(self, *args, **kwargs):
        return False

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
        return CarInfo(color="white", brand="unknown")


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr("src.pipeline.Camera", FakeCamera)
    cfg = load_config("config/config.local.yaml")
    # номера в этом тесте не нужны, а «финализация по vote_frames» не должна
    # срабатывать раньше, чем машина исчезнет
    cfg.car._data["plate_enabled"] = False
    cfg.car._data["plate_vote_frames"] = 99
    p = Pipeline(cfg, {"name": "cam", "source": "test.mp4"}, FakeDB(), FakeNotifier())
    p.car_analyzer = FakeAnalyzer()
    monkeypatch.setattr(p, "_save_image_bytes", lambda *a, **k: "")
    return p


def test_car_finalized_when_it_disappears(pipe):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    det = Detection("car", 0.9, 20, 30, 80, 90)

    pipe._process_vehicles(frame, [det], frame, 1.0)
    assert pipe.db.cars == []          # один кадр — ещё не событие

    max_dis = pipe.car_tracker.max_disappeared
    for _ in range(max_dis + 2):       # машина уехала
        pipe._process_vehicles(frame, [], frame, 1.0)

    assert len(pipe.db.cars) == 1, "машина должна попасть в БД после исчезновения"
    assert len(pipe.notifier.events) == 1, "и должно уйти одно уведомление"


def test_empty_detections_without_tracks_are_safe(pipe):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pipe._process_vehicles(frame, [], frame, 1.0)
    assert pipe.db.cars == []


def test_best_crop_keeps_largest_and_clearer(pipe):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    small = np.full((20, 20, 3), 50, dtype=np.uint8)
    big = np.full((100, 100, 3), 50, dtype=np.uint8)
    data: dict = {}

    assert pipe._remember_best_crop(data, small, frame, 0.9)
    assert pipe._remember_best_crop(data, big, frame, 0.5)
    assert data["best_crop"].shape[:2] == (100, 100)
    # меньшая вырезка уже не должна перебивать лучшую
    assert not pipe._remember_best_crop(data, small, frame, 1.0)
    assert data["best_crop"].shape[:2] == (100, 100)


def test_brand_analyzed_once_per_track(pipe):
    calls = []

    class CountingAnalyzer:
        def analyze(self, crop):
            calls.append(crop.shape)
            return CarInfo(color="white", brand="Toyota", brand_confidence=0.9)

    pipe.car_analyzer = CountingAnalyzer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    det = Detection("car", 0.9, 20, 30, 100, 110)

    for _ in range(4):                       # несколько кадров одной машины
        pipe._process_vehicles(frame, [det], frame, 1.0)
    assert calls == [], "марка не должна считаться на каждом кадре"

    for _ in range(pipe.car_tracker.max_disappeared + 2):
        pipe._process_vehicles(frame, [], frame, 1.0)

    assert len(calls) == 1, "марка считается один раз — при финализации"
    assert pipe.db.cars and pipe.db.cars[0][2] == "Toyota"
