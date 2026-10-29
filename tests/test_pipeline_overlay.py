"""Оверлей панели: рамки объектов и «чертёж лица» на кадре."""
import numpy as np
import pytest

from src.config import load_config
from src.pipeline import Pipeline


class FakeCamera:
    fps = 25.0
    is_file = True

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class FakeDB:
    def is_in_cooldown(self, *args, **kwargs):
        return False

    def is_object_in_cooldown(self, *args, **kwargs):
        return False


class FakeNotifier:
    def send_event(self, *args, **kwargs):
        return True


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr("src.pipeline.Camera", FakeCamera)
    cfg = load_config("config/config.yaml")
    cfg.web._data["overlay"] = True
    cfg.web._data["face_mesh"] = True
    return Pipeline(cfg, {"name": "cam", "source": "test.mp4"}, FakeDB(), FakeNotifier())


def test_face_mesh_is_drawn(pipe):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    points = np.array([[120, 80], [200, 80], [200, 160], [120, 160]], dtype=np.float32)
    kps = np.array([[130, 100], [190, 100], [160, 130], [140, 150], [180, 150]],
                   dtype=np.float32)
    pipe._draw_face_mesh(frame, {"landmarks": points, "kps": kps}, (0, 255, 0))
    assert np.count_nonzero(frame) > 0
    # контур лица (hull) проходит по краям прямоугольника
    assert frame[80, 160].any()
    assert frame[160, 160].any()


def test_overlay_draws_boxes_and_skips_without_landmarks(pipe):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pipe._set_overlays([
        {"bbox": (20, 20, 120, 120), "label": "car 0.90", "color": (0, 255, 0)},
        {"bbox": (150, 40, 250, 180), "label": "Аня 0.95", "color": (80, 255, 170),
         "landmarks": np.array([[160, 60], [240, 60], [240, 160], [160, 160]],
                               dtype=np.float32),
         "kps": None},
    ])
    pipe._draw_overlays(frame)
    assert frame[20, 20].any() or frame[21, 21].any()
    assert frame[60, 200].any()


def test_overlay_can_be_disabled(monkeypatch):
    monkeypatch.setattr("src.pipeline.Camera", FakeCamera)
    cfg = load_config("config/config.yaml")
    cfg.web._data["overlay"] = False
    p = Pipeline(cfg, {"name": "cam", "source": "test.mp4"}, FakeDB(), FakeNotifier())
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    p._set_overlays([{"bbox": (10, 10, 60, 60), "label": "dog", "color": (0, 255, 0)}])
    p._draw_overlays(frame)
    assert np.count_nonzero(frame) == 0
