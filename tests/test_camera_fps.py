"""FPS камеры: запрос к камере и ограничение частоты чтения."""
import numpy as np

from src.camera import Camera


class _FakeCap:
    def __init__(self):
        self.sets = []

    def read(self):
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def set(self, prop, value):
        self.sets.append((prop, value))


def _camera(target_fps, source=0):
    cam = Camera.__new__(Camera)
    cam.source = source
    cam.target_fps = target_fps
    cam._cap = _FakeCap()
    cam._next_at = 0.0
    cam._last_error_at = 0.0
    cam.reconnect_delay = 0.0
    return cam


def test_read_limits_fps(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.camera.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("src.camera.time.perf_counter", lambda: 100.0)

    cam = _camera(10)
    assert cam.read() is not None
    assert sleeps and abs(sleeps[0] - 0.1) < 1e-6


def test_read_without_limit_does_not_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.camera.time.sleep", lambda s: sleeps.append(s))
    cam = _camera(0)
    assert cam.read() is not None
    assert sleeps == []


def test_video_file_is_not_throttled(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.camera.time.sleep", lambda s: sleeps.append(s))
    cam = _camera(1, source="data/test_video.mp4")
    assert cam.read() is not None
    assert sleeps == []


def test_set_fps_updates_capture(monkeypatch):
    cam = _camera(0)
    cam.set_fps(24)
    assert cam.target_fps == 24
    assert cam._cap.sets and cam._cap.sets[-1][1] == 24.0
