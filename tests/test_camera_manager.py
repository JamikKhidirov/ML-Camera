"""CameraManager: статусы, старт/стоп, снимок и файл статусов."""
from __future__ import annotations

import threading
import time

import pytest

from src.camera_manager import CameraManager
from src.camera_status import read_status, status_path
from src.config import Config


class FakePipeline:
    """Минимальная замена Pipeline для проверки менеджера без камеры."""

    instances: list["FakePipeline"] = []

    def __init__(self, cfg, camera_cfg, db, notifier):
        self.cfg = cfg
        self.camera_cfg = camera_cfg
        self.db = db
        self.notifier = notifier
        self.frames = 0
        self.prepared = 0
        self._stop = threading.Event()
        self.last_frame_ts = 0.0
        self._frame_size = (0, 0)
        FakePipeline.instances.append(self)

    @property
    def frames_processed(self) -> int:
        return self.frames

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def stop(self) -> None:
        self._stop.set()

    def prepare_start(self) -> None:
        self.prepared += 1
        self._stop.clear()

    def run(self, max_frames=None) -> None:
        while not self._stop.is_set():
            self.frames += 1
            self.last_frame_ts = time.time()
            self._frame_size = (640, 360)
            if max_frames is not None and self.frames >= max_frames:
                return
            time.sleep(0.01)

    def snapshot_jpeg(self, quality=85):
        return b"\xff\xd8\xfffake" if self.frames else None


class BoomPipeline(FakePipeline):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("камера недоступна")


class FakeDB:
    def close(self):
        pass


class FakeNotifier:
    pass


@pytest.fixture(autouse=True)
def _clean_instances():
    FakePipeline.instances.clear()
    yield
    FakePipeline.instances.clear()


def _cfg(tmp_path):
    return Config({
        "database": {"path": str(tmp_path / "cam.db")},
        "storage": {"events_dir": str(tmp_path / "events")},
        "web": {"host": "127.0.0.1", "port": 0},
        "cameras": [
            {"name": "Двор", "source": "rtsp://x/1", "enabled": True},
            {"name": "Ворота", "source": "rtsp://x/2", "enabled": False},
        ],
    })


def _manager(cfg, factory=FakePipeline):
    return CameraManager(cfg, FakeDB(), FakeNotifier(),
                         cameras=cfg.get("cameras"), pipeline_factory=factory)


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_list_reports_config_cameras(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    info = manager.list()
    assert [c["name"] for c in info] == ["Двор", "Ворота"]
    assert info[0]["enabled"] is True and info[1]["enabled"] is False
    assert all(c["status"] == "stopped" for c in info)


def test_start_stop_and_restart(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)

    started = manager.start_enabled(max_frames=None)
    assert started == 1, "выключенная камера не должна стартовать"
    assert _wait_for(lambda: manager.get(0).info()["frames"] > 0)

    info = manager.get(0).info()
    assert info["status"] == "running"
    assert info["alive"] is True
    assert manager.get(1).info()["status"] == "stopped"

    ok, message = manager.stop(0)
    assert ok and message == ""
    assert manager.get(0).info()["status"] == "stopped"
    assert manager.get(0).info()["alive"] is False

    # повторный запуск использует тот же Pipeline и переоткрывает источник
    ok, _ = manager.start(0)
    assert ok
    assert _wait_for(lambda: manager.get(0).pipe.prepared == 1)
    assert manager.get(0).pipe is FakePipeline.instances[0]
    manager.stop_all()


def test_start_twice_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    assert manager.start(0)[0] is True
    assert _wait_for(lambda: manager.get(0).info()["frames"] > 0)
    ok, message = manager.start(0)
    assert ok is False and "уже запущена" in message
    manager.stop_all()


def test_error_status_when_camera_unavailable(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg, factory=BoomPipeline)
    ok, _ = manager.start(0)
    assert ok is True, "запуск асинхронный — ошибка появляется в статусе"
    assert _wait_for(lambda: manager.get(0).info()["status"] == "error"
                     and not manager.get(0).info()["alive"])
    info = manager.get(0).info()
    assert "камера недоступна" in info["error"]
    assert info["alive"] is False


def test_snapshot_and_status_file(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    assert manager.snapshot(0) is None

    manager.start(0)
    assert _wait_for(lambda: manager.snapshot(0) is not None)
    assert manager.snapshot(0).startswith(b"\xff\xd8\xff")
    manager.stop_all()

    manager.write_status()
    data = read_status(cfg)
    assert status_path(cfg).is_file()
    assert [c["name"] for c in data["cameras"]] == ["Двор", "Ворота"]
    assert data["updated_at"] > 0


def test_wait_returns_after_stop(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    manager.start(0)
    assert _wait_for(lambda: manager.any_alive)
    manager.stop(0)
    assert manager.wait(timeout=2.0) is True


def test_info_has_ui_fields(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    info = manager.get(0).info()
    for key in ("proc_fps", "width", "height", "source_kind", "recording",
                "recording_mode", "started_at", "uptime"):
        assert key in info, key
    assert info["source_kind"] == "rtsp"
    assert info["proc_fps"] == 0.0
    assert info["width"] == 0 and info["height"] == 0
    assert info["uptime"] == 0.0

    manager.start(0)
    assert _wait_for(lambda: manager.get(0).info()["width"] == 640)
    assert _wait_for(lambda: manager.get(0).info()["uptime"] > 0)
    info = manager.get(0).info()
    assert info["width"] == 640 and info["height"] == 360
    assert info["uptime"] > 0
    assert info["proc_fps"] > 0
    manager.stop_all()
    assert manager.get(0).info()["proc_fps"] == 0.0


def test_add_camera_on_the_fly(tmp_path):
    cfg = _cfg(tmp_path)
    manager = _manager(cfg)
    assert len(manager) == 2

    webcam = manager.add_camera("Ноутбук (веб-камера)", 0)
    assert len(manager) == 3
    info = webcam.info()
    assert info["index"] == 2
    assert info["name"] == "Ноутбук (веб-камера)"
    assert info["source_kind"] == "webcam"
    assert info["enabled"] is True

    assert webcam.start()[0] is True
    assert _wait_for(lambda: webcam.info()["frames"] > 0)
    assert webcam.info()["status"] == "running"
    manager.stop_all()
