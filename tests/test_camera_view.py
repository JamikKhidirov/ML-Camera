"""Превью для Tkinter: захват, анализ лиц, скелет, зеркало и снимок."""
import time

import numpy as np

from src.camera_view import CameraPreview, FaceOverlay
from src.config import Config
from src.face_recognizer import FaceMatch


class FakeCamera:
    """Отдаёт готовый кадр без железа."""

    fps = 30.0
    is_file = False

    def __init__(self, *args, **kwargs):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # яркая точка, чтобы проверить, что кадр реально доходит до окна
        self.frame[100:120, 100:120] = (0, 0, 255)

    def read(self):
        time.sleep(0.002)
        return self.frame.copy()

    def open(self):
        pass

    def close(self):
        pass


class FakeRecognizer:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def recognize(self, frame):
        landmarks = np.zeros((106, 2), dtype=np.float32)
        for i in range(106):
            landmarks[i] = (80 + i % 10 * 8, 80 + i // 10 * 6)
        kps = landmarks[[10, 20, 30, 40, 50]]
        match = FaceMatch(name="Аня", similarity=0.93,
                          bbox=(60, 60, 180, 200), landmarks=landmarks,
                          kps=kps)
        return [match], []


def _cfg(tmp_path):
    return Config({
        "camera": {"width": 320, "height": 240, "fps": 30},
        "preview": {"fps": 30, "width": 320, "analysis_fps": 20,
                    "analysis_width": 320, "face_mesh": True,
                    "boxes": True, "mirror": True},
        "people": {"unknown_label": "Незнакомец"},
        "database": {"path": str(tmp_path / "db.sqlite")},
        "storage": {"events_dir": str(tmp_path / "events")},
    })


def test_preview_receives_frames_and_faces(tmp_path, monkeypatch):
    monkeypatch.setattr("src.camera_view.Camera", FakeCamera)
    preview = CameraPreview(_cfg(tmp_path), {"name": "Тест", "source": 0},
                            recognizer_factory=FakeRecognizer)
    preview.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and not preview.latest_faces():
            time.sleep(0.05)
        stats = preview.stats()
        assert stats["running"]
        assert stats["frames"] > 0
        assert stats["faces"] == 1
        assert preview.latest_faces()[0].label.startswith("Аня")

        frame = preview.frame_for_display()
        assert frame is not None
        assert frame.shape[2] == 3
        assert frame.any(), "скелет и рамка должны оставить пиксели"
    finally:
        preview.stop()
    assert not preview.running


def test_snapshot_saves_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("src.camera_view.Camera", FakeCamera)
    preview = CameraPreview(_cfg(tmp_path), {"name": "Двор", "source": 0},
                            recognizer_factory=FakeRecognizer)
    preview.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and preview.latest_frame() is None:
            time.sleep(0.02)
        out = tmp_path / "shot.jpg"
        assert preview.snapshot(out) == str(out)
        assert out.read_bytes().startswith(b"\xff\xd8")
    finally:
        preview.stop()


def test_mirror_moves_landmarks_and_bbox():
    face = FaceOverlay(
        bbox=(10, 20, 30, 40), label="Аня", color=(0, 255, 0),
        landmarks=np.array([[10.0, 5.0], [30.0, 5.0], [20.0, 30.0]],
                           dtype=np.float32),
    )
    mirrored = CameraPreview._mirror_overlay(face, width=100)
    assert mirrored.bbox == (69, 20, 89, 40)
    assert mirrored.landmarks[0, 0] == 89.0
    assert mirrored.landmarks[1, 0] == 69.0


def test_switches_change_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("src.camera_view.Camera", FakeCamera)
    preview = CameraPreview(_cfg(tmp_path), {"name": "Тест", "source": 0},
                            recognizer_factory=FakeRecognizer)
    preview.set_mesh_enabled(False)
    preview.set_boxes_enabled(False)
    preview.set_mirror(False)
    assert preview.face_mesh_enabled is False
    assert preview.boxes_enabled is False
    assert preview.mirror is False
