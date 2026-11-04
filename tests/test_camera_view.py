"""Превью для Tkinter: захват, анализ лиц, скелет, зеркало и снимок."""
import time

import numpy as np

from src.camera_view import CameraPreview, FaceOverlay
from src.config import Config
from src.face_recognizer import FaceMatch
from src.pose import PoseOverlay


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


class FakeFileCamera(FakeCamera):
    """Видеофайл: читается быстро, но окно должно играть в своём темпе."""

    is_file = True
    fps = 40.0


class FakePose:
    """17 точек без загрузки YOLO — тесты не тянут веса."""

    error = ""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def detect(self, frame):
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[:, 0] = 30
        keypoints[:, 1] = np.linspace(40, 220, 17)
        return [PoseOverlay(bbox=(20, 40, 60, 220),
                            keypoints=keypoints,
                            confidence=np.ones(17, dtype=np.float32),
                            score=0.9)]


def _cfg_pose(tmp_path):
    cfg = _cfg(tmp_path)
    cfg._data["preview"].update({"pose": True, "pose_fps": 20,
                                 "pose_conf": 0.35})
    return cfg


def _cfg(tmp_path):
    return Config({
        "camera": {"width": 320, "height": 240, "fps": 30},
        "preview": {"fps": 30, "width": 320, "analysis_fps": 20,
                    "analysis_width": 320, "face_mesh": True,
                    "boxes": True, "mirror": True, "pose": False},
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
    preview.set_pose_enabled(False)
    assert preview.face_mesh_enabled is False
    assert preview.boxes_enabled is False
    assert preview.mirror is False
    assert preview.pose_enabled is False


def test_mirror_pose_moves_keypoints_and_bbox():
    pose = PoseOverlay(
        bbox=(10, 20, 30, 40),
        keypoints=np.array([[10.0, 5.0], [30.0, 5.0]], dtype=np.float32),
        confidence=np.ones(2, dtype=np.float32),
    )
    mirrored = CameraPreview._mirror_pose(pose, width=100)
    assert mirrored.bbox == (69, 20, 89, 40)
    assert mirrored.keypoints[0, 0] == 89.0
    assert mirrored.keypoints[1, 0] == 69.0


def test_preview_detects_body(tmp_path, monkeypatch):
    monkeypatch.setattr("src.camera_view.Camera", FakeCamera)
    preview = CameraPreview(_cfg_pose(tmp_path), {"name": "Тело", "source": 0},
                            recognizer_factory=FakeRecognizer,
                            pose_factory=FakePose)
    preview.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and not preview.latest_poses():
            time.sleep(0.05)
        stats = preview.stats()
        assert stats["pose_enabled"] is True
        assert stats["poses"] == 1
        poses = preview.latest_poses()
        assert poses[0].keypoints.shape == (17, 2)
        frame = preview.frame_for_display()
        assert frame is not None and frame.any()
    finally:
        preview.stop()


def test_video_file_plays_in_real_time(tmp_path, monkeypatch):
    monkeypatch.setattr("src.camera_view.Camera", FakeFileCamera)
    preview = CameraPreview(_cfg(tmp_path), {"name": "Видео", "source": "x.mp4"},
                            recognizer_factory=FakeRecognizer)
    preview.start()
    try:
        time.sleep(0.5)
        frames = preview.stats()["frames"]
    finally:
        preview.stop()
    # 40 FPS за 0,5 с — около 20 кадров; без ограничения было бы тысячи.
    assert 1 <= frames <= 100
