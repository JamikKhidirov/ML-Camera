"""Скелет тела: связи COCO-17, отрисовка и разбор ответа YOLO pose."""
import numpy as np

from src.config import Config
from src.pose import (POSE_EDGES, PoseEstimator, PoseOverlay, draw_pose)


def _keypoints() -> np.ndarray:
    """Похожий на человека набор 17 точек."""
    points = np.array([
        [50, 10],    # 0 нос
        [46, 8], [54, 8],       # глаза
        [42, 10], [58, 10],     # уши
        [35, 30], [65, 30],     # плечи
        [30, 50], [70, 50],     # локти
        [28, 70], [72, 70],     # запястья
        [42, 70], [58, 70],     # таз
        [40, 100], [60, 100],   # колени
        [38, 130], [62, 130],   # стопы
    ], dtype=np.float32)
    return points


def test_pose_edges_use_existing_points():
    used = {index for edge in POSE_EDGES for index in edge}
    assert len(POSE_EDGES) >= 16
    assert used <= set(range(17))
    assert {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} <= used


def test_draw_pose_marks_pixels():
    frame = np.zeros((160, 100, 3), dtype=np.uint8)
    item = {
        "bbox": (20, 0, 80, 150),
        "keypoints": _keypoints(),
        "confidence": np.ones(17, dtype=np.float32),
    }
    drawn = draw_pose(frame, item, thickness=1)
    assert drawn >= 16
    assert frame.any(), "скелет тела должен оставить след на кадре"


def test_draw_pose_respects_confidence():
    frame = np.zeros((160, 100, 3), dtype=np.uint8)
    item = {
        "bbox": (20, 0, 80, 150),
        "keypoints": _keypoints(),
        "confidence": np.zeros(17, dtype=np.float32),
    }
    assert draw_pose(frame, item, min_conf=0.35) == 0
    assert not frame.any()


def test_draw_pose_without_points_is_safe():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert draw_pose(frame, {"bbox": (0, 0, 10, 10)}) == 0
    assert draw_pose(frame, {}) == 0


class _Boxes:
    xyxy = np.array([[10, 5, 90, 150]], dtype=np.float32)
    conf = np.array([0.91], dtype=np.float32)


class _Keypoints:
    xy = _keypoints()[None, ...]
    conf = np.ones((1, 17), dtype=np.float32)


class _Result:
    boxes = _Boxes()
    keypoints = _Keypoints()


class _FakeYolo:
    def __call__(self, frame, **kwargs):
        return [_Result()]


def _cfg() -> Config:
    return Config({
        "models": {"pose": "data/models/yolov8n-pose.pt"},
        "preview": {"pose_conf": 0.35},
    })


def test_estimator_maps_yolo_result_to_overlay():
    estimator = PoseEstimator(_cfg())
    estimator._model = _FakeYolo()          # без загрузки весов
    poses = estimator.detect(np.zeros((200, 120, 3), dtype=np.uint8))
    assert len(poses) == 1
    pose = poses[0]
    assert isinstance(pose, PoseOverlay)
    assert pose.bbox == (10, 5, 90, 150)
    assert pose.keypoints.shape == (17, 2)
    assert pose.confidence.shape == (17,)
    assert abs(pose.score - 0.91) < 1e-6


def test_estimator_handles_broken_model():
    class _Broken:
        def __call__(self, frame, **kwargs):
            raise RuntimeError("нет весов")

    estimator = PoseEstimator(_cfg())
    estimator._model = _Broken()
    assert estimator.detect(np.zeros((40, 40, 3), dtype=np.uint8)) == []
    assert "нет весов" in estimator.error


def test_estimator_handles_empty_frame():
    estimator = PoseEstimator(_cfg())
    assert estimator.detect(None) == []
    assert estimator.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
