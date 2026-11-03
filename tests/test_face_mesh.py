"""Скелет лица: связи 106 точек и отрисовка на кадре."""
import numpy as np

from src.face_mesh import (as_points, draw_face_box, draw_face_mesh,
                           mesh_edges)


def _face_points(count=106):
    """Правдоподобный набор точек: овал + внутренние точки."""
    points = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        radius_x = 55 + 8 * (i % 3)
        radius_y = 75 + 6 * (i % 5)
        points.append([120 + radius_x * np.cos(angle),
                       120 + radius_y * np.sin(angle)])
    return np.asarray(points, dtype=np.float32)


def test_as_points_accepts_flat_and_none():
    assert as_points(None) is None
    assert as_points([]) is None
    assert as_points([[1, 2], [3, 4]]) is None          # меньше 3 точек
    points = as_points([1, 2, 3, 4, 5, 6])
    assert points is not None and points.shape == (3, 2)


def test_mesh_edges_are_valid_and_not_too_long():
    points = _face_points()
    edges = mesh_edges(points)
    assert edges, "триангуляция должна дать хотя бы несколько рёбер"
    assert all(0 <= i < len(points) and 0 <= j < len(points)
               for i, j in edges)
    assert all(i != j for i, j in edges)
    # нет рёбер через всё лицо: длина ограничена размером лица
    max_len = max(np.hypot(*(points[i] - points[j])) for i, j in edges)
    assert max_len <= 160.0


def test_draw_face_mesh_marks_pixels():
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    points = _face_points()
    kps = points[[40, 45, 50, 55, 60]]
    drawn = draw_face_mesh(frame, {
        "bbox": (40, 40, 200, 220),
        "landmarks": points,
        "kps": kps,
        "color": (0, 255, 0),
    }, (0, 255, 0))
    assert drawn > 0
    assert np.count_nonzero(frame) > 100


def test_draw_face_box_with_label():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    draw_face_box(frame, {"bbox": (10, 20, 110, 110), "label": "Аня 0.95",
                          "color": (0, 255, 0)})
    assert frame[20, 10].any()
    assert np.count_nonzero(frame) > 0


def test_draw_face_mesh_without_landmarks_is_safe():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert draw_face_mesh(frame, {"bbox": (1, 1, 20, 20)}) == 0
    assert np.count_nonzero(frame) == 0
