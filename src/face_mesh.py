"""Скелет лица по 106 точкам InsightFace.

Одна и та же отрисовка используется:
  - веб-панелью и окном OpenCV (src/pipeline.py);
  - Tkinter-окном камеры (src/camera_window.py).

Почему не «сетка Делоне»: триангуляция всех 106 точек даёт сотни длинных
линий через всё лицо — скелет превращается в кашу. Здесь связи заданы по
смыслу: овал лица, брови, глаза, нос и губы. Так лицо читается, а не
мерцает паутиной.

Индексы 106-точечной модели InsightFace:
  0–32    овал лица (порядок в модели «не по кругу», см. CONTOUR_ORDER)
  33–42   левый глаз (на изображении)
  43–51   левая бровь
  52–71   губы
  72–86   нос
  87–96   правый глаз
  97–105  правая бровь
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

PointLike = Sequence[Sequence[float]]

# Первые 33 точки — овал лица. В самой модели они лежат не по порядку,
# поэтому обход задан вручную: от левого виска вниз, через подбородок,
# вверх по правой стороне. Верх лба намеренно не замыкаем — иначе
# получается «обруч» через лоб.
CONTOUR_COUNT = 33
CONTOUR_ORDER = [
    1, 9, 10, 11, 12, 13, 14, 15, 16,
    2, 3, 4, 5, 6, 7, 8, 0,
    24, 23, 22, 21, 20, 19, 18,
    32, 31, 30, 29, 28, 27, 26, 25, 17,
]

EYE_LEFT = list(range(33, 43))       # замкнутый контур глаза
EYEBROW_LEFT = list(range(43, 52))   # линия брови
MOUTH_OUTER = list(range(52, 72))    # внешний контур губ
NOSE_BRIDGE = [72, 73, 74]           # переносица
NOSE_LEFT = [75, 76, 77, 78, 79, 80]
NOSE_RIGHT = [81, 82, 83, 84, 85, 86]
EYE_RIGHT = list(range(87, 97))
EYEBROW_RIGHT = list(range(97, 106))
# Внутренняя линия губ (между верхней и нижней губой).
MOUTH_INNER = [54, 66, 62, 60, 71, 70, 69]

DEFAULT_COLOR = (80, 255, 170)


def as_points(value) -> Optional[np.ndarray]:
    """Привести landmarks к массиву (N, 2) float32 или вернуть None."""
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.size == 0:
        return None
    try:
        points = points.reshape(-1, 2)
    except ValueError:
        return None
    if len(points) < 3:
        return None
    return points


def _chain(indices: Sequence[int], count: int, closed: bool = False):
    """Пары соседних индексов, которые точно существуют в массиве точек."""
    valid = [int(i) for i in indices if 0 <= int(i) < count]
    edges = [(valid[i], valid[i + 1]) for i in range(len(valid) - 1)]
    if closed and len(valid) > 2:
        edges.append((valid[-1], valid[0]))
    return edges


def _hull_chain(indices: Sequence[int], points: np.ndarray):
    """Замкнутый контур по выпуклой оболочке группы точек.

    Глаза, брови и губы — почти выпуклые фигуры, поэтому оболочка даёт
    аккуратный контур без ручного перечисления порядка 10–20 точек.
    """
    valid = [int(i) for i in indices if 0 <= int(i) < len(points)]
    if len(valid) < 3:
        return []
    group = points[valid]
    hull = cv2.convexHull(group.astype(np.int32), returnPoints=False)
    if hull is None or len(hull) < 3:
        return []
    ordered = [valid[int(i)] for i in hull.reshape(-1)]
    return _chain(ordered, len(points), closed=True)


def mesh_edges(points: PointLike) -> list[tuple[int, int]]:
    """Пары индексов точек, которые нужно соединить линиями.

    Результат кэшируется в оверлее на кадр анализа, поэтому во время
    отрисовки видео ничего не пересчитывается.
    """
    pts = as_points(points)
    if pts is None:
        return []
    count = len(pts)
    edges: list[tuple[int, int]] = []
    edges += _chain(CONTOUR_ORDER, count, closed=False)
    edges += _hull_chain(EYE_LEFT, pts)
    edges += _hull_chain(EYE_RIGHT, pts)
    edges += _hull_chain(EYEBROW_LEFT, pts)
    edges += _hull_chain(EYEBROW_RIGHT, pts)
    edges += _chain(NOSE_BRIDGE, count, closed=False)
    edges += _chain(NOSE_LEFT, count, closed=False)
    edges += _chain(NOSE_RIGHT, count, closed=False)
    edges += _hull_chain(MOUTH_OUTER, pts)
    edges += _hull_chain(MOUTH_INNER, pts)
    # без дублей, порядок стабилен для тестов и кэша
    return sorted({(min(i, j), max(i, j)) for i, j in edges if i != j})


def _dim(color: tuple[int, int, int], factor: float = 0.6) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, channel * factor))) for channel in color)  # type: ignore[return-value]


def draw_face_mesh(frame: np.ndarray, item: dict,
                   color: tuple[int, int, int] = DEFAULT_COLOR,
                   thickness: int = 1,
                   draw_edges: bool = True) -> int:
    """Нарисовать скелет лица из элемента оверлея.

    ``item`` — словарь с ключами ``landmarks`` (106×2), ``kps`` (5×2),
    ``bbox`` и, необязательно, заранее посчитанными ``edges``.
    Возвращает число нарисованных связей (для тестов/статистики).
    """
    points = as_points(item.get("landmarks"))
    if points is None:
        return 0

    bbox = item.get("bbox")
    small_face = False
    if bbox and len(bbox) == 4:
        bw = max(1, int(bbox[2]) - int(bbox[0]))
        bh = max(1, int(bbox[3]) - int(bbox[1]))
        radius = max(1, min(3, int(min(bw, bh) * 0.008)))
        # На маленьком лице 106 точек сливаются в «кашу». Оставляем линии,
        # а точки рисуем только когда лицо достаточно крупное.
        small_face = min(bw, bh) < 90
    else:
        radius = 1

    edges = item.get("edges")
    if edges is None:
        edges = mesh_edges(points) if draw_edges else []
    edge_count = 0
    if draw_edges and edges:
        line_color = _dim(color, 0.8)
        for i, j in edges:
            if 0 <= int(i) < len(points) and 0 <= int(j) < len(points):
                p1 = (int(points[i][0]), int(points[i][1]))
                p2 = (int(points[j][0]), int(points[j][1]))
                cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)
                edge_count += 1

    # Овал лица — рисуем ярче и чуть толще остальных линий.
    contour = [int(i) for i in CONTOUR_ORDER if 0 <= int(i) < len(points)]
    if len(contour) >= 3:
        polyline = np.array([[int(points[i][0]), int(points[i][1])]
                             for i in contour], dtype=np.int32)
        cv2.polylines(frame, [polyline], False, color,
                      max(1, thickness + 1), cv2.LINE_AA)

    # Сами точки: маленькие — landmarks, крупные — 5 ключевых точек (kps).
    if not small_face:
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    kps = as_points(item.get("kps"))
    if kps is not None:
        big = max(2, radius + 2)
        for x, y in kps:
            cv2.circle(frame, (int(x), int(y)), big, (255, 255, 255), -1,
                       cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), big + 1, color, 1, cv2.LINE_AA)
    return edge_count


def draw_face_box(frame: np.ndarray, item: dict,
                  thickness: int = 2) -> None:
    """Рамка + подпись имени/сходства над лицом."""
    bbox = item.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = item.get("color", DEFAULT_COLOR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = str(item.get("label", "") or "")
    if not label:
        return
    font_scale = max(0.45, min(0.8, (x2 - x1) / 260.0))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, 1)
    top = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 7), color, -1)
    cv2.putText(frame, label, (x1 + 4, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (8, 12, 20), 1,
                cv2.LINE_AA)


__all__ = [
    "CONTOUR_COUNT", "CONTOUR_ORDER", "DEFAULT_COLOR",
    "as_points", "mesh_edges", "draw_face_mesh", "draw_face_box",
]
