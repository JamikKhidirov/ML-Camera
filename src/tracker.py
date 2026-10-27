"""Трекинг объектов между кадрами (CentroidTracker) и голосование.

Зачем:
  - один человек, стоящий перед камерой 2 минуты, = ОДНО уведомление
    (а не каждые cooldown_minutes);
  - размытый номер машины собирается с нескольких кадров и берётся
    «голосованием» — самый частый результат OCR;
  - можно определять «вошёл / вышел» по пересечению линии в кадре.

Алгоритм простой (без машинного обучения):
  каждый кадр получаем центры боксов; связываем их с прошлыми центрами
  по ближайшему расстоянию. Трек, пропавший на max_disappeared кадров,
  удаляется.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Optional

import numpy as np


def majority_vote(items: list) -> Optional[object]:
    """Самый частый элемент списка (None, если список пуст)."""
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def centroid_of(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Track:
    """Один отслеживаемый объект. data — место для доп. состояния (имя, номера...)."""

    def __init__(self, track_id: int, centroid):
        self.id = track_id
        self.centroid = centroid
        self.prev_centroid: Optional[tuple] = None
        self.disappeared = 0      # сколько кадров подряд не видели
        self.seen = 0             # сколько кадров подряд видели
        self.data: dict = {}      # свободное хранилище для pipeline


class CentroidTracker:
    def __init__(self, max_disappeared: int = 20, max_distance: float = 100.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks: dict[int, Track] = {}
        # Треки, которые только что исчезли (нужны, чтобы успеть отправить
        # уведомление о машине, которая уехала до набора vote_frames).
        self.removed: list[Track] = []

    # ---------- внутреннее ----------
    def _register(self, centroid) -> Track:
        tr = Track(self.next_id, centroid)
        self.tracks[self.next_id] = tr
        self.next_id += 1
        return tr

    def _deregister(self, track_id: int) -> None:
        tr = self.tracks.pop(track_id, None)
        if tr is not None:
            self.removed.append(tr)

    def pop_removed(self) -> list[Track]:
        """Вернуть исчезнувшие треки и очистить список."""
        out, self.removed = self.removed, []
        return out

    def _match(self, centroids: list) -> list:
        """Гридди-матчинг: каждой новой точке — ближайший живой трек (жадный)."""
        tracks = list(self.tracks.values())
        if not tracks or not centroids:
            return []
        used_t, used_c = set(), set()
        pairs = []
        for ti, tc in itertools.product(tracks, range(len(centroids))):
            d = np.hypot(ti.centroid[0] - centroids[tc][0], ti.centroid[1] - centroids[tc][1])
            pairs.append((d, ti.id, tc))
        pairs.sort(key=lambda x: x[0])
        out = []
        for d, tid, cidx in pairs:
            if tid in used_t or cidx in used_c:
                continue
            if d > self.max_distance:
                continue
            used_t.add(tid)
            used_c.add(cidx)
            out.append((tid, cidx))
        return out

    # ---------- главный метод ----------
    def update(self, boxes: list) -> list:
        """Принимает боксы (x1,y1,x2,y2) текущего кадра.

        Возвращает список той же длины, что и boxes: Track (или None),
        т.е. для каждого бокса — к какому треку он относится.
        """
        if not boxes:
            # все живые треки теряют объект
            for tr in list(self.tracks.values()):
                tr.disappeared += 1
                if tr.disappeared > self.max_disappeared:
                    self._deregister(tr.id)
            return [None] * len(boxes)
        centroids = [centroid_of(b) for b in boxes]
        matches = self._match(centroids)
        matched_tracks = {tid: cidx for tid, cidx in matches}
        per_box: list = [None] * len(boxes)

        for tid, tr in self.tracks.items():
            if tid in matched_tracks:
                cidx = matched_tracks[tid]
                tr.prev_centroid = tr.centroid
                tr.centroid = centroids[cidx]
                tr.disappeared = 0
                tr.seen += 1
                per_box[cidx] = tr

        # треки без пары — «пропали» на кадр.
        # list(...) обязателен: _deregister меняет словарь, а не только значения.
        for tid, tr in list(self.tracks.items()):
            if tid not in matched_tracks:
                tr.disappeared += 1
                if tr.disappeared > self.max_disappeared:
                    self._deregister(tid)

        # новые боксы -> новые треки
        for cidx, c in enumerate(centroids):
            if per_box[cidx] is None:
                per_box[cidx] = self._register(c)
        return per_box

    def prune(self) -> None:
        """Удалить все завершённые треки (вызывается редко, чтобы не копить память)."""
        for tr in list(self.tracks.values()):
            if tr.disappeared > 0:
                self._deregister(tr.id)