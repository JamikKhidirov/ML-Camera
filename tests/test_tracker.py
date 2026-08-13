"""Тесты CentroidTracker и голосования по номерам."""
from src.tracker import CentroidTracker, majority_vote


def _box(cx, cy, size=20):
    return (cx - size, cy - size, cx + size, cy + size)


def test_majority_vote():
    assert majority_vote([]) is None
    assert majority_vote(["A", "B", "A", "A", "C"]) == "A"
    assert majority_vote([1, 2, 2, 2, 3]) == 2


def test_tracker_keeps_same_object():
    tr = CentroidTracker(max_disappeared=5)
    t1 = tr.update([_box(100, 100)])[0]
    assert t1 is not None and t1.seen == 0
    t2 = tr.update([_box(102, 101)])[0]
    assert t2 is t1, "тот же объект должен сохранять тот же трек"
    assert t1.seen == 1


def test_tracker_new_object_get_new_id():
    tr = CentroidTracker(max_disappeared=5)
    a = tr.update([_box(0, 0)])[0]
    b = tr.update([_box(300, 300), _box(0, 0)])
    assert b[1] is a
    assert b[0] is not a


def test_tracker_disappears():
    tr = CentroidTracker(max_disappeared=3)
    t = tr.update([_box(50, 50)])[0]
    for _ in range(3):          # объект пропал на 3 кадра (считается)
        tr.update([])
    alive = [x for x in tr.update([_box(50, 50)]) if x is not None]
    assert alive[0] is t        # ещё в пределах max_disappeared=3 -> тот же трек
    for _ in range(5):          # пропал дольше лимита
        tr.update([])
    t2 = tr.update([_box(50, 50)])[0]
    assert t2 is not t          # лимит превышен -> новый трек


def test_tracks_aligned_with_boxes():
    tr = CentroidTracker(max_disappeared=5)
    res = tr.update([_box(10, 10), _box(200, 200)])
    assert len(res) == 2
    assert all(x is not None for x in res)
    ids = {r.id for r in res if r}
    assert len(ids) == 2