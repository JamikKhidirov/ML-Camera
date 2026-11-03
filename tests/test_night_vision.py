"""Ночное усиление: ярче, но без лилового шума матрицы."""
import cv2
import numpy as np

from src.night_vision import enhance_low_light, ensure_night_quality, is_low_light


def _dark_purple() -> np.ndarray:
    frame = np.full((80, 80, 3), (60, 10, 60), dtype=np.uint8)
    frame[20:60, 20:60] = (90, 80, 100)
    return frame


def test_low_light_detected():
    assert is_low_light(_dark_purple(), threshold=60)
    assert not is_low_light(np.full((40, 40, 3), 200, dtype=np.uint8),
                            threshold=60)


def test_enhance_makes_frame_brighter():
    frame = _dark_purple()
    out = enhance_low_light(frame)
    assert out.shape == frame.shape
    assert float(out.mean()) > float(frame.mean())


def test_enhance_suppresses_colour_noise():
    """Раньше насыщенность повышалась, и ИК-кадр становился фиолетовым."""
    frame = _dark_purple()
    before = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())
    after = float(cv2.cvtColor(enhance_low_light(frame),
                               cv2.COLOR_BGR2HSV)[..., 1].mean())
    assert after < before * 0.6


def test_bright_frame_is_returned_unchanged():
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    changed, out = ensure_night_quality(frame, threshold=60)
    assert changed is False
    assert out is frame
