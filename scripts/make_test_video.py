"""Генерация тестового видео (движущийся квадрат) для отладки без камеры.

Запуск: python scripts/make_test_video.py
Создаст data/test_video.mp4, затем:
  python -m src.main --config config/config.local.yaml --once
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    out = ROOT / "data/test_video.mp4"
    out.parent.mkdir(exist_ok=True)
    fps, w, h, frames_n = 25, 640, 360, 150
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames_n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(20 + i * 4) % (w - 80)
        cv2.rectangle(img, (x, 120), (x + 60, 200), (90, 90, 90), -1)   # "машина"
        cv2.circle(img, (x + 60, 160), 20, (120, 120, 120), -1)
        if i % 5 == 0:  # каждые 5 кадров чуть темнее — ночная имитация
            img = cv2.addWeighted(img, 0.4, np.zeros_like(img), 0, 30)
        writer.write(img)
    writer.release()
    print(f"Тестовое видео создано: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())