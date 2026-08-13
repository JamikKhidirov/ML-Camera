"""Генератор иконки MLCamera (pc/MLCamera.ico).

Рисует простую «камеру» с красной точкой REC. Нужен Pillow (уже в requirements).
Запуск:  python scripts/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Нужен Pillow:  pip install Pillow")


def _draw(size: int) -> Image.Image:
    s = float(size)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # корпус камеры (тёмный скруглённый прямоугольник)
    body = (0.06 * s, 0.20 * s, 0.94 * s, 0.74 * s)
    d.rounded_rectangle(body, radius=0.10 * s, fill=(28, 32, 40, 255))

    # «объектив»
    cx, cy = 0.46 * s, 0.47 * s
    d.ellipse((0.12 * s, 0.24 * s, 0.80 * s, 0.70 * s), fill=(9, 13, 20, 255))
    d.ellipse((cx - 0.20 * s, cy - 0.21 * s, cx + 0.20 * s, cy + 0.21 * s),
              fill=(38, 140, 255, 255))
    d.ellipse((cx - 0.08 * s, cy - 0.09 * s, cx + 0.08 * s, cy + 0.09 * s),
              fill=(180, 230, 255, 255))

    # индикатор записи (красная точка)
    d.ellipse((0.80 * s, 0.30 * s, 0.92 * s, 0.42 * s), fill=(255, 60, 60, 255))

    # «вспышка» справа сверху
    d.polygon([(0.78 * s, 0.16 * s), (0.92 * s, 0.10 * s), (0.94 * s, 0.18 * s)],
              fill=(255, 210, 120, 255))
    return img


def main() -> int:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    out = ROOT / "pc" / "MLCamera.ico"
    out.parent.mkdir(exist_ok=True)
    frames = [_draw(sz) for sz in sizes]
    # ico может хранить каждый размер отдельной «страницей»
    frames[0].save(out, format="ICO", sizes=[(sz, sz) for sz in sizes],
                   append_images=frames[1:])
    print(f"Иконка готова: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())