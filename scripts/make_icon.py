"""Генератор иконки MLCamera (pc/MLCamera.ico).

Рисует простую «камеру» с синим объективом. Пишет НАСТОЯЩИЙ многоразмерный
ICO (PNG-изображения внутри, поддерживается Windows Vista+) — Pillow сам
многоразмерный ICO сохранять не умеет.

Запуск:  python scripts/make_icon.py
"""
from __future__ import annotations

import io
import struct
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
    d.rounded_rectangle((0.06 * s, 0.20 * s, 0.94 * s, 0.74 * s),
                        radius=0.10 * s, fill=(28, 32, 40, 255))
    # «объектив»
    d.ellipse((0.12 * s, 0.24 * s, 0.80 * s, 0.70 * s), fill=(9, 13, 20, 255))
    d.ellipse((0.20 * s, 0.26 * s, 0.72 * s, 0.68 * s), fill=(38, 140, 255, 255))
    d.ellipse((0.32 * s, 0.38 * s, 0.48 * s, 0.54 * s), fill=(190, 235, 255, 255))
    # индикатор записи (красная точка)
    d.ellipse((0.80 * s, 0.28 * s, 0.93 * s, 0.41 * s), fill=(255, 60, 60, 255))
    # «вспышка»
    d.polygon([(0.78 * s, 0.10 * s), (0.94 * s, 0.06 * s), (0.95 * s, 0.16 * s)],
              fill=(255, 200, 90, 255))
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_ico(images: list[tuple[int, bytes]]) -> bytes:
    """Собирает ICO-файл вручную из PNG-картинок (формат PNG-in-ICO)."""
    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    entries = b""
    offset = 6 + 16 * count
    data = b""
    for size, png in images:
        entries += struct.pack("<BBBBHHII", size & 0xFF, size & 0xFF, 0, 0, 1, 32,
                               len(png), offset)
        data += png
        offset += len(png)
    return header + entries + data


def main() -> int:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    pngs = [(sz, _png_bytes(_draw(sz))) for sz in sizes]
    out = ROOT / "pc" / "MLCamera.ico"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(_build_ico(pngs))
    print(f"Иконка готова: {out} ({out.stat().st_size} байт, {len(pngs)} размеров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())