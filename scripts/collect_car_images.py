"""Собрать фото машин из событий в папку для разметки датасета марок.

Камера сохраняет фото каждой машины в data/events/<камера>/<дата>/car_*.jpg.
Этот скрипт копирует их в data/datasets/cars/inbox/ без дублей, чтобы потом
вручную разложить по маркам:

    data/datasets/cars/train/Lada/
    data/datasets/cars/train/Kia/
    ...

Запуск:
    python scripts/collect_car_images.py
    python scripts/collect_car_images.py --source data/events --out data/datasets/cars/inbox
    python scripts/collect_car_images.py --move      # перенести, а не копировать
    python scripts/collect_car_images.py --clean     # очистить inbox перед сбором

Дальше: docs/BRAND_RU.md и scripts/train_brand_classifier.py.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать фото машин для датасета марок")
    parser.add_argument("--source", default="data/events", help="папка событий камеры")
    parser.add_argument("--out", default="data/datasets/cars/inbox", help="куда складывать")
    parser.add_argument("--move", action="store_true", help="перенести файлы вместо копирования")
    parser.add_argument("--clean", action="store_true", help="очистить папку назначения")
    parser.add_argument("--limit", type=int, default=0, help="максимум файлов (0 = все)")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not source.is_absolute():
        source = ROOT / source
    if not out.is_absolute():
        out = ROOT / out
    if not source.is_dir():
        print(f"Нет папки с событиями: {source}")
        return 1

    if args.clean and out.is_dir():
        shutil.rmtree(out)
        print(f"Очищено: {out}")
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for existing in out.glob("*"):
        if existing.is_file():
            try:
                seen.add(_digest(existing))
            except OSError:
                pass

    candidates = []
    for path in source.rglob("car_*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime)

    copied, skipped = 0, 0
    for path in candidates:
        if args.limit and copied >= args.limit:
            break
        try:
            digest = _digest(path)
        except OSError:
            skipped += 1
            continue
        if digest in seen:
            skipped += 1
            continue
        seen.add(digest)
        target = out / f"{path.parent.name}_{path.name}"
        if target.exists():
            target = out / f"{path.parent.name}_{path.stem}_{copied}{path.suffix}"
        try:
            if args.move:
                shutil.move(str(path), str(target))
            else:
                shutil.copy2(path, target)
            copied += 1
        except OSError as exc:
            print(f"Пропуск {path}: {exc}")
            skipped += 1

    print(f"Готово: добавлено {copied}, пропущено дублей/ошибок {skipped}")
    print(f"Папка: {out}")
    print("Разложи фото по маркам:")
    print(f"  {out.parent / 'train' / 'Lada'}/*.jpg")
    print(f"  {out.parent / 'train' / 'Kia'}/*.jpg")
    print("Затем: python scripts/train_brand_classifier.py --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
