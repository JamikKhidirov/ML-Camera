"""Создание структуры папок для датасета и БД.

Запуск:
  python scripts/make_dirs.py                          # 5 человек по умолчанию
  python scripts/make_dirs.py --names Аня Оля Толя     # свои имена
Сколько фотографий каждого человека нужно — см. docs/DATASET.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PERSONS = ["Алексей", "Ольга", "Сергей", "Аня", "Катя"]

DIRS = [
    "data/datasets/raw",
    "data/datasets/faces",
    "data/datasets/cars",
    "data/datasets/plates",
    "data/datasets/auto_unknown",
    "data/models",
    "data/db",
    "data/events",
    "data/logs",
    "data/snapshots",
    "data/videos",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None,
                        help="имена членов семьи (папки). По умолчанию: " + ", ".join(DEFAULT_PERSONS))
    args = parser.parse_args()
    persons = args.names if args.names else DEFAULT_PERSONS

    for d in DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"[ok] {p.relative_to(ROOT)}")

    for name in persons:
        (ROOT / "data/datasets/raw" / name).mkdir(parents=True, exist_ok=True)
        (ROOT / "data/datasets/faces" / name).mkdir(parents=True, exist_ok=True)
        print(f"[ok] data/datasets/raw/{name}   (сюда кладёшь фото)")
        print(f"[ok] data/datasets/faces/{name} (сюда скрипт положит вырезанные лица)")

    print("\nГотово. Собери фото и запусти: python scripts/collect_faces.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())