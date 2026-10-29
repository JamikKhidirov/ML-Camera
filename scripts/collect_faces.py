"""Сбор датасета лиц: вырезает лица из сырых фото.

Как пользоваться:
  1) Кинь 20-30 фотографий КАЖДОГО члена семьи в папки:
       data/datasets/raw/мама/    data/datasets/raw/папа/
       data/datasets/raw/я/       data/datasets/raw/сестра1/
  2) Запусти:  python scripts/collect_faces.py   (или python main.py --register)
  3) Скрипт вырежет лица в data/datasets/faces/<имя>/

После этого зарегистрируй лица в базе:  python scripts/register_faces.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402
from src.face_setup import collect_faces  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Вырезать лица из сырых фото")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (напр. config/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    total = collect_faces(cfg)
    if total:
        c = f" --config {args.config}" if args.config else ""
        print(f"Теперь: python scripts/register_faces.py{c}  (или python main.py{c} --register)")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())