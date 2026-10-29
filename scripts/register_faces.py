"""Регистрация членов семьи в базе (эмбеддинги лиц).

Берёт вырезанные лица из data/datasets/faces/<имя>/ и сохраняет
в data/db/embeddings_<модель>.json. Запуск:
  python scripts/register_faces.py
  python scripts/register_faces.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402
from src.face_setup import register_faces  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Запомнить лица в базе")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (напр. config/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registered = register_faces(cfg)
    c = f" --config {args.config}" if args.config else ""
    print(f"\nГотово. Запускай камеру:  python main.py{c}")
    return 0 if registered else 1


if __name__ == "__main__":
    sys.exit(main())