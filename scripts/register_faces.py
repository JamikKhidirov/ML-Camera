"""Регистрация членов семьи в базе (эмбеддинги лиц).

Берёт вырезанные лица из data/datasets/faces/<имя>/ и сохраняет
в data/db/embeddings.json. Запуск: python scripts/register_faces.py
"""
from __future__ import annotations

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
    registered = register_faces(load_config())
    print("\nГотово. Запускай камеру:  python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())