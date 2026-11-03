"""Ярлык для запуска окна камеры Tkinter: python scripts/camera_view.py.

То же самое: ``python main.py --view`` или ``camera.bat``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera_window import main  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
