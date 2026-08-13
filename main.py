"""Точка входа MLCamera — запуск БЕЗ Docker, прямо с диска.

САМЫЙ ПРОСТОЙ СПОСОБ ЗАПУСКА:
    python main.py

По умолчанию берёт config/config.pc.yaml (ПК-версия: превью, запись видео,
уведомления). Любой другой конфиг — флагом:

    python main.py --config config/config.vps.yaml     # серверная (без видео)
    python main.py --config config/config.local.yaml   # отладка на тест-видео

Полезные флаги (полный список — `python main.py --help`):
    --preview        живое окно с видео и рамками
    --recording      запись видео в data/videos/
    --self-test      проверить, что уведомления доходят
    --once           прогнать ~30 кадров и выйти
    --cameras 0 2    запустить только камеры 0 и 2
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# по умолчанию ПК-версия, если не указан другой конфиг
if "--config" not in sys.argv and "--self-test" not in sys.argv:
    sys.argv += ["--config", str(Path(__file__).resolve().parent / "config/config.pc.yaml")]

from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())