"""Точка входа MLCamera (общая).

Просто запускает основной модуль. Все варианты запуска и настройки — через флаги
и конфиг, см. `python main.py --help` и docs/НАЧАЛО.md.

Для ПК (окно с кадрами, запись видео) удобнее:  python pc/main.py
Для сборки .exe:                              python pc/main.py --build
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
