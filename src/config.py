"""Загрузка конфигурации из config/config.yaml."""
from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# В .exe (PyInstaller onedir) весь проект лежит в папке рядом с MLCamera.exe:
# там должны быть config/ и data/. В исходниках — корень репозитория.
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Тонкая обёртка над словарём YAML с доступом вида cfg.face.similarity_threshold."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            return Config(self._data[item]) if isinstance(self._data[item], dict) else self._data[item]
        raise AttributeError(f"Раздел '{item}' не найден в конфиге")

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def as_dict(self) -> dict[str, Any]:
        return {k: (v.as_dict() if isinstance(v, Config) else v) for k, v in self._data.items()}

    def resolve(self, path: str) -> Path:
        """Путь относительно корня проекта (data/... -> PROJECT_ROOT/data/...)."""
        p = Path(path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def __repr__(self) -> str:
        return f"Config({self._data.keys()})"


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else CONFIG_PATH
    # Если путь относительный и его нет в текущей папке — ищем от корня проекта.
    # Так скрипты можно запускать из любой директории, а не только из корня.
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)


def init_logging(level: str = "INFO", logfile: str | None = None) -> None:
    """Настройка логгера для всего приложения."""
    handlers: list[Any] = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("insightface").setLevel(logging.WARNING)