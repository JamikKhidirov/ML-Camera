"""Статусы камер в JSON-файле — без тяжёлых зависимостей.

Процесс камеры периодически пишет data/db/camera_status.json, а веб-панель,
запущенная отдельно, читает его. Так панель показывает состояние камер,
даже когда не может ими управлять.

Модуль намеренно не импортирует OpenCV/YOLO/Pipeline: его читает лёгкий
процесс scripts/web_panel.py.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.config import Config

log = logging.getLogger(__name__)


def status_path(cfg: Config) -> Path:
    """Путь к общему файлу статусов (рядом с базой)."""
    return cfg.database.resolve(cfg.database.path).parent / "camera_status.json"


def write_status(cfg: Config, payload: dict) -> None:
    """Атомарно записать статусы (tmp + replace)."""
    path = status_path(cfg)
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.debug("Не удалось записать статус камер: %s", exc)


def read_status(cfg: Config) -> dict:
    """Прочитать статусы. Если файла нет — пустой список камер."""
    path = status_path(cfg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"updated_at": 0.0, "cameras": []}
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        return {"updated_at": 0.0, "cameras": []}
    data.setdefault("updated_at", 0.0)
    return data


def empty_payload() -> dict:
    return {"updated_at": time.time(), "cameras": []}
