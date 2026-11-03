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

    def __init__(self, data: dict[str, Any], path: str | Path | None = None):
        self._data = data
        # Откуда прочитан конфиг: по этому пути веб-панель и Tkinter сохраняют
        # изменения. Для тестов можно передать временный путь.
        self._path = Path(path).resolve() if path else CONFIG_PATH

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

    @property
    def config_path(self) -> Path:
        """Путь к YAML-файлу, из которого загружен конфиг."""
        return self._path


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str) -> bool | None:
    value = _env_text(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on", "да", "вкл")


def _env_int(name: str) -> int | None:
    value = _env_text(name)
    if value is None:
        return None
    try:
        return int(float(value.replace(",", ".")))
    except ValueError:
        return None


def apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Переопределить часть конфига переменными окружения (удобно в Docker).

    MLCAMERA_CONFIG               — путь к config.yaml
    MLCAMERA_WEB_HOST             — 0.0.0.0, чтобы панель была доступна с хоста
    MLCAMERA_WEB_PORT             — порт панели
    MLCAMERA_WEB_TOKEN            — пароль панели
    MLCAMERA_OPEN_BROWSER         — 0 в контейнере (браузера там нет)
    MLCAMERA_CAMERA_FPS           — FPS камеры
    MLCAMERA_FALLBACK_WEBCAM      — 0 в контейнере (веб-камеры там нет)
    MLCAMERA_CAMERA_SOURCE        — заменить список камер одной: rtsp://… или 0
    MLCAMERA_CAMERA_NAME          — имя этой камеры
    """
    if not isinstance(data, dict):
        return data

    web = data.get("web")
    if not isinstance(web, dict):
        web = {}
        data["web"] = web
    host = _env_text("MLCAMERA_WEB_HOST")
    if host:
        web["host"] = host
    port = _env_int("MLCAMERA_WEB_PORT")
    if port:
        web["port"] = port
    token = _env_text("MLCAMERA_WEB_TOKEN")
    if token:
        web["token"] = token
    open_browser = _env_bool("MLCAMERA_OPEN_BROWSER")
    if open_browser is not None:
        web["open_browser"] = open_browser

    camera = data.get("camera")
    if not isinstance(camera, dict):
        camera = {}
        data["camera"] = camera
    fps = _env_int("MLCAMERA_CAMERA_FPS")
    if fps is not None:
        camera["fps"] = max(0, fps)
    fallback = _env_bool("MLCAMERA_FALLBACK_WEBCAM")
    if fallback is not None:
        camera["fallback_webcam"] = fallback

    source = _env_text("MLCAMERA_CAMERA_SOURCE")
    if source:
        if source.isdigit():
            source = int(source)
        name = _env_text("MLCAMERA_CAMERA_NAME") or "Камера"
        data["cameras"] = [{"name": name, "source": source, "enabled": True}]
    return data


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("MLCAMERA_CONFIG") or CONFIG_PATH)
    # Если путь относительный и его нет в текущей папке — ищем от корня проекта.
    # Так скрипты можно запускать из любой директории, а не только из корня.
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = apply_env_overrides(data)
    return Config(data, path=cfg_path)


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