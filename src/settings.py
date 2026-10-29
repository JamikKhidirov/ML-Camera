"""Единые настройки MLCamera: схема, чтение/запись config.yaml, применение.

Один и тот же список полей используется:
  - веб-панелью (src/web_panel.py) — вкладка «Настройки»;
  - Tkinter-приложением (scripts/settings.py);
  - тестами.

Конфиг один: config/config.yaml. Путь хранится в Config.config_path,
поэтому и панель, и Tkinter сохраняют изменения в тот же файл.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from src.config import CONFIG_PATH, Config

log = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class SettingsError(ValueError):
    """Ошибка проверки настроек — показывается пользователю как есть."""

    def __init__(self, errors: Iterable[str]):
        self.errors = [str(e) for e in errors]
        super().__init__("; ".join(self.errors))


# ---------------------------------------------------------------------------
#  Разделы и поля
# ---------------------------------------------------------------------------
SECTIONS: list[dict] = [
    {"id": "stream", "title": "Камера и FPS", "icon": "📷",
     "hint": "Разрешение и частота кадров. FPS = 0 — камера отдаёт свой максимум."},
    {"id": "analysis", "title": "Анализ кадров", "icon": "🧠",
     "hint": "Как часто запускать ИИ. Больше пропуск — легче процессору, но меньше событий."},
    {"id": "motion", "title": "Движение", "icon": "🎯",
     "hint": "Анализ включается, когда в кадре что-то происходит."},
    {"id": "face", "title": "Лица", "icon": "👤",
     "hint": "Распознавание людей и «чертёж лица» из 106 точек."},
    {"id": "car", "title": "Машины", "icon": "🚗",
     "hint": "Номер, цвет и марка автомобиля."},
    {"id": "objects", "title": "Животные и вещи", "icon": "🐾",
     "hint": "Собаки, кошки, рюкзаки, чемоданы и оставленные сумки."},
    {"id": "recording", "title": "Запись видео", "icon": "🎬",
     "hint": "Короткие клипы вокруг событий или непрерывная запись."},
    {"id": "web", "title": "Веб-панель и поток", "icon": "🌐",
     "hint": "Адрес панели, live-поток, рамки и «чертёж лица»."},
    {"id": "notify", "title": "Уведомления", "icon": "🔔",
     "hint": "Telegram и push на телефон."},
    {"id": "digest", "title": "Дайджест", "icon": "📰",
     "hint": "Ежедневная сводка «кто приходил»."},
    {"id": "logging", "title": "Логи", "icon": "📝",
     "hint": "Подробность журнала работы."},
]

# type: int | float | bool | select | text | password | list | time
FIELDS: list[dict] = [
    # ---- камера / FPS ----
    {"path": "camera.fps", "section": "stream", "label": "FPS камеры",
     "type": "int", "min": 0, "max": 60, "step": 1, "default": 30, "unit": "кадр/с",
     "help": "Сколько кадров в секунду запрашивать у камеры. 0 — не ограничивать "
             "(камера отдаёт своё). 30 — плавно и не грузит CPU."},
    {"path": "camera.width", "section": "stream", "label": "Ширина кадра",
     "type": "int", "min": 160, "max": 3840, "step": 16, "default": 1280, "unit": "px"},
    {"path": "camera.height", "section": "stream", "label": "Высота кадра",
     "type": "int", "min": 120, "max": 2160, "step": 8, "default": 720, "unit": "px"},
    {"path": "camera.fallback_webcam", "section": "stream", "label": "Резервная веб-камера",
     "type": "bool", "default": True,
     "help": "Если IP-камеры не открылись — подключить веб-камеру ноутбука (0)."},
    {"path": "camera.night_enabled", "section": "stream", "label": "Ночной режим",
     "type": "bool", "default": True},
    {"path": "camera.night_brightness_threshold", "section": "stream",
     "label": "Порог темноты", "type": "int", "min": 0, "max": 255, "default": 60},
    {"path": "camera.frame_averaging_night", "section": "stream",
     "label": "Усреднять ночные кадры", "type": "bool", "default": True},
    {"path": "camera.frame_averaging_count", "section": "stream",
     "label": "Кадров для усреднения", "type": "int", "min": 2, "max": 10, "default": 3},

    # ---- анализ ----
    {"path": "analysis.detect_every_n_frames", "section": "analysis",
     "label": "Анализировать каждый N-й кадр", "type": "int", "min": 1, "max": 60,
     "default": 4},
    {"path": "analysis.frame_scale", "section": "analysis",
     "label": "Масштаб кадра для ИИ", "type": "float", "min": 0.1, "max": 1.0,
     "step": 0.05, "default": 0.5,
     "help": "0.5 — вдвое быстрее: YOLO и лица всё равно работают в 640 px."},
    {"path": "analysis.multi_resolution", "section": "analysis",
     "label": "Мультиразрешение", "type": "bool", "default": True},
    {"path": "analysis.small_frame_scale", "section": "analysis",
     "label": "Мелкий кадр для прочёса", "type": "float", "min": 0.1, "max": 1.0,
     "step": 0.05, "default": 0.4},
    {"path": "analysis.debug_delay", "section": "analysis",
     "label": "Пауза на кадр", "type": "float", "min": 0.0, "max": 2.0,
     "step": 0.01, "default": 0.0, "unit": "с"},

    # ---- движение ----
    {"path": "motion.enabled", "section": "motion", "label": "Детекция движения",
     "type": "bool", "default": True},
    {"path": "motion.threshold", "section": "motion", "label": "Порог движения",
     "type": "int", "min": 100, "max": 20000, "step": 50, "default": 1500},
    {"path": "motion.scale", "section": "motion", "label": "Масштаб сравнения",
     "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.25},
    {"path": "motion.force_interval_frames", "section": "motion",
     "label": "Прочёс без движения", "type": "int", "min": 1, "max": 3600,
     "default": 60, "unit": "кадров",
     "help": "Раз в N кадров анализ запускается даже без движения."},

    # ---- лица ----
    {"path": "face.enabled", "section": "face", "label": "Распознавание лиц",
     "type": "bool", "default": True},
    {"path": "face.model_name", "section": "face", "label": "Модель InsightFace",
     "type": "select", "options": ["buffalo_s", "buffalo_l"], "default": "buffalo_s"},
    {"path": "face.similarity_threshold", "section": "face", "label": "Порог сходства",
     "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "default": 0.45},
    {"path": "face.min_det_score", "section": "face", "label": "Порог детекции",
     "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "default": 0.4},
    {"path": "face.mesh_every_n_frames", "section": "face",
     "label": "Обновлять «чертёж» каждые", "type": "int", "min": 1, "max": 120,
     "default": 8, "unit": "кадров"},
    {"path": "face.cooldown_minutes", "section": "face", "label": "Кулдаун лица",
     "type": "int", "min": 0, "max": 1440, "default": 30, "unit": "мин"},
    {"path": "face.notify_unknown", "section": "face", "label": "Сообщать о незнакомцах",
     "type": "bool", "default": True},
    {"path": "face.save_unknown", "section": "face", "label": "Сохранять фото незнакомцев",
     "type": "bool", "default": True},
    {"path": "face.auto_capture_unknown", "section": "face",
     "label": "Авто-обучение незнакомцев", "type": "bool", "default": True},

    # ---- машины ----
    {"path": "car.enabled", "section": "car", "label": "Распознавание машин",
     "type": "bool", "default": True},
    {"path": "car.conf_threshold", "section": "car", "label": "Порог детекции машины",
     "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "default": 0.35},
    {"path": "car.color_enabled", "section": "car", "label": "Определять цвет",
     "type": "bool", "default": True},
    {"path": "car.brand_enabled", "section": "car", "label": "Определять марку",
     "type": "bool", "default": True},
    {"path": "car.brand_tta", "section": "car", "label": "Точная марка (TTA)",
     "type": "bool", "default": True},
    {"path": "car.brand_min_conf", "section": "car", "label": "Мин. уверенность марки",
     "type": "float", "min": 0.05, "max": 0.95, "step": 0.01, "default": 0.25},
    {"path": "car.plate_enabled", "section": "car", "label": "Читать номера",
     "type": "bool", "default": True},
    {"path": "car.plate_engine", "section": "car", "label": "Движок номеров",
     "type": "select", "options": ["tesseract", "easyocr", "fastalpr"],
     "default": "tesseract"},
    {"path": "car.plate_vote_frames", "section": "car", "label": "Кадров для голосования",
     "type": "int", "min": 1, "max": 30, "default": 5},
    {"path": "car.cooldown_minutes", "section": "car", "label": "Кулдаун машины",
     "type": "int", "min": 0, "max": 1440, "default": 60, "unit": "мин"},

    # ---- объекты ----
    {"path": "objects.enabled", "section": "objects", "label": "Животные и вещи",
     "type": "bool", "default": True},
    {"path": "objects.animals.enabled", "section": "objects", "label": "Животные",
     "type": "bool", "default": True},
    {"path": "objects.animals.conf_threshold", "section": "objects",
     "label": "Порог животного", "type": "float", "min": 0.1, "max": 0.95,
     "step": 0.01, "default": 0.35},
    {"path": "objects.animals.cooldown_minutes", "section": "objects",
     "label": "Кулдаун животного", "type": "int", "min": 0, "max": 1440,
     "default": 30, "unit": "мин"},
    {"path": "objects.packages.enabled", "section": "objects", "label": "Оставленные вещи",
     "type": "bool", "default": True},
    {"path": "objects.packages.conf_threshold", "section": "objects",
     "label": "Порог вещи", "type": "float", "min": 0.1, "max": 0.95,
     "step": 0.01, "default": 0.40},
    {"path": "objects.packages.stationary_seconds", "section": "objects",
     "label": "Вещь «оставлена» через", "type": "int", "min": 5, "max": 600,
     "default": 45, "unit": "с"},
    {"path": "objects.packages.cooldown_minutes", "section": "objects",
     "label": "Кулдаун вещи", "type": "int", "min": 0, "max": 1440,
     "default": 120, "unit": "мин"},

    # ---- запись ----
    {"path": "recording.enabled", "section": "recording", "label": "Запись видео",
     "type": "bool", "default": False},
    {"path": "recording.mode", "section": "recording", "label": "Режим",
     "type": "select", "options": ["events", "continuous"], "default": "events",
     "help": "events — клипы вокруг события; continuous — непрерывно."},
    {"path": "recording.fps", "section": "recording", "label": "FPS записи",
     "type": "int", "min": 1, "max": 60, "default": 12, "unit": "кадр/с"},
    {"path": "recording.seconds_before", "section": "recording",
     "label": "Секунд до события", "type": "int", "min": 0, "max": 60, "default": 3},
    {"path": "recording.seconds_after", "section": "recording",
     "label": "Секунд после события", "type": "int", "min": 0, "max": 120, "default": 3},
    {"path": "recording.rotate_seconds", "section": "recording",
     "label": "Длина файла", "type": "int", "min": 10, "max": 3600, "default": 300,
     "unit": "с"},

    # ---- веб ----
    {"path": "web.stream_fps", "section": "web", "label": "FPS live-потока",
     "type": "int", "min": 1, "max": 30, "default": 10, "unit": "кадр/с",
     "help": "Сколько кадров в секунду браузер получает из панели."},
    {"path": "web.stream_quality", "section": "web", "label": "Качество JPEG",
     "type": "int", "min": 30, "max": 95, "default": 70, "unit": "%"},
    {"path": "web.overlay", "section": "web", "label": "Рамки объектов",
     "type": "bool", "default": True},
    {"path": "web.face_mesh", "section": "web", "label": "«Чертёж лица» (106 точек)",
     "type": "bool", "default": True},
    {"path": "web.autostart", "section": "web", "label": "Запускать панель вместе с камерой",
     "type": "bool", "default": True},
    {"path": "web.open_browser", "section": "web",
     "label": "Открывать панель в браузере", "type": "bool", "default": True,
     "help": "При запуске main.py браузер сам откроет страницу панели."},
    {"path": "web.control", "section": "web", "label": "Разрешить управление камерами",
     "type": "bool", "default": True},
    {"path": "web.host", "section": "web", "label": "Адрес панели", "type": "text",
     "default": "127.0.0.1",
     "help": "127.0.0.1 — только этот компьютер; 0.0.0.0 — доступ с телефона."},
    {"path": "web.port", "section": "web", "label": "Порт панели", "type": "int",
     "min": 1024, "max": 65535, "default": 8088},
    {"path": "web.token", "section": "web", "label": "Пароль панели", "type": "password",
     "default": "",
     "help": "Пусто — без пароля. Обязательно задай, если открываешь панель наружу."},

    # ---- уведомления ----
    {"path": "notify.min_interval_seconds", "section": "notify",
     "label": "Мин. интервал", "type": "int", "min": 0, "max": 3600,
     "default": 30, "unit": "с"},
    {"path": "notify.telegram.enabled", "section": "notify", "label": "Telegram",
     "type": "bool", "default": True},
    {"path": "notify.telegram.bot_token", "section": "notify", "label": "Токен бота",
     "type": "password", "default": ""},
    {"path": "notify.telegram.chat_ids", "section": "notify", "label": "Chat ID",
     "type": "list", "default": ["111111111"],
     "help": "Несколько получателей — через запятую."},
    {"path": "notify.push.enabled", "section": "notify", "label": "Push (ntfy)",
     "type": "bool", "default": True},
    {"path": "notify.push.topic", "section": "notify", "label": "Тема ntfy",
     "type": "text", "default": "my_camera_unique_secret_topic"},
    {"path": "notify.delete_images_after_send", "section": "notify",
     "label": "Удалять фото после отправки", "type": "bool", "default": False},

    # ---- дайджест ----
    {"path": "digest.enabled", "section": "digest", "label": "Ежедневный дайджест",
     "type": "bool", "default": True},
    {"path": "digest.time", "section": "digest", "label": "Время отправки",
     "type": "time", "default": "09:00"},
    {"path": "digest.hours", "section": "digest", "label": "За сколько часов",
     "type": "int", "min": 1, "max": 168, "default": 24, "unit": "ч"},

    # ---- логи ----
    {"path": "logging.level", "section": "logging", "label": "Уровень логов",
     "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
     "default": "INFO"},
]

_FIELD_BY_PATH = {f["path"]: f for f in FIELDS}


# ---------------------------------------------------------------------------
#  Доступ к вложенным значениям
# ---------------------------------------------------------------------------
def get_path(data: dict, path: str, default: Any = None) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return copy.deepcopy(default)
        node = node[part]
    return copy.deepcopy(node)


def set_path(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on", "да", "вкл", "y")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[,;\n]+", str(value))
    return [x.strip() for x in items if x.strip()]


def _coerce(field: dict, value: Any) -> Any:
    kind = field["type"]
    default = field.get("default")
    if kind == "bool":
        return _as_bool(value)
    if kind == "int":
        try:
            number = int(float(str(value).strip().replace(",", ".")))
        except (TypeError, ValueError):
            number = int(default or 0)
        if "min" in field:
            number = max(int(field["min"]), number)
        if "max" in field:
            number = min(int(field["max"]), number)
        return number
    if kind == "float":
        try:
            number = float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            number = float(default or 0.0)
        if "min" in field:
            number = max(float(field["min"]), number)
        if "max" in field:
            number = min(float(field["max"]), number)
        return round(number, 3)
    if kind == "select":
        text = str(value).strip()
        options = [str(o) for o in field.get("options") or []]
        if options and text not in options:
            raise ValueError(f"{field['label']}: допустимо {', '.join(options)}")
        return text
    if kind == "list":
        return _as_list(value)
    if kind == "time":
        text = str(value).strip()
        if not _TIME_RE.match(text):
            raise ValueError(f"{field['label']}: время в формате ЧЧ:ММ")
        hour, minute = text.split(":")
        return f"{int(hour):02d}:{int(minute):02d}"
    # text / password
    return value if kind == "password" else str(value).strip()


def validate_values(values: dict) -> dict[str, Any]:
    """Проверить и привести значения к типам. Бросает SettingsError."""
    clean: dict[str, Any] = {}
    errors: list[str] = []
    for path, value in (values or {}).items():
        field = _FIELD_BY_PATH.get(path)
        if field is None:
            continue                      # чужие поля игнорируем
        try:
            clean[path] = _coerce(field, value)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise SettingsError(errors)
    return clean


def normalize_cameras(cameras: Any) -> list[dict]:
    """Привести список камер к виду {name, source, enabled}.

    source "0" (строка) превращается в число 0 — это веб-камера.
    """
    result: list[dict] = []
    if not isinstance(cameras, list):
        return result
    for i, cam in enumerate(cameras):
        if not isinstance(cam, dict):
            continue
        name = str(cam.get("name", "") or "").strip()
        source = cam.get("source", 0)
        if isinstance(source, str):
            source = source.strip()
            if source.isdigit():
                source = int(source)
        if not name:
            name = f"Камера {i + 1}"
        if source == "":
            source = 0
        result.append({
            "name": name,
            "source": source,
            "enabled": _as_bool(cam.get("enabled", True)),
        })
    return result


# ---------------------------------------------------------------------------
#  Полезная нагрузка для панели / Tkinter
# ---------------------------------------------------------------------------
def public_fields() -> list[dict]:
    """Поля без служебных ключей — можно отдавать в JSON."""
    return [dict(f) for f in FIELDS]


def settings_payload(cfg: Config) -> dict:
    raw = copy.deepcopy(cfg._data)
    values = {f["path"]: get_path(raw, f["path"], f.get("default")) for f in FIELDS}
    return {
        "values": values,
        "cameras": normalize_cameras(raw.get("cameras") or []),
        "sections": [dict(s) for s in SECTIONS],
        "fields": public_fields(),
        "config_path": str(cfg.config_path),
        "stream": {
            "fps": int(get_path(raw, "web.stream_fps", 10)),
            "quality": int(get_path(raw, "web.stream_quality", 70)),
        },
        "restartable": True,
    }


# ---------------------------------------------------------------------------
#  Сохранение
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# =====================================================================\n"
        "#  MLCamera — единственный конфиг. Сохранён из панели/настроек "
        f"{datetime.now():%d.%m.%Y %H:%M}.\n"
        "#  Все параметры можно менять во вкладке «Настройки» веб-панели\n"
        "#  или в приложении scripts/settings.py (Tkinter).\n"
        "# =====================================================================\n\n"
    )
    text += yaml.safe_dump(data, allow_unicode=True, sort_keys=False,
                           default_flow_style=False, width=110)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            log.debug("Не удалось сделать резервную копию конфига", exc_info=True)
    os.replace(tmp, path)


def save_settings(cfg: Config, values: dict,
                  cameras: Optional[list[dict]] = None) -> dict:
    """Проверить, записать в config.yaml и обновить cfg в памяти.

    Возвращает новый сырой словарь конфига. Бросает SettingsError.
    """
    raw = copy.deepcopy(cfg._data)
    clean = validate_values(values or {})
    for path, value in clean.items():
        set_path(raw, path, value)

    if cameras is not None:
        norm = normalize_cameras(cameras)
        for i, cam in enumerate(norm):
            if cam["source"] == "" or cam["source"] is None:
                raise SettingsError([f"Камера {i + 1}: укажи источник (0 или RTSP-адрес)"])
        raw["cameras"] = norm

    path = cfg.config_path
    _write_yaml(path, raw)

    # Тот же объект cfg используют все камеры и панель — обновляем на месте.
    cfg._data.clear()
    cfg._data.update(raw)
    log.info("Настройки сохранены: %s", path)
    return raw


# ---------------------------------------------------------------------------
#  Применение без перезапуска процесса
# ---------------------------------------------------------------------------
def apply_live(cfg: Config, manager=None) -> list[str]:
    """Применить то, что можно без перезапуска: рамки и «чертёж лица»."""
    notes: list[str] = []
    if manager is None:
        return notes
    web = cfg.get("web", {}) or {}
    overlay = bool(web.get("overlay", True))
    mesh = bool(web.get("face_mesh", True))
    for item in manager:
        pipe = getattr(item, "pipe", None)
        if pipe is None:
            continue
        pipe.cfg = cfg
        pipe.overlay_enabled = overlay
        pipe.face_mesh_enabled = mesh
        notes.append(item.name)
    return notes


def reload_cameras_async(manager, start: bool = True) -> None:
    """Перезапустить камеры в фоне, чтобы новые настройки вступили в силу."""
    if manager is None:
        return
    import threading

    def _run():
        try:
            manager.reload(start=start)
        except Exception:
            log.exception("Не удалось перезапустить камеры после смены настроек")

    threading.Thread(target=_run, name="mlcamera-settings-reload", daemon=True).start()


def apply_to_running_panel(cfg: Config, values: dict,
                           cameras: Optional[list[dict]] = None,
                           timeout: float = 4.0) -> tuple[bool, str]:
    """Tkinter → работающая панель: отправить настройки по HTTP.

    Нужно, чтобы «Сохранить» в Tkinter сразу применялось к запущенной камере.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    web = cfg.get("web", {}) or {}
    host = str(web.get("host", "127.0.0.1"))
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    port = int(web.get("port", 8088))
    token = str(web.get("token", "") or "")
    url = f"http://{host}:{port}/api/settings"
    if token:
        url += "?token=" + urllib.parse.quote(token)
    body = json.dumps({"values": values, "cameras": cameras,
                       "restart": True}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-MLCamera": "1"})
    if token:
        req.add_header("X-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("ok")), str(data.get("message") or "применено")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError) as exc:
        return False, str(exc)


__all__ = [
    "SECTIONS", "FIELDS", "SettingsError",
    "settings_payload", "save_settings", "validate_values", "normalize_cameras",
    "apply_live", "reload_cameras_async", "apply_to_running_panel",
    "get_path", "set_path",
]
