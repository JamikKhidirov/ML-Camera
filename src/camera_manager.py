"""Управление камерами во время работы: статус, старт/стоп, снимок кадра.

Зачем: камеры должны запускаться в отдельных потоках, чтобы веб-панель могла
показать их состояние, остановить поток и снова запустить без перезапуска
всего процесса. Модели (YOLO, InsightFace, ALPR) при этом остаются в памяти.

Используется:
  - src/main.py            — поднимает камеры через CameraManager;
  - src/web_panel.py       — читает статусы и вызывает start/stop/snapshot.

Если панель запущена отдельным процессом (scripts/web_panel.py), управлять
камерами она не может: статусы читаются из data/db/camera_status.json,
который пишет процесс камеры.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from src.camera_status import write_status
from src.config import Config
from src.database import Database
from src.notifier import Notifier
from src.pipeline import Pipeline

log = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_STARTING = "starting"
STATUS_STOPPING = "stopping"
STATUS_STOPPED = "stopped"
STATUS_ERROR = "error"


class CameraRuntime:
    """Одна камера: поток Pipeline + состояние для панели."""

    def __init__(self, index: int, cfg: Config, camera_cfg, db: Database,
                 notifier: Notifier, pipeline_factory=Pipeline):
        self.index = int(index)
        self.cfg = cfg
        self.camera_cfg = camera_cfg if isinstance(camera_cfg, dict) else {}
        self.name = str(self.camera_cfg.get("name", f"Камера {self.index + 1}"))
        self.source = str(self.camera_cfg.get("source", ""))
        self.enabled_in_config = bool(self.camera_cfg.get("enabled", True))
        self.db = db
        self.notifier = notifier
        self._factory = pipeline_factory

        self.pipe: Optional[Pipeline] = None
        self.thread: Optional[threading.Thread] = None
        self._state = STATUS_STOPPED
        self._error = ""
        self._started_at = 0.0
        self._frames_at_start = 0
        self._stop_requested = False
        self._lock = threading.Lock()
        self._info_lock = threading.Lock()
        self._fps_ts = 0.0
        self._fps_frames = 0
        self._proc_fps = 0.0

    # ---------- состояние ----------
    @property
    def alive(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive()

    @property
    def source_kind(self) -> str:
        """Тип источника без логина/пароля и пути: rtsp / http / webcam / file."""
        src = self.source.strip().lower()
        if src.startswith(("rtsp://", "rtsps://")):
            return "rtsp"
        if src.startswith(("http://", "https://")):
            return "http"
        if src.isdigit():
            return "webcam"
        return "file"

    @property
    def status(self) -> str:
        state = self._state
        if state == STATUS_STARTING and self.pipe is not None:
            if self.pipe.frames_processed > self._frames_at_start:
                return STATUS_RUNNING
        return state

    def info(self) -> dict:
        pipe = self.pipe
        fps = 0.0
        width = height = 0
        frames = pipe.frames_processed if pipe is not None else 0
        if pipe is not None:
            cam = getattr(pipe, "cam", None)
            try:
                fps = float(getattr(cam, "fps", 0.0) or 0.0)
            except (TypeError, ValueError):
                fps = 0.0
            size = getattr(pipe, "frame_size", (0, 0)) or (0, 0)
            try:
                width, height = int(size[0]), int(size[1])
            except (TypeError, ValueError, IndexError):
                width = height = 0

        # Реальная скорость обработки (а не паспортный fps камеры).
        now = time.time()
        with self._info_lock:
            if (pipe is not None and self._fps_ts
                    and frames > self._fps_frames and now > self._fps_ts):
                inst = (frames - self._fps_frames) / (now - self._fps_ts)
                self._proc_fps = (inst if self._proc_fps <= 0
                                  else self._proc_fps * 0.6 + inst * 0.4)
            self._fps_ts = now
            self._fps_frames = frames
            proc_fps = self._proc_fps if self.alive else 0.0

        rec = self.cfg.get("recording", None)
        recording = bool(rec.get("enabled", False)) if isinstance(rec, dict) else False
        recording_mode = str(rec.get("mode", "")) if isinstance(rec, dict) else ""
        camera_cfg = self.cfg.get("camera", {}) or {}
        target_fps = int(camera_cfg.get("fps", 0) or 0) if isinstance(camera_cfg, dict) else 0
        started_at = self._started_at
        uptime = max(0.0, now - started_at) if started_at and self.alive else 0.0

        return {
            "index": self.index,
            "name": self.name,
            "enabled": self.enabled_in_config,
            "status": self.status,
            "alive": self.alive,
            "fps": round(fps, 1),
            "target_fps": target_fps,
            "proc_fps": round(proc_fps, 1),
            "frames": frames,
            "last_frame_ts": pipe.last_frame_ts if pipe is not None else 0.0,
            "has_image": bool(pipe is not None and pipe.last_frame_ts > 0),
            "width": width,
            "height": height,
            "source_kind": self.source_kind,
            "recording": recording,
            "recording_mode": recording_mode,
            "started_at": started_at,
            "uptime": round(uptime, 0),
            "error": self._error,
        }

    # ---------- управление ----------
    def start(self, max_frames: Optional[int] = None) -> tuple[bool, str]:
        with self._lock:
            if self.alive:
                return False, "Камера уже запущена"
            self._error = ""
            self._stop_requested = False
            self._state = STATUS_STARTING
            self._started_at = time.time()
            self._frames_at_start = (self.pipe.frames_processed
                                     if self.pipe is not None else 0)
            self.thread = threading.Thread(
                target=self._run, args=(max_frames,),
                name=f"mlcamera-cam-{self.index}", daemon=True)
            self.thread.start()
            return True, ""

    def _run(self, max_frames: Optional[int]) -> None:
        try:
            if self.pipe is None:
                self.pipe = self._factory(
                    self.cfg, self.camera_cfg, self.db, self.notifier)
            else:
                # модели уже загружены — переоткрываем только источник
                self.pipe.prepare_start()
            self._frames_at_start = self.pipe.frames_processed
            if self._stop_requested:
                self.pipe.stop()
            self.pipe.run(max_frames)
            if self._state != STATUS_STOPPING:
                self._state = STATUS_STOPPED
        except Exception as exc:  # камера недоступна, битый поток и т.п.
            self._error = str(exc) or exc.__class__.__name__
            self._state = STATUS_ERROR
            log.exception("Камера '%s' остановилась с ошибкой", self.name)
        finally:
            if self._state == STATUS_STOPPING:
                self._state = STATUS_STOPPED

    def stop(self, timeout: float = 5.0) -> tuple[bool, str]:
        with self._lock:
            self._stop_requested = True
            if not self.alive:
                if self._state != STATUS_ERROR:
                    self._state = STATUS_STOPPED
                return False, "Камера уже остановлена"
            self._state = STATUS_STOPPING
            pipe = self.pipe
        if pipe is not None:
            pipe.stop()
        thread = self.thread
        if thread is not None:
            thread.join(timeout)
        if thread is not None and thread.is_alive():
            return True, "Останавливается…"
        self._state = STATUS_STOPPED
        return True, ""

    def snapshot(self, quality: int = 85) -> Optional[bytes]:
        pipe = self.pipe
        return pipe.snapshot_jpeg(quality) if pipe is not None else None


class CameraManager:
    """Набор камер процесса: запуск, остановка, статусы, снимки."""

    def __init__(self, cfg: Config, db: Database, notifier: Notifier,
                 cameras=None, pipeline_factory=Pipeline):
        if cameras is None:
            cameras = cfg.get("cameras", [])
            if isinstance(cameras, dict):
                cameras = cameras.get("cameras") or []
        self.cfg = cfg
        self.db = db
        self.notifier = notifier
        self._factory = pipeline_factory
        self.items = [CameraRuntime(i, cfg, cam, db, notifier, pipeline_factory)
                      for i, cam in enumerate(cameras or [])]
        self._writer_stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None

    # ---------- доступ ----------
    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def get(self, index: int) -> Optional[CameraRuntime]:
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    @property
    def any_alive(self) -> bool:
        return any(item.alive for item in self.items)

    def list(self) -> list[dict]:
        return [item.info() for item in self.items]

    def add_camera(self, name: str, source, enabled: bool = True) -> CameraRuntime:
        """Добавить камеру на лету (например, веб-камеру ноутбука)."""
        camera_cfg = {"name": name, "source": source, "enabled": enabled}
        item = CameraRuntime(len(self.items), self.cfg, camera_cfg, self.db,
                             self.notifier, self._factory)
        self.items.append(item)
        log.info("Добавлена камера '%s' (источник: %s)", name, source)
        return item

    def reload(self, cameras=None, start: bool = True) -> int:
        """Пересоздать камеры из нового списка (после сохранения настроек).

        Останавливает текущие потоки, подхватывает имена/источники/enabled
        и запускает включённые камеры заново. Модели при этом перезагружаются
        только у новых Pipeline — это неизбежно при смене разрешения/FPS.
        """
        self.stop_all()
        if cameras is None:
            cameras = self.cfg.get("cameras", [])
            if isinstance(cameras, dict):
                cameras = cameras.get("cameras") or []
        self.items = [CameraRuntime(i, self.cfg, cam, self.db, self.notifier,
                                    self._factory)
                      for i, cam in enumerate(cameras or [])]
        log.info("Камеры перезагружены из настроек: %d", len(self.items))
        if start:
            return self.start_enabled()
        return 0

    # ---------- запуск/остановка ----------
    def start(self, index: int, max_frames: Optional[int] = None) -> tuple[bool, str]:
        item = self.get(index)
        if item is None:
            return False, "Нет такой камеры"
        return item.start(max_frames)

    def start_enabled(self, max_frames: Optional[int] = None) -> int:
        started = 0
        for item in self.items:
            if item.enabled_in_config and item.start(max_frames)[0]:
                started += 1
        return started

    def start_all(self, max_frames: Optional[int] = None) -> int:
        return sum(1 for item in self.items if item.start(max_frames)[0])

    def stop(self, index: int, timeout: float = 5.0) -> tuple[bool, str]:
        item = self.get(index)
        if item is None:
            return False, "Нет такой камеры"
        return item.stop(timeout)

    def stop_all(self, timeout: float = 5.0) -> None:
        for item in self.items:
            try:
                item.stop(timeout)
            except Exception:
                log.exception("Не удалось остановить камеру '%s'", item.name)

    def snapshot(self, index: int, quality: int = 85) -> Optional[bytes]:
        item = self.get(index)
        return item.snapshot(quality) if item is not None else None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Дождаться завершения всех потоков. True — все остановились."""
        deadline = None if timeout is None else time.time() + timeout
        while self.any_alive:
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.15)
        return True

    # ---------- статусы для отдельной панели ----------
    def status_payload(self) -> dict:
        return {"updated_at": time.time(), "cameras": self.list()}

    def write_status(self) -> None:
        write_status(self.cfg, self.status_payload())

    def start_status_writer(self, interval: float = 2.0) -> None:
        if self._writer_thread is not None:
            return
        self.write_status()

        def _loop():
            while not self._writer_stop.wait(interval):
                self.write_status()
            self.write_status()

        self._writer_thread = threading.Thread(
            target=_loop, name="mlcamera-status", daemon=True)
        self._writer_thread.start()

    def stop_status_writer(self) -> None:
        self._writer_stop.set()
        thread = self._writer_thread
        if thread is not None:
            thread.join(2.0)
        self._writer_thread = None


__all__ = [
    "CameraManager",
    "CameraRuntime",
    "STATUS_RUNNING",
    "STATUS_STARTING",
    "STATUS_STOPPING",
    "STATUS_STOPPED",
    "STATUS_ERROR",
]
