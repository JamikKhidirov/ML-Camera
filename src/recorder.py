"""Запись видео с камеры в ОТДЕЛЬНУЮ ПАПКУ.

Два режима (recording.mode в config):
  events      — короткие клипы вокруг события (лицо/машина/незнакомец);
                кадры постоянно складываются в кольцевой буфер, при событии
                trigger() дорисовывает ещё seconds_after сек и сохраняет mp4.
  continuous  — непрерывная запись потока; файлы ротируются каждые
                rotate_seconds сек. Работает только на ПК-версии (на VPS
                запись отключена — там без видео).

Все файлы пишутся в recording.output_dir (по умолчанию data/videos/<камера>/).
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2

log = logging.getLogger(__name__)

_FOURCC = {"mp4v": "mp4v", "avc1": "avc1", "h264": "avc1"}


class ClipRecorder:
    def __init__(self, cfg, camera_name: str):
        r = cfg.recording
        self.enabled = bool(getattr(r, "enabled", False))
        self.mode = str(getattr(r, "mode", "events")).lower()
        self.fps = max(1, int(getattr(r, "fps", 12)))
        self.seconds_before = float(getattr(r, "seconds_before", 3))
        self.seconds_after = float(getattr(r, "seconds_after", 3))
        self.rotate_seconds = int(getattr(r, "rotate_seconds", 300))
        fourcc_str = _FOURCC.get(str(getattr(r, "fourcc", "mp4v")).lower(), "mp4v")
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.camera_name = camera_name
        self.output_dir = cfg.resolve(getattr(r, "output_dir", "data/videos")) / camera_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buf: deque = deque(maxlen=int(self.seconds_before * self.fps))
        self._writer: Optional[cv2.VideoWriter] = None
        self._lock = Lock()
        self._end_at = 0.0
        self._seg_start = 0.0
        self._size: Optional[tuple] = None
        self._path: Optional[Path] = None

    # ---------- кадры ----------
    def push(self, frame) -> None:
        """Вызывается для КАЖДОГО кадра потока камеры."""
        if not self.enabled:
            return
        with self._lock:
            self._size = (frame.shape[1], frame.shape[0])
            if self.mode == "continuous":
                self._push_continuous(frame)
            else:  # events
                self._buf.append(frame.copy())
                if self._writer is not None:
                    self._writer.write(frame)
                    if time.time() >= self._end_at:
                        self._close_writer()

    def _push_continuous(self, frame) -> None:
        if self._writer is None or time.time() - self._seg_start >= self.rotate_seconds:
            self._close_writer()
            self._open_writer("stream")
        self._writer.write(frame)

    def _open_writer(self, tag: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self.output_dir / f"{tag}_{ts}.mp4"
        self._writer = cv2.VideoWriter(str(self._path), self.fourcc, self.fps, self._size)
        self._seg_start = time.time()
        log.info("Видео-файл: %s", self._path)

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:
                log.warning("Ошибка закрытия видео: %s", exc)
            self._writer = None
            self._path = None

    # ---------- событийный режим ----------
    def trigger(self, event_name: str) -> None:
        """Начать клип вокруг события (только для mode: events)."""
        if not self.enabled or self.mode != "events":
            return
        with self._lock:
            if self._writer is not None:
                return  # уже пишем другой клип
            if not self._buf or self._size is None:
                return
            self._open_writer(event_name)
            for f in self._buf:
                self._writer.write(f)
            self._end_at = time.time() + self.seconds_after

    def close(self) -> None:
        with self._lock:
            self._close_writer()