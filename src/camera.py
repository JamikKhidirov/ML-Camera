"""Захват видеокадров из RTSP-потока, веб-камеры или файла."""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

log = logging.getLogger(__name__)


class Camera:
    """Обёртка над cv2.VideoCapture с переподключением при обрыве потока."""

    def __init__(self, source, width=1280, height=720, reconnect_delay: float = 5.0):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_error_at = 0.0
        self.fps = 0.0
        self.open()

    @staticmethod
    def _is_file(source) -> bool:
        if isinstance(source, int):
            return False
        s = str(source).lower()
        return s.endswith((".mp4", ".avi", ".mkv", ".mov", ".jpg", ".png"))

    @property
    def is_file(self) -> bool:
        """True для видеофайла/картинки (а не RTSP/веб-камеры)."""
        return self._is_file(self.source)

    def open(self) -> None:
        self._release()
        src = self.source
        log.info(f"Открываю источник: {src}")
        if isinstance(src, int):
            self._cap = cv2.VideoCapture(src)
        elif self._is_file(src):
            self._cap = cv2.VideoCapture(str(src))
        else:
            # RTSP: используем FFmpeg-бэкенд, задаём буферы
            self._cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not (self._cap and self._cap.isOpened()):
            raise RuntimeError(f"Не удалось открыть источник {src}")
        if not self._is_file(self.source):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            self.fps = 0.0

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read(self) -> Optional[cv2.Mat]:
        """Возвращает кадр BGR или None, если источник жив, но кадров нет (файл кончился)."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if ok:
            return frame
        # Источник отвалился / конец файла
        if self._is_file(self.source):
            self._release()
            return None
        now = time.time()
        if now - self._last_error_at < self.reconnect_delay:
            return None
        self._last_error_at = now
        log.warning("Поток оборвался, переподключаюсь через %.1f сек...", self.reconnect_delay)
        time.sleep(self.reconnect_delay)
        try:
            self.open()
        except Exception as exc:
            log.error("Не удалось переподключиться: %s", exc)
        return None

    def close(self) -> None:
        self._release()

    def __del__(self):
        self._release()