"""Захват видеокадров из RTSP-потока, веб-камеры или файла."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2

log = logging.getLogger(__name__)


class Camera:
    """Обёртка над cv2.VideoCapture с переподключением при обрыве потока."""

    def __init__(self, source, width=1280, height=720, reconnect_delay: float = 5.0,
                 fps: int = 0, backend: str = "auto"):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.target_fps = int(fps or 0)
        self.backend = str(backend or "auto").lower()
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_error_at = 0.0
        self._next_at = 0.0
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

    def _apply_webcam_settings(self, cap: cv2.VideoCapture) -> None:
        """MJPG + разрешение + FPS. Порядок важен: FourCC → FPS → размер."""
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if os.name == "nt":
                # YUY2 на 720p часто ограничен 5–10 FPS. MJPG снимает это
                # ограничение и даёт 30 FPS при том же разрешении.
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            log.debug("Не удалось включить MJPG-режим", exc_info=True)
        # Частоту просим ДО разрешения: некоторые камеры подбирают режим
        # по паре (разрешение, fps).
        if self.target_fps > 0:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
            except Exception:
                pass
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass

    @staticmethod
    def _probe_brightness(cap: cv2.VideoCapture, samples: int = 12) -> float:
        """Максимальная средняя яркость за несколько кадров.

        Некоторые камеры на MSMF открываются «успешно», но отдают полностью
        чёрные кадры (а потом вообще ошибку). По яркости это видно сразу —
        и мы успеваем переключиться на DirectShow.
        """
        best = 0.0
        got = 0
        for _ in range(samples):
            try:
                ok, frame = cap.read()
            except cv2.error:
                break
            if not ok or frame is None:
                continue
            got += 1
            try:
                best = max(best, float(frame.mean()))
            except Exception:
                pass
        if got == 0:
            return 0.0
        return best

    def _open_webcam(self, index: int):
        """Открыть веб-камеру лучшим из доступных бэкендов.

        ``camera.backend``: auto (DSHOW -> MSMF), msmf, dshow, any.
        На Linux/macOS всегда используется бэкенд по умолчанию.
        """
        if os.name != "nt":
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                self._apply_webcam_settings(cap)
            return cap
        if self.backend == "dshow":
            candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif self.backend == "msmf":
            candidates = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
        elif self.backend == "any":
            candidates = [cv2.CAP_ANY]
        else:  # auto: DirectShow первым — он стабильнее отдаёт живую картинку
            candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for position, backend in enumerate(candidates):
            try:
                cap = (cv2.VideoCapture(index)
                       if backend == cv2.CAP_ANY
                       else cv2.VideoCapture(index, backend))
            except cv2.error:
                continue
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue
            self._apply_webcam_settings(cap)
            brightness = self._probe_brightness(cap)
            # Чёрный кадр при живом isOpened() — почти всегда «сломанный»
            # MSMF. Если есть куда переключиться — переключаемся.
            if brightness < 2.0 and position + 1 < len(candidates):
                log.info("Веб-камера %s: бэкенд даёт чёрный кадр, пробую другой",
                         index)
                cap.release()
                continue
            name = {cv2.CAP_MSMF: "MSMF", cv2.CAP_DSHOW: "DirectShow",
                    cv2.CAP_ANY: "по умолчанию"}.get(backend, str(backend))
            log.info("Веб-камера %s: бэкенд %s (яркость %.1f)",
                     index, name, brightness)
            return cap
        return cv2.VideoCapture(index)

    def open(self) -> None:
        self._release()
        src = self.source
        log.info(f"Открываю источник: {src}")
        if isinstance(src, int):
            # На Windows DirectShow и Media Foundation ведут себя по-разному:
            # у одной камеры быстрее MSMF, у другой — DSHOW. DirectShow берём
            # первым: он стабильнее отдаёт непустые кадры.
            self._cap = self._open_webcam(src)
        elif self._is_file(src):
            self._cap = cv2.VideoCapture(str(src))
        else:
            # RTSP: используем FFmpeg-бэкенд, задаём буферы
            self._cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not (self._cap and self._cap.isOpened()):
            raise RuntimeError(f"Не удалось открыть источник {src}")
        if not self._is_file(self.source) and not isinstance(src, int):
            # RTSP: частоту и размер просим у самого потока.
            if self.target_fps > 0:
                self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            actual = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            actual = 0.0
        # Если камера не умеет сообщать fps — показываем запрошенную частоту.
        self.fps = actual if actual > 0 else float(self.target_fps or 0)
        self._next_at = 0.0

    def set_fps(self, fps: int) -> None:
        """Сменить желаемый FPS на лету (камера может применить после reopen)."""
        self.target_fps = max(0, int(fps or 0))
        if self._cap is not None and not self._is_file(self.source):
            try:
                if self.target_fps > 0:
                    self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
            except Exception:
                log.debug("Не удалось сменить FPS камеры", exc_info=True)

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
            # Ограничение частоты: если камера отдаёт 30 fps, а просили 10 —
            # спим между кадрами, чтобы не жечь CPU на лишних кадрах.
            if self.target_fps > 0 and not self._is_file(self.source):
                now = time.perf_counter()
                if self._next_at <= 0 or self._next_at < now - 1.0:
                    self._next_at = now
                self._next_at += 1.0 / self.target_fps
                delay = self._next_at - now
                if delay > 0:
                    time.sleep(min(delay, 1.0))
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