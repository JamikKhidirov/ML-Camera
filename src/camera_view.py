"""Живое превью камеры для Tkinter: кадры и скелет лица в разных потоках.

Зачем отдельный модуль:
  - окно Tkinter должно рисовать кадры с высоким FPS (30–60) и не ждать,
    пока нейросеть найдёт лицо;
  - поэтому захват кадров и анализ лиц работают в своих потоках, а окно
    просто забирает последний готовый кадр;
  - анализ лица идёт на уменьшенном кадре (по умолчанию 640 px), а скелет
    пересчитывается обратно в координаты полного кадра.

Модуль не импортирует tkinter — его можно тестировать и использовать
где угодно. Окно живёт в src/camera_window.py.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera import Camera
from src.face_mesh import draw_face_box, draw_face_mesh, mesh_edges
from src.face_recognizer import FaceRecognizer
from src.night_vision import ensure_night_quality

log = logging.getLogger(__name__)

KNOWN_COLOR = (80, 255, 170)     # BGR: зелёный
UNKNOWN_COLOR = (80, 150, 255)   # BGR: оранжевый


def _int_setting(section: dict, key: str, default: int,
                 low: int, high: int) -> int:
    try:
        value = int(section.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _float_setting(section: dict, key: str, default: float,
                   low: float, high: float) -> float:
    try:
        value = float(section.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


@dataclass
class FaceOverlay:
    """Одно найденное лицо в координатах полного кадра."""

    bbox: tuple[int, int, int, int]
    label: str
    color: tuple[int, int, int]
    landmarks: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    edges: list[tuple[int, int]] = field(default_factory=list)


class CameraPreview:
    """Камера + фоновый анализ лиц + готовый кадр для окна.

    Публичный интерфейс:
        preview.start()               — открыть камеру и потоки;
        preview.frame_for_display()   — BGR-кадр с рамками и скелетом;
        preview.stats()               — FPS, число лиц, статус;
        preview.snapshot()            — сохранить текущий кадр в data/snapshots;
        preview.stop()                — остановить всё.
    """

    def __init__(self, cfg, camera_cfg: dict,
                 recognizer_factory=FaceRecognizer):
        self.cfg = cfg
        self.camera_cfg = dict(camera_cfg or {})
        self.name = str(self.camera_cfg.get("name", "Камера"))
        self.source = self.camera_cfg.get("source", 0)
        self._recognizer_factory = recognizer_factory

        preview_cfg = cfg.get("preview", {}) or {}
        self.display_fps = _int_setting(preview_cfg, "fps", 30, 1, 60)
        self.display_width = _int_setting(preview_cfg, "width", 960, 320, 3840)
        self.analysis_fps = _float_setting(preview_cfg, "analysis_fps", 12.0,
                                           1.0, 30.0)
        self.analysis_width = _int_setting(preview_cfg, "analysis_width", 640,
                                           160, 1920)
        self.face_mesh_enabled = bool(preview_cfg.get("face_mesh", True))
        self.boxes_enabled = bool(preview_cfg.get("boxes", True))
        self.mirror = bool(preview_cfg.get("mirror", True))

        camera_cfg_all = cfg.get("camera", {}) or {}
        self.width = int(camera_cfg_all.get("width", 1280) or 1280)
        self.height = int(camera_cfg_all.get("height", 720) or 720)
        self.target_fps = int(camera_cfg_all.get("fps", 0) or 0)
        self.night_enabled = bool(camera_cfg_all.get("night_enabled", True))
        self.night_threshold = int(camera_cfg_all.get("night_brightness_threshold", 60))

        self._cam: Optional[Camera] = None
        self._stop = threading.Event()
        self._read_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts = 0.0
        self._faces: list[FaceOverlay] = []
        self._faces_ts = 0.0
        self._frame_count = 0
        self._capture_fps = 0.0
        self._analysis_fps = 0.0
        # Кэш «зеркало + ночное усиление» для показа. Камера отдаёт 10–30
        # кадров/с, а окно тикает чаще: без кэша CLAHE считался бы на каждом
        # тике и съедал больше половины CPU.
        self._display_base: Optional[np.ndarray] = None
        self._display_base_ts = -1.0
        self._display_base_mirror: Optional[bool] = None
        self._status = "Не запущено"
        self._error = ""
        self._started_at = 0.0
        self._last_error = ""

    # ------------------------------------------------------------------ запуск
    def start(self) -> None:
        if self._read_thread is not None and self._read_thread.is_alive():
            return
        self._stop.clear()
        self._error = ""
        self._status = "Открываю камеру…"
        self._display_base = None
        self._display_base_ts = -1.0
        self._display_base_mirror = None
        try:
            self._cam = Camera(
                self.source,
                width=self.width,
                height=self.height,
                fps=self.target_fps,
                backend=str((self.cfg.get("camera", {}) or {}).get("backend", "auto")),
            )
        except Exception as exc:  # камера занята/недоступна
            self._error = str(exc) or exc.__class__.__name__
            self._status = f"Ошибка камеры: {self._error}"
            raise
        self._started_at = time.time()
        self._read_thread = threading.Thread(
            target=self._read_loop, name=f"mlcamera-view-{self.name}", daemon=True)
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, name=f"mlcamera-faces-{self.name}",
            daemon=True)
        self._read_thread.start()
        self._analysis_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._read_thread, self._analysis_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._read_thread = None
        self._analysis_thread = None
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                log.debug("Не удалось закрыть камеру превью", exc_info=True)
        with self._lock:
            self._status = "Остановлено"
        self._display_base = None
        self._display_base_ts = -1.0
        self._display_base_mirror = None

    @property
    def running(self) -> bool:
        return self._read_thread is not None and self._read_thread.is_alive()

    # ------------------------------------------------------------------ потоки
    def _set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def _read_loop(self) -> None:
        """Читаем кадры максимально быстро; анализ лица не блокирует поток."""
        fps_window_start = time.perf_counter()
        fps_window_frames = 0
        empty_streak = 0
        while not self._stop.is_set():
            cam = self._cam
            if cam is None:
                break
            frame = cam.read()
            if frame is None:
                empty_streak += 1
                if empty_streak == 1:
                    if cam.is_file:
                        self._set_status("Видеофайл закончился")
                    else:
                        self._set_status("Жду кадры от камеры…")
                if cam.is_file and empty_streak >= 5:
                    # видеофайл удобно смотреть по кругу
                    try:
                        cam.open()
                        empty_streak = 0
                        self._set_status("Видеофайл: повтор")
                        continue
                    except Exception:
                        self._set_status("Видеофайл закончился")
                        self._stop.wait(0.5)
                        continue
                self._stop.wait(0.03)
                continue
            empty_streak = 0
            now = time.perf_counter()
            with self._lock:
                self._latest = frame
                self._latest_ts = now
                self._frame_count += 1
                if self._status.startswith(("Открываю", "Жду", "Видеофайл")):
                    self._status = "Камера работает"
            fps_window_frames += 1
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                with self._lock:
                    self._capture_fps = fps_window_frames / elapsed
                fps_window_start = now
                fps_window_frames = 0

    def _analysis_loop(self) -> None:
        """Детекция лиц и скелет — реже, чем показ кадров."""
        self._set_status("Загружаю модель лиц…")
        try:
            recognizer = self._recognizer_factory(self.cfg)
        except Exception as exc:
            self._error = str(exc) or exc.__class__.__name__
            self._set_status(f"Лица недоступны: {self._error}")
            log.exception("Не удалось подготовить распознавание лиц")
            return

        self._set_status("Ищу лица…")
        interval = 1.0 / max(1.0, self.analysis_fps)
        fps_window_start = time.perf_counter()
        fps_window_frames = 0
        while not self._stop.is_set():
            started = time.perf_counter()
            frame = self.latest_frame()
            if frame is None:
                self._stop.wait(0.05)
                continue
            try:
                overlays = self._recognize_faces(recognizer, frame)
            except Exception as exc:
                # модель/кадр могут сломаться — показываем кадр без скелета
                self._last_error = str(exc) or exc.__class__.__name__
                log.warning("Анализ лиц: %s", self._last_error)
                self._stop.wait(0.5)
                continue
            now = time.time()
            with self._lock:
                self._faces = overlays
                self._faces_ts = now
                if self._last_error:
                    self._status = "Камера работает (лица: ошибка)"
                elif overlays:
                    self._status = "Камера работает"
                else:
                    self._status = "Камера работает — лиц не видно"
            fps_window_frames += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= 1.0:
                with self._lock:
                    self._analysis_fps = fps_window_frames / elapsed
                fps_window_start = time.perf_counter()
                fps_window_frames = 0

            # не жжём CPU: спим остаток до следующего анализа
            spent = time.perf_counter() - started
            self._stop.wait(max(0.0, interval - spent))

    def _recognize_faces(self, recognizer, frame: np.ndarray) -> list[FaceOverlay]:
        height, width = frame.shape[:2]
        scale = min(1.0, float(self.analysis_width) / float(max(1, width)))
        if scale < 0.999:
            work = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = frame
        if self.night_enabled:
            try:
                _, work = ensure_night_quality(work, self.night_threshold)
            except Exception:
                log.debug("Ночное усиление кадра не удалось", exc_info=True)

        matches, _faces = recognizer.recognize(work)
        inv = 1.0 / scale if scale > 0 else 1.0

        def to_full(value):
            if value is None:
                return None
            points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
            return points * inv

        overlays: list[FaceOverlay] = []
        for match in matches:
            x1, y1, x2, y2 = (int(v * inv) for v in match.bbox)
            if match.unknown:
                label = str(self.cfg.people.get("unknown_label", "Незнакомец"))
                color = UNKNOWN_COLOR
            else:
                label = f"{match.name} {match.similarity:.2f}"
                color = KNOWN_COLOR
            landmarks = to_full(match.landmarks)
            overlays.append(FaceOverlay(
                bbox=(x1, y1, x2, y2),
                label=label,
                color=color,
                landmarks=landmarks,
                kps=to_full(match.kps),
                edges=mesh_edges(landmarks) if landmarks is not None else [],
            ))
        return overlays

    # ------------------------------------------------------------------ кадры
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def latest_faces(self, max_age: float = 1.5) -> list[FaceOverlay]:
        with self._lock:
            if time.time() - self._faces_ts > max_age:
                return []
            return list(self._faces)

    @staticmethod
    def _mirror_overlay(item: FaceOverlay, width: int) -> FaceOverlay:
        x1, y1, x2, y2 = item.bbox
        landmarks = None
        if item.landmarks is not None:
            landmarks = item.landmarks.copy()
            landmarks[:, 0] = width - 1 - landmarks[:, 0]
        kps = None
        if item.kps is not None:
            kps = item.kps.copy()
            kps[:, 0] = width - 1 - kps[:, 0]
        return FaceOverlay(
            bbox=(width - 1 - x2, y1, width - 1 - x1, y2),
            label=item.label,
            color=item.color,
            landmarks=landmarks,
            kps=kps,
            edges=list(item.edges),
        )

    def _base_for_display(self, mirror: bool) -> Optional[np.ndarray]:
        """Кадр после зеркала и ночного усиления — один раз на кадр камеры."""
        with self._lock:
            frame = self._latest
            ts = self._latest_ts
        if frame is None:
            return None
        if (self._display_base is not None
                and ts == self._display_base_ts
                and mirror == self._display_base_mirror):
            return self._display_base

        base = frame.copy()
        if mirror:
            base = cv2.flip(base, 1)
        if self.night_enabled:
            try:
                _, base = ensure_night_quality(base, self.night_threshold)
            except Exception:
                log.debug("Ночное усиление кадра превью не удалось", exc_info=True)
        self._display_base = base
        self._display_base_ts = ts
        self._display_base_mirror = mirror
        return base

    def frame_for_display(self, mirror: Optional[bool] = None) -> Optional[np.ndarray]:
        """Кадр с рамками и скелетом, уже уменьшенный под окно."""
        do_mirror = self.mirror if mirror is None else bool(mirror)
        base = self._base_for_display(do_mirror)
        if base is None:
            return None
        # Копия нужна, потому что рамки/скелет рисуются поверх кадра.
        frame = base.copy()
        height, width = frame.shape[:2]

        faces = self.latest_faces()
        if faces:
            drawn: list[FaceOverlay] = []
            for face in faces:
                if max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) < 12:
                    continue
                drawn.append(self._mirror_overlay(face, width) if do_mirror else face)
            self._draw_faces(frame, drawn)

        if width > self.display_width:
            scale = self.display_width / float(width)
            frame = cv2.resize(
                frame,
                (self.display_width, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _draw_faces(self, frame: np.ndarray,
                    faces: list[FaceOverlay]) -> None:
        height, width = frame.shape[:2]
        thickness = 1 if width < 800 else 2
        for face in faces:
            item = {
                "bbox": face.bbox,
                "label": face.label,
                "color": face.color,
                "landmarks": face.landmarks,
                "kps": face.kps,
                "edges": face.edges or (mesh_edges(face.landmarks)
                                        if face.landmarks is not None else []),
            }
            if self.boxes_enabled:
                draw_face_box(frame, item, thickness=thickness)
            if self.face_mesh_enabled:
                draw_face_mesh(frame, item, face.color, thickness=thickness)

    def snapshot(self, path: str | Path | None = None, quality: int = 95) -> Optional[str]:
        """Сохранить текущий кадр (как в окне) и вернуть путь."""
        frame = self.frame_for_display()
        if frame is None:
            return None
        if path is None:
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                                for ch in self.name) or "camera"
            out_dir = self.cfg.resolve("data/snapshots")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{safe_name}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
            return str(path)
        return None

    # ---------------------------------------------------------------- свойства
    def stats(self) -> dict:
        with self._lock:
            frame_size = (0, 0)
            if self._latest is not None:
                frame_size = (int(self._latest.shape[1]), int(self._latest.shape[0]))
            return {
                "name": self.name,
                "source": self.source,
                "running": self.running,
                "status": self._status,
                "error": self._error,
                "capture_fps": round(self._capture_fps, 1),
                "analysis_fps": round(self._analysis_fps, 1),
                "target_fps": self.target_fps,
                "faces": len(self._faces) if time.time() - self._faces_ts < 1.5 else 0,
                "frames": self._frame_count,
                "width": frame_size[0],
                "height": frame_size[1],
                "uptime": round(max(0.0, time.time() - self._started_at), 0),
            }

    def set_mesh_enabled(self, enabled: bool) -> None:
        self.face_mesh_enabled = bool(enabled)

    def set_boxes_enabled(self, enabled: bool) -> None:
        self.boxes_enabled = bool(enabled)

    def set_mirror(self, enabled: bool) -> None:
        self.mirror = bool(enabled)


__all__ = ["CameraPreview", "FaceOverlay", "KNOWN_COLOR", "UNKNOWN_COLOR"]
