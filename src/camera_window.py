"""Окно камеры в Tkinter: живой поток + скелет лица + высокий FPS.

Запуск:
    python -m src.camera_window                 # камера из config/config.yaml
    python main.py --view                       # то же из главного меню
    python -m src.camera_window --camera 1      # вторая камера из списка
    python -m src.camera_window --source 0      # веб-камера напрямую
    camera.bat                                  # ярлык для Windows

Почему FPS высокий: кадры читает один поток, лица ищет второй (на
уменьшенном кадре), а окно только рисует готовый кадр. Нейросеть больше
не блокирует показ видео, поэтому 30 FPS держатся даже на слабом CPU.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.camera_view import CameraPreview
from src.config import load_config

log = logging.getLogger(__name__)

BG = "#0b1220"
PANEL = "#111c2e"
TEXT = "#e8eefc"
MUTED = "#8fa3c4"
ACCENT = "#40d6ff"


def _camera_list(cfg) -> list[dict]:
    """Все камеры из конфига; если список пуст — веб-камера ноутбука."""
    cameras = cfg.get("cameras", []) or []
    if isinstance(cameras, dict):
        cameras = cameras.get("cameras") or []
    result: list[dict] = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            continue
        name = str(camera.get("name") or f"Камера {index + 1}")
        result.append({
            "name": name,
            "source": camera.get("source", 0),
            "enabled": bool(camera.get("enabled", True)),
        })
    if not result:
        result.append({"name": "Ноутбук (веб-камера)", "source": 0,
                       "enabled": True})
    return result


def _short(text: str, limit: int = 38) -> str:
    """Короткое имя для заголовка: длинный RTSP-адрес не ломает окно."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class CameraWindow:
    """Tkinter-окно с камерой, скелетом лица и кнопками управления."""

    def __init__(self, cfg, camera_cfg: dict):
        self.cfg = cfg
        self.cameras = _camera_list(cfg)
        self.current = dict(camera_cfg)
        self.preview = CameraPreview(cfg, self.current)
        self.paused = False
        self._photo: ImageTk.PhotoImage | None = None

        # Корень Tkinter создаём ДО BooleanVar: переменные ищут default root.
        self.root = tk.Tk()
        self.root.title(f"MLCamera — {_short(self.current.get('name', 'камера'))}")
        self.root.geometry("1120x760")
        self.root.minsize(640, 480)
        self.root.configure(bg=BG)

        self._show_mesh = tk.BooleanVar(value=self.preview.face_mesh_enabled)
        self._show_boxes = tk.BooleanVar(value=self.preview.boxes_enabled)
        self._mirror = tk.BooleanVar(value=self.preview.mirror)
        self._display_frames = 0
        self._display_window_start = time.perf_counter()
        self._display_fps = 0.0
        self._tick_count = 0
        self._last_error = ""

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._bind_keys()

        try:
            self.preview.start()
        except Exception as exc:  # камера занята или не найдена
            self._last_error = str(exc) or exc.__class__.__name__
            self._set_message(f"Не удалось открыть камеру:\n{self._last_error}")
            log.exception("Не удалось открыть камеру в Tkinter")

        self.root.after(0, self._tick)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Cam.TLabel", background=BG, foreground=TEXT)
        style.configure("CamMuted.TLabel", background=BG, foreground=MUTED)
        style.configure("CamTitle.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 13, "bold"))

        top = tk.Frame(self.root, bg=PANEL)
        top.pack(fill="x", side="top")

        # Переключатели прижимаем к правому краю ПЕРВЫМИ: иначе длинное имя
        # камеры выдавливает их за пределы окна.
        switches = tk.Frame(top, bg=PANEL)
        switches.pack(side="right", padx=8)
        for text, variable, command in (
                ("Скелет", self._show_mesh, self._toggle_mesh),
                ("Рамки", self._show_boxes, self._toggle_boxes),
                ("Зеркало", self._mirror, self._toggle_mirror)):
            tk.Checkbutton(
                switches, text=text, variable=variable, command=command,
                bg=PANEL, fg=TEXT, activebackground=PANEL,
                activeforeground=TEXT, selectcolor="#1f3352",
                font=("Segoe UI", 9), bd=0, highlightthickness=0,
            ).pack(side="right", padx=3)

        tk.Label(top, text="📷", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Emoji", 16)).pack(side="left", padx=(10, 4),
                                                  pady=8)
        self.title_var = tk.StringVar(value=_short(self.current.get("name", "камера"), 24))
        tk.Label(top, textvariable=self.title_var, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        self.cam_box = ttk.Combobox(
            top, state="readonly", width=22,
            values=[f"{i}: {c['name']}" for i, c in enumerate(self.cameras)])
        self.cam_box.current(self._current_index())
        self.cam_box.pack(side="left", padx=(10, 4), pady=8)
        self.cam_box.bind("<<ComboboxSelected>>", self._on_camera_selected)

        ttk.Button(top, text="⟳", width=3,
                   command=self.restart).pack(side="left", padx=2)
        self.pause_btn = ttk.Button(top, text="⏸ Пауза", width=8,
                                    command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(6, 2))
        ttk.Button(top, text="📷 Снимок", command=self.take_snapshot).pack(
            side="left", padx=2)
        ttk.Button(top, text="⛶ Экран", command=self.toggle_fullscreen).pack(
            side="left", padx=2)
        ttk.Button(top, text="⚙", width=3, command=self.open_settings).pack(
            side="left", padx=2)

        self.video = tk.Label(self.root, bg="#000000", fg=MUTED, bd=0,
                              font=("Segoe UI", 12), compound="center")
        self.video.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", side="bottom", padx=10, pady=(0, 6))
        self.status_var = tk.StringVar(value="Запускаю камеру…")
        tk.Label(bottom, textvariable=self.status_var, bg=BG, fg=MUTED,
                 font=("Consolas", 9), anchor="w").pack(side="left")
        self.hint_var = tk.StringVar(
            value="Пробел — пауза · S — снимок · F11 — полный экран · Esc — выход")
        tk.Label(bottom, textvariable=self.hint_var, bg=BG, fg="#5c6b82",
                 font=("Segoe UI", 8), anchor="e").pack(side="right")

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.root.bind("<space>", lambda _e: self.toggle_pause())
        self.root.bind("<s>", lambda _e: self.take_snapshot())
        self.root.bind("<S>", lambda _e: self.take_snapshot())

    # ------------------------------------------------------------- управление
    def _current_index(self) -> int:
        for index, camera in enumerate(self.cameras):
            if (str(camera.get("name")) == str(self.current.get("name"))
                    and str(camera.get("source")) == str(self.current.get("source"))):
                return index
        return 0

    def _on_camera_selected(self, event=None) -> None:
        index = int(self.cam_box.current())
        if 0 <= index < len(self.cameras):
            self.switch_camera(self.cameras[index])

    def switch_camera(self, camera_cfg: dict) -> None:
        self.preview.stop()
        self.current = dict(camera_cfg)
        self.preview = CameraPreview(self.cfg, self.current)
        self._show_mesh.set(self.preview.face_mesh_enabled)
        self._show_boxes.set(self.preview.boxes_enabled)
        self._mirror.set(self.preview.mirror)
        self.title_var.set(_short(self.current.get("name", "камера"), 24))
        self.root.title(f"MLCamera — {_short(self.current.get('name', 'камера'))}")
        self._photo = None
        self.video.configure(image="", text="Открываю камеру…")
        try:
            self.preview.start()
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc) or exc.__class__.__name__
            self._set_message(f"Не удалось открыть камеру:\n{self._last_error}")

    def restart(self) -> None:
        self.switch_camera(self.current)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.configure(
            text="▶ Продолжить" if self.paused else "⏸ Пауза")
        if self.paused:
            self.status_var.set("Пауза — камера работает, показ остановлен")

    def _toggle_mesh(self) -> None:
        self.preview.set_mesh_enabled(bool(self._show_mesh.get()))

    def _toggle_boxes(self) -> None:
        self.preview.set_boxes_enabled(bool(self._show_boxes.get()))

    def _toggle_mirror(self) -> None:
        self.preview.set_mirror(bool(self._mirror.get()))

    def take_snapshot(self) -> None:
        path = self.preview.snapshot()
        if path:
            self.hint_var.set(f"Снимок сохранён: {path}")
        else:
            self.hint_var.set("Снимок не получился: камера ещё не дала кадр")

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen",
                             not bool(self.root.attributes("-fullscreen")))

    def open_settings(self) -> None:
        script = Path(__file__).resolve().parent.parent / "scripts" / "settings.py"
        if not script.exists():
            messagebox.showwarning("MLCamera", "Файл настроек не найден.")
            return
        command = [sys.executable, str(script), "--config",
                   str(self.cfg.config_path)]
        try:
            subprocess.Popen(command, cwd=str(script.parent.parent))
        except OSError as exc:
            messagebox.showerror("MLCamera", f"Не удалось открыть настройки:\n{exc}")

    # -------------------------------------------------------------- отрисовка
    def _set_message(self, text: str) -> None:
        self._photo = None
        self.video.configure(image="", text=text)

    def _render(self, frame) -> None:
        if frame is None or frame.size == 0:
            return
        height, width = frame.shape[:2]
        available_w = max(160, self.video.winfo_width() - 4)
        available_h = max(120, self.video.winfo_height() - 4)
        scale = min(available_w / float(width), available_h / float(height))
        # мелкое окно — уменьшаем, большой экран — можем увеличить до 2×
        scale = max(0.2, min(2.0, scale))
        target_w = max(1, int(width * scale))
        target_h = max(1, int(height * scale))
        if target_w != width or target_h != height:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=interpolation)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self._photo, text="")

    def _tick(self) -> None:
        """Главный цикл окна: только рисуем, ничего не считаем."""
        started = time.perf_counter()
        if not self.paused:
            frame = self.preview.frame_for_display()
            if frame is not None:
                self._render(frame)
                self._display_frames += 1
            elif self._last_error:
                self._set_message(f"Камера недоступна:\n{self._last_error}")

        now = time.perf_counter()
        elapsed = now - self._display_window_start
        if elapsed >= 1.0:
            self._display_fps = self._display_frames / elapsed
            self._display_frames = 0
            self._display_window_start = now

        self._tick_count += 1
        if self._tick_count % 5 == 0:
            self._update_status()
        delay_ms = max(1, int(1000.0 / max(1, self.preview.display_fps)
                              - (time.perf_counter() - started) * 1000.0))
        self.root.after(delay_ms, self._tick)

    def _update_status(self) -> None:
        stats = self.preview.stats()
        self.status_var.set(
            f"Камера: {stats['capture_fps']:>4.1f} FPS  ·  "
            f"Показ: {self._display_fps:>4.1f} FPS  ·  "
            f"Анализ: {stats['analysis_fps']:>4.1f} FPS  ·  "
            f"Лиц: {stats['faces']}  ·  "
            f"{stats['width']}×{stats['height']}  ·  {stats['status']}")

    # ------------------------------------------------------------------ жизнь
    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.preview.stop()

    def close(self) -> None:
        try:
            self.preview.stop()
        finally:
            self.root.destroy()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLCamera — камера в окне Tkinter со скелетом лица")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (по умолчанию config/config.yaml)")
    parser.add_argument("--camera", type=int, default=None,
                        help="индекс камеры в списке cameras (0, 1, …)")
    parser.add_argument("--source", default=None,
                        help="источник напрямую: 0 / 1 / rtsp://… / видеофайл")
    parser.add_argument("--fps", type=int, default=None,
                        help="FPS показа (переопределяет preview.fps)")
    parser.add_argument("--width", type=int, default=None,
                        help="ширина кадра в окне (переопределяет preview.width)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="не зеркалить кадр (по умолчанию зеркалится)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)

    preview = cfg._data.get("preview")
    if not isinstance(preview, dict):
        preview = {}
        cfg._data["preview"] = preview
    if args.fps:
        preview["fps"] = args.fps
    if args.width:
        preview["width"] = args.width
    if args.no_mirror:
        preview["mirror"] = False

    cameras = _camera_list(cfg)
    if args.source is not None:
        source = args.source
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        if isinstance(source, str) and Path(source).suffix.lower() in (
                ".mp4", ".avi", ".mkv", ".mov"):
            name = f"Видео: {Path(source).name}"
        else:
            name = f"Источник {args.source}"
        camera_cfg = {"name": name, "source": source, "enabled": True}
    else:
        index = 0 if args.camera is None else args.camera
        if not 0 <= index < len(cameras):
            print(f"Нет камеры с индексом {index}. Всего камер: {len(cameras)}")
            return 2
        camera_cfg = cameras[index]

    app = CameraWindow(cfg, camera_cfg)
    app.run()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
