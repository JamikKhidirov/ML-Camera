"""Настройки MLCamera в окне Tkinter (без браузера).

Запуск:
    python scripts/settings.py
    scripts\\settings.bat

Окно читает и пишет ТОТ ЖЕ config/config.yaml, что и камера. Все поля —
из общей схемы src/settings.py, поэтому здесь есть ровно те же параметры,
что и во вкладке «🛠 Настройки» веб-панели.

Кнопка «Сохранить и применить» дополнительно отправляет значения работающей
панели (если она запущена) — камеры перезапустятся сами.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.settings import (FIELDS, SECTIONS, SettingsError,  # noqa: E402
                          apply_to_running_panel, get_path, save_settings)

HELP_COLOR = "#5c6b82"


# ---------------------------------------------------------------------------
#  Прокручиваемая вкладка
# ---------------------------------------------------------------------------
class ScrollFrame(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(12, 10))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        # add="+" — иначе каждая вкладка перезапишет колесо предыдущей.
        self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_wheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_wheel, add="+")

    def _on_inner(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width - 4)

    def _on_wheel(self, event) -> None:
        # Крутим только если курсор над этим canvas (иначе мешаем другим вкладкам).
        widget = self.winfo_containing(event.x_root, event.y_root)
        parent = widget
        while parent is not None:
            if parent is self.canvas:
                if getattr(event, "delta", 0):
                    step = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
                else:
                    step = -1 if getattr(event, "num", 0) == 4 else 1
                self.canvas.yview_scroll(step, "units")
                return
            parent = getattr(parent, "master", None)


# ---------------------------------------------------------------------------
#  Приложение
# ---------------------------------------------------------------------------
class SettingsApp:
    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path or None)
        self.root = tk.Tk()
        self.root.title("MLCamera — настройки")
        self.root.geometry("940x760")
        self.root.minsize(720, 560)

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Head.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Path.TLabel", foreground=HELP_COLOR, font=("Consolas", 9))
        style.configure("Hint.TLabel", foreground=HELP_COLOR, font=("Segoe UI", 8))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        # path -> (field, widget, kind, variable)
        self.widgets: list[tuple[dict, tk.Widget, str, tk.Variable]] = []
        self.cam_rows: list[dict] = []

        self._build_header()
        self._build_tabs()
        self._build_footer()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_header(self) -> None:
        head = ttk.Frame(self.root, padding=(14, 12, 14, 4))
        head.pack(fill="x")
        ttk.Label(head, text="🎥 MLCamera — настройки", style="Head.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text=("Один конфиг для камеры, веб-панели, ИИ, записи и уведомлений.\n"
                  "После сохранения камеры перезапустятся и применят новые FPS/разрешение."),
            justify="left",
        ).pack(anchor="w", pady=(2, 4))
        ttk.Label(head, text=f"Файл: {self.cfg.config_path}", style="Path.TLabel").pack(anchor="w")

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(6, 4))

        # Первая вкладка — камеры (список источников).
        self.cams_tab = ScrollFrame(self.nb)
        self.nb.add(self.cams_tab, text="🎥 Камеры")
        self._fill_cameras_tab()

        for section in SECTIONS:
            fields = [f for f in FIELDS if f["section"] == section["id"]]
            if not fields:
                continue
            tab = ScrollFrame(self.nb)
            title = f"{section.get('icon', '•')} {section['title']}"
            self.nb.add(tab, text=title)
            if section.get("hint"):
                ttk.Label(tab.inner, text=section["hint"], style="Hint.TLabel",
                          wraplength=820, justify="left").pack(anchor="w", pady=(0, 8))
            for field in fields:
                self._make_field(tab.inner, field)

        self.cams_tab.inner.columnconfigure(1, weight=1)

    def _build_footer(self) -> None:
        foot = ttk.Frame(self.root, padding=(12, 4, 12, 10))
        foot.pack(fill="x")
        self.status = tk.StringVar(value="Готово")
        ttk.Label(foot, textvariable=self.status, style="Hint.TLabel").pack(side="left")

        ttk.Button(foot, text="🚀 Сохранить и применить", style="Accent.TButton",
                   command=self.save_and_apply).pack(side="right", padx=(6, 0))
        ttk.Button(foot, text="💾 Сохранить", command=self.save_only).pack(side="right")
        ttk.Button(foot, text="⟳ Перечитать", command=self.reload).pack(side="right", padx=(0, 6))

    # -------------------------------------------------------------- камеры
    def _fill_cameras_tab(self) -> None:
        frame = self.cams_tab.inner
        ttk.Label(
            frame,
            text=("Источники видео. 0 — веб-камера ноутбука. Для IP-камеры: "
                  "rtsp://логин:пароль@192.168.1.100:554/stream1\n"
                  "Первый запуск новой камеры может занять 10–30 секунд."),
            justify="left", wraplength=840,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.cam_box = ttk.Frame(frame)
        self.cam_box.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.cam_box.columnconfigure(1, weight=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="+ Добавить камеру", command=self._add_cam_row).pack(side="left")

    def _add_cam_row(self, cam: dict | None = None) -> None:
        cam = cam or {"name": f"Камера {len(self.cam_rows) + 1}",
                      "source": 0, "enabled": True}
        row = ttk.Frame(self.cam_box)
        row.pack(fill="x", pady=2)

        name = tk.StringVar(value=str(cam.get("name", "")))
        source = tk.StringVar(value=str(cam.get("source", 0)))
        enabled = tk.BooleanVar(value=bool(cam.get("enabled", True)))

        ttk.Label(row, text="Имя").pack(side="left")
        ttk.Entry(row, textvariable=name, width=26).pack(side="left", padx=(4, 10))
        ttk.Label(row, text="Источник").pack(side="left")
        ttk.Entry(row, textvariable=source, width=42).pack(
            side="left", padx=(4, 10), fill="x", expand=True)
        ttk.Checkbutton(row, text="включена", variable=enabled).pack(side="left")

        item = {"frame": row, "name": name, "source": source, "enabled": enabled}

        def remove() -> None:
            item["frame"].destroy()
            self.cam_rows.remove(item)

        ttk.Button(row, text="✕", width=3, command=remove).pack(side="left", padx=(8, 0))
        self.cam_rows.append(item)

    # --------------------------------------------------------------- поля
    def _make_field(self, parent: ttk.Frame, field: dict) -> None:
        label = field["label"] + (f", {field['unit']}" if field.get("unit") else "")
        kind = field["type"]
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(3, 5))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text=label).grid(row=0, column=0, sticky="w", padx=(0, 12))

        var: tk.Variable
        widget: tk.Widget
        if kind == "bool":
            var = tk.BooleanVar(value=bool(field.get("default", False)))
            widget = ttk.Checkbutton(box, variable=var)
        elif kind == "select":
            var = tk.StringVar(value=str(field.get("default", "")))
            widget = ttk.Combobox(box, textvariable=var, state="readonly",
                                  values=[str(o) for o in field.get("options", [])],
                                  width=22)
        else:
            var = tk.StringVar(value=str(field.get("default", "")))
            if kind in ("int", "float"):
                widget = ttk.Spinbox(
                    box, textvariable=var, width=16,
                    from_=field.get("min", 0), to=field.get("max", 10 ** 9),
                    increment=field.get("step", 1),
                )
            elif kind == "password":
                widget = ttk.Entry(box, textvariable=var, show="•", width=34)
            else:
                widget = ttk.Entry(box, textvariable=var, width=34)

        widget.grid(row=0, column=1, sticky="w")
        if field.get("help"):
            ttk.Label(box, text=field["help"], style="Hint.TLabel",
                      wraplength=760, justify="left").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(1, 0))
        self.widgets.append((field, widget, kind, var))

    # ----------------------------------------------------------- значения
    def reload(self) -> None:
        """Перечитать конфиг с диска и заполнить виджеты."""
        try:
            self.cfg = load_config(str(self.cfg.config_path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("MLCamera", f"Не удалось прочитать конфиг:\n{exc}")
            return

        raw = self.cfg._data
        for field, _widget, kind, var in self.widgets:
            value = get_path(raw, field["path"], field.get("default"))
            try:
                if kind == "bool":
                    var.set(bool(value))
                elif kind == "list":
                    var.set(", ".join(str(v) for v in (value or [])))
                else:
                    var.set("" if value is None else str(value))
            except tk.TclError:
                pass

        for item in self.cam_rows:
            item["frame"].destroy()
        self.cam_rows.clear()
        cams = raw.get("cameras") or []
        if isinstance(cams, dict):
            cams = cams.get("cameras") or []
        for cam in cams:
            if isinstance(cam, dict):
                self._add_cam_row(cam)
        self.status.set(f"Загружено: {self.cfg.config_path}")

    def collect(self) -> tuple[dict, list[dict]]:
        values: dict = {}
        for field, _widget, _kind, var in self.widgets:
            values[field["path"]] = var.get()

        cameras = []
        for item in self.cam_rows:
            cameras.append({
                "name": item["name"].get().strip(),
                "source": item["source"].get().strip(),
                "enabled": bool(item["enabled"].get()),
            })
        return values, cameras

    # ----------------------------------------------------------- действия
    def _save(self, apply_running: bool) -> None:
        values, cameras = self.collect()
        try:
            save_settings(self.cfg, values, cameras)
        except SettingsError as exc:
            messagebox.showerror("Проверь поля", "\n".join(exc.errors))
            return
        except OSError as exc:
            messagebox.showerror("MLCamera", f"Не удалось записать конфиг:\n{exc}")
            return

        if not apply_running:
            self.status.set(f"Сохранено: {self.cfg.config_path}")
            messagebox.showinfo("MLCamera", "Настройки сохранены.\n"
                                           "Они применятся при следующем запуске камеры.")
            return

        ok, message = apply_to_running_panel(self.cfg, values, cameras)
        if ok:
            self.status.set("Сохранено и отправлено работающей панели")
            messagebox.showinfo("MLCamera", f"Сохранено.\n{message}")
        else:
            self.status.set("Сохранено (панель не запущена)")
            messagebox.showwarning(
                "MLCamera",
                "Настройки сохранены в файл.\n\n"
                "Работающая панель не ответила — вероятно, камера сейчас не запущена.\n"
                f"Новые значения применятся при запуске main.py.\n\n({message})")


    def save_only(self) -> None:
        self._save(apply_running=False)

    def save_and_apply(self) -> None:
        self._save(apply_running=True)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="MLCamera — настройки (Tkinter)")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (по умолчанию config/config.yaml)")
    args = parser.parse_args()
    app = SettingsApp(args.config)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
