"""ПК-версия MLCamera: запуск камеры с окном ИЛИ сборка .exe с иконкой.

Примеры:
  python pc/main.py            # запустить камеру → окно с кадрами + запись + уведомления
  python pc/main.py --build    # собрать pc/dist/MLCamera/MLCamera.exe (с иконкой)
  python pc/main.py --run      # то же, что и без флагов (явно)

Собранный MLCamera.exe сам открывает окно с кадрами камеры.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# В .exe весь проект в папке рядом с MLCamera.exe; в исходниках — папка репо
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

PC_DIR = ROOT / "pc"
ICON = PC_DIR / "MLCamera.ico"
PC_CONFIG = ROOT / "config/config.yaml"


def run(extra_args: list[str] | None = None) -> int:
    """Запуск камеры: окно с кадрами (превью) + запись видео + уведомления."""
    argv = [sys.argv[0], "--config", str(PC_CONFIG), "--preview"]
    if extra_args:
        argv += extra_args
    sys.argv = argv
    from src.main import main
    return main()


def _copy_app_data(dst: Path) -> None:
    """Класть config/ и data/ рядом с exe, чтобы можно было дополнять без пересборки."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "config", dst / "config", dirs_exist_ok=True)
    shutil.copytree(ROOT / "data", dst / "data", dirs_exist_ok=True)
    # объекты InsightFace (meanshape и т.п.) — exe ищет их в _internal/objects
    try:
        import insightface
        objects_src = Path(insightface.__file__).parent / "data" / "objects"
        if objects_src.exists():
            shutil.copytree(objects_src, dst / "_internal" / "objects", dirs_exist_ok=True)
    except Exception:
        pass


def build() -> int:
    """Сборка MLCamera.exe (PyInstaller, ~5-15 минут). Иконка уже внутри."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir",
        "--name", "MLCamera",
        "--icon", str(ICON),
        "--distpath", str(PC_DIR / "dist"),
        "--workpath", str(ROOT / "build" / "pyi"),
        "--collect-data", "insightface",
        "--collect-data", "ultralytics",
        str(PC_DIR / "main.py"),          # точка входа — этот же файл (запуск с окном)
    ]
    print("Собираю MLCamera.exe (5-15 минут)...")
    res = subprocess.call(cmd, cwd=str(ROOT))
    if res != 0:
        print("ОШИБКА сборки. Смотри вывод выше.")
        return res
    exe_dir = PC_DIR / "dist" / "MLCamera"
    _copy_app_data(exe_dir)
    print()
    print("ГОТОВО! Программа: ", exe_dir / "MLCamera.exe")
    print("Создай ярлык на рабочий стол и запускай двойным кликом.")
    print("Фото людей:    ", exe_dir / "data/datasets/raw/ИМЯ/")
    print("Настройки:     ", exe_dir / "config/config.yaml")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MLCamera — ПК-версия")
    ap.add_argument("--build", action="store_true",
                    help="собрать .exe с иконкой (PyInstaller) и выйти")
    ap.add_argument("--run", action="store_true",
                    help="запустить камеру (окно с кадрами) — это действие по умолчанию")
    args, rest = ap.parse_known_args()
    if args.build:
        return build()
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())