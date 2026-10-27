"""Установка нейросетевого распознавания номеров (FastALPR, MIT).

Что ставится:
  fast-alpr         — обёртка «детектор + OCR»
  open-image-models — детектор номера YOLOv9 (ONNX)
  fast-plate-ocr    — OCR номеров (ONNX, европейская модель)

Почему не обычный `pip install fast-alpr`:
  он тянет opencv-python-headless, которая затирает GUI-сборку OpenCV и ломает
  окно предпросмотра (cv2.imshow). Поэтому ставим с --no-deps и отдельно
  проверяем/чиним OpenCV.

Запуск:
  python scripts/install_alpr.py            # установить и скачать модели
  python scripts/install_alpr.py --check    # только проверить
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import metadata


def _run(args: list[str]) -> int:
    print("  >", " ".join(args))
    return subprocess.call([sys.executable, "-m", "pip", *args])


def _has(dist: str) -> bool:
    try:
        metadata.distribution(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _cv2_has_gui() -> bool:
    try:
        import cv2
    except ImportError:
        return False
    info = cv2.getBuildInformation()
    line = next((ln for ln in info.splitlines() if ln.strip().startswith("GUI:")), "")
    return "NONE" not in line.upper()


def _fix_opencv() -> None:
    """Убирает headless-сборку, если она перекрыла GUI-сборку OpenCV."""
    gui = _cv2_has_gui()
    print(f"  OpenCV GUI: {'да' if gui else 'НЕТ'}")
    if gui:
        return
    if _has("opencv-python-headless"):
        print("  Удаляю opencv-python-headless (она ломает cv2.imshow)...")
        _run(["uninstall", "-y", "opencv-python-headless"])
    ver = _version("opencv-python")
    print("  Ставлю GUI-сборку opencv-python...")
    if ver:
        _run(["install", "--force-reinstall", "--no-deps", f"opencv-python=={ver}"])
    else:
        _run(["install", "--no-deps", "opencv-python"])
    print(f"  OpenCV GUI теперь: {'да' if _cv2_has_gui() else 'НЕТ (проверь вручную)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Установка FastALPR (номера)")
    parser.add_argument("--check", action="store_true", help="только проверить, не ставить")
    parser.add_argument("--no-models", action="store_true", help="не скачивать веса моделей")
    args = parser.parse_args()

    print("=== 1/3 Проверка зависимостей ===")
    _fix_opencv()

    need = [d for d in ("onnxruntime", "numpy", "pyyaml", "rich", "tqdm")
            if not _has(d)]
    if need:
        print(f"  Не хватает: {', '.join(need)}")
        if not args.check:
            _run(["install", *need])
    else:
        print("  Базовые зависимости на месте")

    installed = all(_has(d) for d in ("fast-alpr", "fast-plate-ocr", "open-image-models"))
    if not installed and not args.check:
        print("=== 2/3 Установка FastALPR (без зависимостей) ===")
        rc = _run(["install", "--no-deps", "--upgrade",
                   "fast-alpr>=0.4.0", "fast-plate-ocr>=1.1.0", "open-image-models>=0.5.1"])
        if rc != 0:
            print("Установка не удалась. Проверь интернет/версию Python (нужен 3.10+).")
            return rc
    elif args.check:
        print("=== 2/3 Проверка пакетов ===")
        for dist in ("fast-alpr", "fast-plate-ocr", "open-image-models"):
            print(f"  {dist}: {_version(dist) or 'НЕ УСТАНОВЛЕН'}")

    print("=== 3/3 Проверка импорта и моделей ===")
    try:
        from fast_alpr import ALPR  # noqa: F401
        print("  fast_alpr импортируется")
    except Exception as exc:
        print(f"  ОШИБКА импорта fast_alpr: {exc}")
        if args.check:
            return 1
        # повторная попытка после установки
        try:
            from fast_alpr import ALPR  # noqa: F811,F401
            print("  fast_alpr импортируется")
        except Exception as exc2:
            print(f"  ОШИБКА: {exc2}")
            return 1

    if args.no_models:
        print("\nГотово. Веса скачаются при первом запуске камеры.")
        return 0
    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="european-plates-mobile-vit-v2-model",
            detector_providers=["CPUExecutionProvider"],
            detector_sess_options=sess_options,
            ocr_providers=["CPUExecutionProvider"],
            ocr_sess_options=sess_options,
        )
        print("  Модели загружены, ALPR готов")
        del alpr
    except Exception as exc:
        print(f"  Не удалось загрузить модели (проверь интернет): {exc}")
        return 1

    print("\nГотово. Включи в конфиге:  car.plate_engine: \"fastalpr\"")
    print("Для ПК это заметно точнее tesseract, для слабого VPS оставь tesseract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
