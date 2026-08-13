"""Автоскачивание/проверка моделей.

Запуск: python scripts/download_models.py

Скачивает:
  - YOLOv8n (детекция людей и машин) — ultralytics качает сама при первом запуске,
    здесь просто проверяем и принудительно грузим.
  - InsightFace buffalo_s (лица) — тоже грузится сама.
Марка автомобиля не скачивается — её надо ОБУЧИТЬ (scripts/train_brand_classifier.py)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402

init_logging("INFO")


def main() -> int:
    cfg = load_config()
    models_dir = cfg.models.resolve("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1/3 Проверяю YOLOv8n (люди + машины) ===")
    from ultralytics import YOLO
    yolo_path = cfg.models.resolve(cfg.models.yolov8)
    m = YOLO(str(yolo_path))  # сама скачает weights, если нет
    print(f"    YOLO готова: {len(m.names)} классов")

    print("=== 2/3 Проверяю InsightFace buffalo_s (лица) ===")
    import platform
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=str(cfg.face.model_name), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=tuple(int(x) for x in cfg.face.detection_size))
    print("    InsightFace готова")

    print("=== 3/3 Проверяю tesseract (номера) ===")
    try:
        import pytesseract
        print(f"    pytesseract OK, версия: {pytesseract.get_tesseract_version()}")
    except Exception as exc:
        print("    ВНИМАНИЕ: pytesseract/tesseract не найдены:", exc)
        print("    Установи tesseract-ocr и tesseract-ocr-rus в систему "
              "(см. docs/SETUP.md) или поставь plate_engine: easyocr")

    print("\nВсе модели готовы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())