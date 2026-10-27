"""Автоскачивание/проверка моделей.

Запуск:
  python scripts/download_models.py                                # config/config.yaml
  python scripts/download_models.py --config config/config.pc.yaml # ПК-версия

Скачивает ровно те модели, которые указаны в выбранном конфиге:
  - YOLO (детекция людей и машин) — ultralytics качает сама при первом запуске;
  - InsightFace (лица) — модель из face.model_name (buffalo_s / buffalo_l);
  - марка автомобиля — готовая ConvNeXt-Tiny (49 марок, Stanford Cars),
    если в конфиге включено car.brand_enabled;
  - OCR номеров — tesseract / easyocr / fastalpr (проверка, что установлено).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить/скачать модели из конфига")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (напр. config/config.pc.yaml)")
    parser.add_argument("--skip-brand", action="store_true",
                        help="не скачивать модель марки (111 МБ)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_logging("INFO")
    models_dir = cfg.models.resolve("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    yolo_path = Path(str(cfg.models.yolov8))
    print(f"=== 1/4 Проверяю YOLO ({yolo_path.name}: люди + машины) ===")
    from ultralytics import YOLO
    m = YOLO(str(cfg.models.resolve(cfg.models.yolov8)))  # сама скачает weights, если нет
    print(f"    YOLO готова: {len(m.names)} классов")

    face_model = str(cfg.face.model_name)
    print(f"=== 2/4 Проверяю InsightFace {face_model} (лица) ===")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=face_model, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=tuple(int(x) for x in cfg.face.detection_size))
    print("    InsightFace готова")

    brand_onnx = cfg.models.resolve(cfg.models.brand_onnx)
    brand_labels = cfg.models.resolve(cfg.models.brand_labels)
    brand_wanted = bool(cfg.car.get("brand_enabled", False)) or brand_onnx.exists()
    print("=== 3/4 Марка автомобиля (ConvNeXt-Tiny, Stanford Cars) ===")
    if args.skip_brand or not brand_wanted:
        print("    пропущено (car.brand_enabled: false)")
    elif brand_onnx.exists() and brand_labels.exists():
        print(f"    уже готова: {brand_onnx.name}")
    else:
        cmd = [sys.executable, str(ROOT / "scripts/download_car_brand_model.py")]
        if args.config:
            cmd += ["--config", str(args.config)]
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print("    ВНИМАНИЕ: не удалось скачать модель марки, марка будет 'unknown'")

    engine = str(cfg.car.get("plate_engine", "tesseract")).lower()
    print(f"=== 4/4 Проверяю OCR номеров ({engine}) ===")
    if engine == "easyocr":
        try:
            import easyocr  # noqa: F401
            print("    easyocr установлен")
        except Exception as exc:
            print("    ВНИМАНИЕ: easyocr не установлен:", exc)
            print("    Поставь: pip install -r requirements-pc.txt")
    elif engine in ("fastalpr", "fast-alpr", "alpr"):
        try:
            from fast_alpr import ALPR  # noqa: F401
            print("    fast-alpr установлен (детектор YOLOv9 + европейский OCR)")
        except Exception as exc:
            print("    ВНИМАНИЕ: fast-alpr не установлен:", exc)
            print("    Поставь: python scripts/install_alpr.py")
    else:
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
