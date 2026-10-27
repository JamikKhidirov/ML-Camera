"""Скачивание готовой модели МАРКИ автомобиля (без обучения).

Источник: ryosiswand/convnext_tiny-stanford-cars-brand (Hugging Face, Apache-2.0)
  ConvNeXt-Tiny, дообучен на Stanford Cars, 49 марок,
  точность ~92.7% на тесте, вход 288x288.

Что делает скрипт:
  1) качает model.pth + training_info.json (~111 МБ);
  2) собирает ту же архитектуру и экспортирует её в ONNX (data/models/car_brand.onnx);
  3) пишет data/models/car_brand_labels.json и car_brand_meta.json
     (размер входа + нормализация ImageNet).

Запуск:
  python scripts/download_car_brand_model.py
  python scripts/download_car_brand_model.py --force   # перекачать/пересобрать

После этого марки включаются в конфиге:  car.brand_enabled: true
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

HF_REPO = "ryosiswand/convnext_tiny-stanford-cars-brand"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
FILES = ("model.pth", "training_info.json")


def _download(name: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"    уже есть: {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
        return dest
    url = f"{HF_BASE}/{name}?download=true"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"    качаю {name} ...")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} МБ "
                      f"({done * 100 / total:5.1f}%)", end="", flush=True)
    print()
    shutil.move(str(tmp), str(dest))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать готовую модель марки авто")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--force", action="store_true", help="перекачать и пересобрать")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_onnx = cfg.models.resolve(cfg.models.brand_onnx)
    out_labels = cfg.models.resolve(cfg.models.brand_labels)
    out_meta = out_onnx.parent / "car_brand_meta.json"
    cache_dir = cfg.resolve("data/models/.cache/car_brand")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    if out_onnx.exists() and out_labels.exists() and not args.force:
        print(f"Модель уже готова: {out_onnx}")
        print("Пересобрать: python scripts/download_car_brand_model.py --force")
        return 0

    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        print("Нужны torch и torchvision:  pip install torch torchvision")
        return 1

    print("=== 1/4 Скачивание checkpoint ===")
    paths = {name: _download(name, cache_dir / name, args.force) for name in FILES}
    info = json.loads(paths["training_info.json"].read_text(encoding="utf-8"))
    classes = [str(c) for c in info["class_names"]]
    num_classes = int(info.get("num_classes", len(classes)))
    image_size = int(info.get("image_size", 288))
    print(f"    марок: {num_classes}, вход: {image_size}x{image_size}, "
          f"точность: {info.get('test_accuracy_tta', info.get('val_accuracy', 0)):.1%}")

    print("=== 2/4 Сборка ConvNeXt-Tiny и загрузка весов ===")
    ckpt = torch.load(paths["model.pth"], map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model_state_dict", "model")):
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt["model"]
    else:
        state = ckpt
    state = {str(k).removeprefix("module."): v for k, v in state.items()}

    arch = (ckpt.get("arch") if isinstance(ckpt, dict) else None) or "convnext_tiny"
    if arch != "convnext_tiny":
        print(f"Неожиданная архитектура: {arch}")
        return 1
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("classifier.2.")]
    if real_missing:
        print(f"В чекпойнте не хватает весов: {real_missing[:5]} ...")
        return 1
    if unexpected:
        print(f"    предупреждение: лишние ключи {unexpected[:3]}")
    model.eval()

    print("=== 3/4 Экспорт в ONNX ===")
    dummy = torch.zeros(1, 3, image_size, image_size)
    tmp_onnx = out_onnx.with_suffix(".onnx.tmp")
    try:
        torch.onnx.export(
            model, dummy, str(tmp_onnx),
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    except TypeError:
        # старые версии torch не знают про dynamo=
        torch.onnx.export(
            model, dummy, str(tmp_onnx),
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    shutil.move(str(tmp_onnx), str(out_onnx))

    print("=== 4/4 Проверка ONNX и запись подписей ===")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    if out.shape[-1] != num_classes:
        print(f"ОШИБКА: выход модели {out.shape}, ожидалось {num_classes} классов")
        return 1
    out_labels.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    out_meta.write_text(json.dumps({
        "input_size": image_size,
        "resize": "shortest_center_crop",
        "resize_ratio": 1.14,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "tta": True,
        "classes": classes,
        "source": HF_REPO,
        "license": "Apache-2.0",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = out_onnx.stat().st_size / 1e6
    print(f"\nГотово: {out_onnx} ({size_mb:.1f} МБ)")
    print(f"        {out_labels} ({len(classes)} марок)")
    print("Включи марки:  car.brand_enabled: true  (в config/*.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
