"""Обучение СВОЕЙ модели марки автомобиля (например, для рынка РФ).

Готовая модель из `scripts/download_car_brand_model.py` знает 49 марок
Stanford Cars (США, 2012). Российских марок (Lada, Kia, Skoda, Renault,
Haval, Chery...) в ней нет. Этот скрипт обучает маленький классификатор
на ВАШИХ фото и экспортирует его в тот же формат ONNX + meta, который
понимает `src/car_analyzer.py`.

Структура датасета (класс = папка с названием марки):

    data/datasets/cars/train/
        Lada/        *.jpg
        Kia/         *.jpg
        Hyundai/     *.jpg
        Toyota/      *.jpg
    data/datasets/cars/val/        # необязательно
        Lada/ ...

Если папки val нет — скрипт сам отложит `--val-split` изображений.
Рекомендации: 100–300 фото на марку, разные ракурсы/цвета/освещение.
Подробная инструкция: docs/BRAND_RU.md

Примеры:
    # посмотреть, что нашлось
    python scripts/train_brand_classifier.py --dry-run

    # обучить MobileNetV3-Small (быстро, хороший старт)
    python scripts/train_brand_classifier.py --epochs 15

    # точнее, но медленнее
    python scripts/train_brand_classifier.py --arch convnext_tiny --epochs 20

    # не перезаписывать готовую модель, а положить рядом
    python scripts/train_brand_classifier.py --output data/models/car_brand_ru.onnx

После обучения в конфиге:
    models:
      brand_onnx: "data/models/car_brand_ru.onnx"
      brand_labels: "data/models/car_brand_labels.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)

DEFAULT_DATA = ROOT / "data/datasets/cars"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RU_BRANDS_HINT = (
    "Lada (ВАЗ), Kia, Hyundai, Toyota, Renault, Skoda, Volkswagen, Nissan, "
    "Ford, Chevrolet, Mitsubishi, Mazda, BMW, Mercedes-Benz, Audi, "
    "Chery, Haval, Geely, Exeed, Omoda, Moskvich, УАЗ, ГАЗ"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение своей модели марки авто")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="папка датасета с train/<Марка>/ (по умолчанию data/datasets/cars)")
    p.add_argument("--val-dir", default=None,
                   help="папка валидации (по умолчанию <data>/val, если есть)")
    p.add_argument("--output", default=str(ROOT / "data/models/car_brand.onnx"),
                   help="куда сохранить ONNX")
    p.add_argument("--labels", default=None,
                   help="куда сохранить car_brand_labels.json (рядом с ONNX)")
    p.add_argument("--meta", default=None,
                   help="куда сохранить car_brand_meta.json (рядом с ONNX)")
    p.add_argument("--arch", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "resnet18", "convnext_tiny", "efficientnet_b0"],
                   help="архитектура (по умолчанию mobilenet_v3_small)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="доля валидации, если папки val нет (0..0.5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0,
                   help="потоки DataLoader (на Windows обычно 0)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-pretrained", action="store_true",
                   help="не скачивать веса ImageNet (обучение с нуля)")
    p.add_argument("--min-images", type=int, default=20,
                   help="предупреждать, если у марки меньше N фото")
    p.add_argument("--dry-run", action="store_true",
                   help="только показать классы и количество фото")
    return p.parse_args()


def _scan_split(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for idx, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                samples.append((path, idx))
    return samples


def _discover(data: Path, val_split: float, seed: int, min_images: int):
    train_dir = data / "train"
    if not train_dir.is_dir():
        raise SystemExit(
            f"Нет папки {train_dir}\n"
            "Создай data/datasets/cars/train/<Марка>/ и положи туда фото.\n"
            f"Типичные марки РФ: {RU_BRANDS_HINT}"
        )
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise SystemExit(
            f"Нужно минимум 2 марки, найдено: {classes or 'ничего'}.\n"
            "Каждая марка — отдельная папка: data/datasets/cars/train/Lada/, .../Kia/, ..."
        )

    counts = {}
    for name in classes:
        counts[name] = sum(1 for p in (train_dir / name).iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS)
    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        raise SystemExit(f"В этих папках нет изображений: {empty}")
    small = [f"{name} ({n})" for name, n in counts.items() if n < min_images]
    if small:
        log.warning("Мало фото (рекомендуется >= %d на марку): %s", min_images, ", ".join(small))

    val_dir = data / "val"
    if val_dir.is_dir() and all((val_dir / name).is_dir() for name in classes):
        train_samples = _scan_split(train_dir, classes)
        val_samples = _scan_split(val_dir, classes)
        source = "отдельная папка val/"
    else:
        all_samples = _scan_split(train_dir, classes)
        rng = random.Random(seed)
        by_class: dict[int, list[tuple[Path, int]]] = {i: [] for i in range(len(classes))}
        for sample in all_samples:
            by_class[sample[1]].append(sample)
        train_samples, val_samples = [], []
        for idx, items in by_class.items():
            rng.shuffle(items)
            n_val = max(1, int(round(len(items) * val_split))) if len(items) >= 3 else 0
            val_samples.extend(items[:n_val])
            train_samples.extend(items[n_val:])
        source = f"автосплит {int(val_split * 100)}%"
        if not val_samples:
            log.warning("Валидационная выборка пуста — точность будет по train.")
            val_samples = list(train_samples)

    if not train_samples:
        raise SystemExit("Не нашлось ни одного изображения для обучения.")
    return classes, train_samples, val_samples, counts, source


def _build_model(arch: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    def weights(name):
        return "DEFAULT" if pretrained else None

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights("mobilenet"))
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=weights("resnet"))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=weights("convnext"))
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights("efficientnet"))
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:  # pragma: no cover - argparse ограничивает выбор
        raise ValueError(f"Неизвестная архитектура: {arch}")
    return model


def _make_transforms(size: int):
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(round(size * 1.14))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def main() -> int:
    args = _parse_args()
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data

    classes, train_samples, val_samples, counts, split_source = _discover(
        data, args.val_split, args.seed, args.min_images)

    log.info("Классов: %d | train: %d | val: %d | сплит: %s",
             len(classes), len(train_samples), len(val_samples), split_source)
    for name, n in counts.items():
        log.info("  %-22s %4d фото", name, n)
    if args.dry_run:
        log.info("--dry-run: обучение не запускалось.")
        return 0

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
    except ImportError:
        log.error("Нет torch/torchvision/Pillow. Установи: pip install torch torchvision")
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tf, val_tf = _make_transforms(args.input_size)

    class FolderDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, label = self.samples[index]
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                    image = self.transform(image)
                return image, label
            except Exception as exc:
                log.warning("Пропускаю битое фото %s: %s", path, exc)
                # чёрная заглушка, чтобы обучение не падало на одном файле
                return torch.zeros(3, args.input_size, args.input_size), label

    train_ds = FolderDataset(train_samples, train_tf)
    val_ds = FolderDataset(val_samples, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Устройство: %s | архитектура: %s | вход: %dx%d",
             device, args.arch, args.input_size, args.input_size)

    model = _build_model(args.arch, len(classes), pretrained=not args.no_pretrained)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    best_path = output.with_suffix(".pt")
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            total += labels.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
        train_loss = running / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1)
                val_correct += int((pred == labels).sum().item())
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)
        scheduler.step()

        marker = ""
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "classes": classes,
                "input_size": args.input_size,
                "val_accuracy": val_acc,
            }, best_path)
            marker = "  <- лучшая"
        log.info("epoch %2d/%2d | loss %.4f | train acc %.3f | val acc %.3f%s",
                 epoch, args.epochs, train_loss, train_acc, val_acc, marker)

    # ----- загрузка лучшей модели и экспорт в ONNX -----
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:  # старые версии torch
            checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    model.eval().to("cpu")

    if output.exists():
        backup = output.with_suffix(output.suffix + ".bak")
        try:
            shutil.copy2(output, backup)
            log.info("Старая модель сохранена как %s", backup)
        except OSError as exc:
            log.warning("Не удалось сделать backup %s: %s", output, exc)

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    export_kwargs = dict(
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(output), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(output), **export_kwargs)
    log.info("ONNX сохранён: %s", output)

    labels_path = Path(args.labels) if args.labels else output.parent / "car_brand_labels.json"
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = Path(args.meta) if args.meta else output.parent / "car_brand_meta.json"
    if not meta_path.is_absolute():
        meta_path = ROOT / meta_path
    meta = {
        "input_size": args.input_size,
        "resize": "shortest_center_crop",
        "resize_ratio": 1.14,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": classes,
        "arch": args.arch,
        "val_accuracy": round(float(best_acc), 4),
        "samples": {"train": len(train_samples), "val": len(val_samples)},
        "dataset": str(data),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Готово: %s + %s + %s", output, labels_path.name, meta_path.name)
    log.info("Включи в конфиге: car.brand_enabled: true, models.brand_onnx: \"%s\"",
             output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
