"""Обучение классификатора МАРКИ автомобиля на датасете Stanford Cars.

Опционально. Пока модель не обучена, марка будет "unknown" — это не ломает
распознавание номеров и цвета. Если хочешь марку:

  1) Скачай Stanford Cars (mat-файлы) ~1.8 ГБ:
       https://ai.stanford.edu/~jkrause/cars/car_dataset.html
     Положи в data/datasets/cars/annotations/ и data/datasets/cars/images/
     (или измени пути в скрипте)
  2) Установи torch:  pip install torch torchvision
  3) Запусти:  python scripts/train_brand_classifier.py
  4) Экспортирует data/models/car_brand.onnx + car_brand_labels.json
     и включит марки в уведомлениях.

Если удобнее не качать 16 тыс. фото — можно обучить на ~500-1000 своих фото
марок машин во дворе (класс = название марки, файлы в data/datasets/cars/train/<марка>/).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)

CARS_DIR = ROOT / "data/datasets/cars"


def _labels_from_dir() -> list[str]:
    """Если юзер сложил свои фото в train/<марка>/, вернём список марок."""
    train_dir = CARS_DIR / "train"
    if train_dir.exists():
        return sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    return []


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        log.error("Нет torch/torchvision. Установи: pip install torch torchvision")
        return 1

    # ---- 1. данные ----
    labels = _labels_from_dir()
    if not labels:
        log.info("Своих фото нет — попробую Stanford Cars (ann_files).")
        labels = ["unknown"]  # здесь твой парсер Stanford Cars
        # TODO: подключить парсер mat-файлов Stanford Cars и собрать список марок
        log.error("Реализуй парсер Stanford Cars (файл scripts/train_brand_classifier.py) "
                  "или положи свои фото в data/datasets/cars/train/<марка>/")
        return 1

    log.info("Классы марок: %s", labels)

    class DirDataset(Dataset):
        def __init__(self, root: Path, transforms_):
            self.items = []
            for label, folder in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
                for img in folder.glob("*.jpg"):
                    self.items.append((str(img), label))
            self.tf = transforms_

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            import cv2
            path, label = self.items[i]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.tf(img)
            return img, label

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = DirDataset(CARS_DIR / "train", tf)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # ---- 2. модель MobileNetV3Small ----
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, len(labels))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- 3. обучение ----
    epochs = 8
    for epoch in range(epochs):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
        log.info("epoch %d/%d loss=%.4f acc=%.3f", epoch + 1, epochs, running / max(1, len(loader)),
                 correct / max(1, total))

    # ---- 4. экспорт в ONNX ----
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    out_path = ROOT / "data/models/car_brand.onnx"
    torch.onnx.export(model, dummy, str(out_path),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    labels_path = ROOT / "data/models/car_brand_labels.json"
    labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    log.info("Модель марки готова: %s + %s", out_path, labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())