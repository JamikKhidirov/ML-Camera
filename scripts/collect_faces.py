"""Сбор датасета лиц: вырезает лица из сырых фото.

Как пользоваться:
  1) Кинь 20-30 фотографий КАЖДОГО члена семьи в папки:
       data/datasets/raw/мама/    data/datasets/raw/папа/
       data/datasets/raw/я/       data/datasets/raw/сестра1/
       data/datasets/raw/сестра2/
     Требования к фото — docs/DATASET.md
  2) Запусти:  python scripts/collect_faces.py
  3) Скрипт вырежет лица и положит в data/datasets/faces/<имя>/

После этого зарегистрируй лица в базе:
      python scripts/register_faces.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)

RAW = ROOT / "data/datasets/raw"
FACES = ROOT / "data/datasets/faces"
MIN_DET_SCORE = 0.6


def main() -> int:
    cfg = load_config()

    print("Загружаю модель InsightFace...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=str(cfg.face.model_name), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=tuple(int(x) for x in cfg.face.detection_size))

    if not RAW.exists():
        log.error("Нет папки %s. Сначала: python scripts/make_dirs.py", RAW)
        return 1

    total_ok = 0
    for person_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        out_dir = FACES / person_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            log.warning("У '%s' нет фотографий в %s", person_dir.name, person_dir)
            continue
        kept = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("  Не прочитал: %s", img_path.name)
                continue
            faces = app.get(img)
            for f in faces:
                if f.det_score < MIN_DET_SCORE:
                    continue
                x1, y1, x2, y2 = (int(v) for v in f.bbox)
                face = img[max(0, y1):y2, max(0, x1):x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (224, 224))
                dest = out_dir / f"{img_path.stem}_{kept}.jpg"
                cv2.imwrite(str(dest), face)
                kept += 1
                total_ok += 1
        log.info("%-10s фото: %d -> вырезано лиц: %d", person_dir.name, len(images), kept)

    print(f"\nГотово. Всего лиц сохранено: {total_ok}")
    print("Теперь запусти: python scripts/register_faces.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())