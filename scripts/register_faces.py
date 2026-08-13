"""Регистрация членов семьи в базе (эмбеддинги лиц).

Берёт вырезанные лица из data/datasets/faces/<имя>/ и для каждого
человека считает средний вектор лица, сохраняет в data/db/embeddings.json.

Запуск: python scripts/register_faces.py
Повторный запуск ПЕРЕЗАПИШЕТ базу (используй с осторожностью).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402
from src.face_recognizer import FaceRecognizer  # noqa: E402

init_logging("INFO")
log = logging.getLogger(__name__)

FACES = ROOT / "data/datasets/faces"
MIN_PER_PERSON = 3  # меньше — не берём человека в базу


def main() -> int:
    cfg = load_config()
    recognizer = FaceRecognizer(cfg)

    if not FACES.exists():
        log.error("Нет папки %s. Сначала: python scripts/collect_faces.py", FACES)
        return 1

    for person_dir in sorted(p for p in FACES.iterdir() if p.is_dir()):
        name = person_dir.name
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(images) < MIN_PER_PERSON:
            log.warning("У '%s' только %d фото лица (нужно >= %d), пропускаю",
                        name, len(images), MIN_PER_PERSON)
            continue
        embeddings = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = recognizer.app.get(img)
            if faces:
                embeddings.append(np.asarray(faces[0].normed_embedding, dtype=np.float32))
        if not embeddings:
            log.error("Не удалось посчитать лица для '%s'", name)
            continue
        emb = np.mean(embeddings, axis=0)
        emb /= np.linalg.norm(emb)
        recognizer.register(name, emb)
        log.info("Зарегистрирован: %-10s (фото=%d, лиц в базе=%d)",
                 name, len(images), len(recognizer.known_names()))

    log.info("База готова: %s", recognizer.known_names())
    log.info("Проверить распознавание можно запуском камеры.")
    return 0


if __name__ == "__main__":
    sys.exit(main())