"""Подготовка людей: папки, вырезка лиц, регистрация в базе.

Это то, что нужно сделать ОДИН раз после того, как в папки
data/datasets/raw/<Имя>/ положены фотографии человека (см. docs/DATASET.md):
  1) создать папки людей из config (people.names);
  2) вырезать лица (collect_faces);
  3) посчитать и сохранить эмбеддинги (register_faces).

Вызывается и из scripts/collect_faces.py / register_faces.py, и из
основного приложения:  python main.py --register
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import Config

log = logging.getLogger(__name__)

MIN_DET_SCORE = 0.6
MIN_PER_PERSON = 3


def ensure_people_dirs(cfg: Config) -> None:
    """Создаёт data/datasets/raw/<Имя>/ для каждого имени из конфига."""
    raw = cfg.resolve("data/datasets/raw")
    raw.mkdir(parents=True, exist_ok=True)
    names = cfg.people.get("names", {}) or {}
    for name in names:
        folder = raw / str(name)
        folder.mkdir(parents=True, exist_ok=True)
        readme = folder / "README.txt"
        if not readme.exists():
            readme.write_text(
                f"Это папка фотографий: {name}.\n"
                f"СЮДА клади 10-25 фотографий этого человека (jpg/png), лицо крупно и чётко.\n"
                f"Ничего не переименовывай и не обрезай.\n"
                f"Готово? Запусти один раз:  python main.py --register\n"
                f"(и тогда камера узнает {name} и будет писать: '{name} у камеры')\n",
                encoding="utf-8")
    (cfg.resolve("data/datasets/auto_unknown")).mkdir(parents=True, exist_ok=True)


def collect_faces(cfg: Config) -> int:
    """Вырезает лица из data/datasets/raw/<имя>/ в data/datasets/faces/<имя>/.

    Возвращает число сохранённых лиц.
    """
    raw = cfg.resolve("data/datasets/raw")
    faces_dir = cfg.resolve("data/datasets/faces")
    faces_dir.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        log.error("Нет папки %s — положи туда фото людей", raw)
        return 0

    print("Загружаю модель InsightFace...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=str(cfg.face.model_name), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=tuple(int(x) for x in cfg.face.detection_size))

    total_ok = 0
    for person_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        out_dir = faces_dir / person_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in person_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            log.warning("У '%s' нет фотографий в %s", person_dir.name, person_dir)
            continue
        kept = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("  Не прочитал: %s", img_path.name)
                continue
            for f in app.get(img):
                if f.det_score < MIN_DET_SCORE:
                    continue
                x1, y1, x2, y2 = (int(v) for v in f.bbox)
                face = img[max(0, y1):y2, max(0, x1):x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (224, 224))
                cv2.imwrite(str(out_dir / f"{img_path.stem}_{kept}.jpg"), face)
                kept += 1
                total_ok += 1
        log.info("%-10s фото: %d -> вырезано лиц: %d", person_dir.name, len(images), kept)

    print(f"\nВырезано лиц: {total_ok}")
    return total_ok


def register_faces(cfg: Config) -> int:
    """Считает средние эмбеддинги из data/datasets/faces/<имя>/ и сохраняет в БД.

    Возвращает число зарегистрированных людей.
    """
    from src.face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer(cfg)
    faces_dir = cfg.resolve("data/datasets/faces")
    if not faces_dir.exists():
        log.error("Нет папки %s — сначала вырежи лица (--register)", faces_dir)
        return 0

    registered = 0
    for person_dir in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        name = person_dir.name
        images = sorted(p for p in person_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
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
        registered += 1
        log.info("Зарегистрирован: %-10s (фото=%d)", name, len(images))

    log.info("Готово, в базе: %s", recognizer.known_names())
    return registered


def register_all(cfg: Config) -> int:
    """--register: папки -> вырезать лица -> зарегистрировать. Всё разом."""
    ensure_people_dirs(cfg)
    collect_faces(cfg)
    return register_faces(cfg)