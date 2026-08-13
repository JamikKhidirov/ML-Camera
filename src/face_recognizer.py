"""Распознавание лиц членов семьи через InsightFace (ArcFace).

Единоразово для каждого человека считаем embedding (вектор лица),
а на видео сравниваем лицо с базой по косинусному сходству.
"""
from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import Config

log = logging.getLogger(__name__)

EMBEDDINGS_FILE = "data/db/embeddings.json"


@dataclass
class FaceMatch:
    name: str
    similarity: float
    bbox: tuple
    unknown: bool = False


@dataclass
class DetectedFace:
    embedding: np.ndarray
    det_score: float
    bbox: tuple


class FaceRecognizer:
    """Инкапсулирует детекцию лиц и сравнение с базой семьи."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.threshold = float(cfg.face.similarity_threshold)
        self.embeddings: dict[str, np.ndarray] = {}
        self._app = None
        self._load_embeddings()

    # ---------- модель ----------
    @property
    def app(self):
        if self._app is None:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            model_name = str(self.cfg.face.model_name)
            # Провайдеры из конфига (DirectML/CUDA/CPU), недоступные — пропускаем
            wanted = list(self.cfg.face.get("providers", ["CPUExecutionProvider"]))
            available = set(ort.get_available_providers())
            providers = [p for p in wanted if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
            log.info("Провайдеры InsightFace: %s", providers)
            self._app = FaceAnalysis(name=model_name, providers=providers)
            self._app.prepare(ctx_id=-1, det_size=tuple(int(x) for x in self.cfg.face.detection_size))
            log.info("InsightFace %s готова", model_name)
        return self._app

    # ---------- база ----------
    def _embedding_path(self) -> Path:
        p = self.cfg.database.resolve(EMBEDDINGS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_embeddings(self) -> None:
        path = self._embedding_path()
        if not path.exists():
            log.warning("База эмбеддингов не найдена: %s. Запусти scripts/register_faces.py", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        log.info("База лиц загружена: %s", list(self.embeddings.keys()))

    def register(self, name: str, embedding: np.ndarray) -> None:
        self.embeddings[name] = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._save()

    def remove(self, name: str) -> bool:
        if name in self.embeddings:
            del self.embeddings[name]
            self._save()
            return True
        return False

    def _save(self) -> None:
        with open(self._embedding_path(), "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in self.embeddings.items()}, f, ensure_ascii=False)

    def known_names(self) -> list[str]:
        return list(self.embeddings.keys())

    # ---------- анализ ----------
    def detect(self, frame) -> list[DetectedFace]:
        faces = self.app.get(frame)
        out = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(DetectedFace(embedding=emb, det_score=float(f.det_score), bbox=(x1, y1, x2, y2)))
        return out

    def recognize(self, frame) -> tuple[list[FaceMatch], list[DetectedFace]]:
        """Возвращает (совпадения, все найденные лица)."""
        faces = self.detect(frame)
        matches: list[FaceMatch] = []
        for face in faces:
            if face.det_score < float(self.cfg.face.min_det_score):
                continue
            name, sim = "unknown", 0.0
            for known, emb in self.embeddings.items():
                s = float(np.dot(face.embedding, emb))
                if s > sim:
                    sim, name = s, known
            if sim >= self.threshold:
                matches.append(FaceMatch(name=name, similarity=sim, bbox=face.bbox))
            else:
                matches.append(FaceMatch(name="unknown", similarity=sim, bbox=face.bbox, unknown=True))
        return matches, faces