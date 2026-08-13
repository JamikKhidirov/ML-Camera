"""Хранение событий в SQLite: лица, автомобили, номера.

Всё хранится локально на сервере в data/db/ml_camera.db.
Таблицы:
  face_events  — кто и когда попадал в кадр
  car_events   — авто: номер, марка, цвет, когда
  persons      — (опционально) имя + эмбеддинг
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.config import Config

log = logging.getLogger(__name__)


class Database:
    """Потокобезопасная обёртка над SQLite (lock + WAL)."""

    def __init__(self, cfg: Config):
        db_path = cfg.database.resolve(cfg.database.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS face_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    camera TEXT,
                    person TEXT NOT NULL,
                    similarity REAL,
                    image_path TEXT,
                    notified INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS car_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    camera TEXT,
                    plate TEXT,
                    brand TEXT,
                    color TEXT,
                    image_path TEXT,
                    crop_path TEXT,
                    notified INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_face_ts ON face_events (person, ts);
                CREATE INDEX IF NOT EXISTS idx_car_plate ON car_events (plate, ts);
                """
            )

    # ---------- вспомогательное ----------
    def _last_seen(self, table: str, key_col: str, key: str, cooldown_min: float) -> Optional[float]:
        cutoff = time.time() - cooldown_min * 60
        with self._lock:
            row = self._conn.execute(
                f"SELECT MAX(ts) AS last_ts FROM {table} WHERE {key_col}=? AND ts > ?",
                (key, cutoff),
            ).fetchone()
        return row["last_ts"] if row else None

    def is_in_cooldown(self, kind: str, key: str, cooldown_min: float) -> bool:
        """True, если по ключу (человек/номер) уведомление было недавно."""
        table, col = {"face": ("face_events", "person"), "car": ("car_events", "plate")}[kind]
        return self._last_seen(table, col, key, cooldown_min) is not None

    # ---------- события ----------
    def save_face_event(self, person: str, similarity: float, camera: str,
                        image_path: str = "") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO face_events (ts, camera, person, similarity, image_path) VALUES (?,?,?,?,?)",
                (time.time(), camera, person, similarity, image_path),
            )
            return int(cur.lastrowid)

    def save_car_event(self, camera: str, plate: str, brand: str, color: str,
                       image_path: str = "", crop_path: str = "") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO car_events (ts, camera, plate, brand, color, image_path, crop_path) "
                "VALUES (?,?,?,?,?,?,?)",
                (time.time(), camera, plate, brand, color, image_path, crop_path),
            )
            return int(cur.lastrowid)

    # ---------- выборки ----------
    def recent_events(self, limit: int = 50):
        rows = []
        with self._lock:
            rows += [dict(r) for r in self._conn.execute(
                "SELECT ts, camera, person AS title, similarity AS extra, image_path FROM face_events "
                "ORDER BY id DESC LIMIT ?", (limit,))]
            rows += [dict(r) for r in self._conn.execute(
                "SELECT ts, camera, plate AS title, color || ' ' || brand AS extra, image_path FROM car_events "
                "ORDER BY id DESC LIMIT ?", (limit,))]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    def digest_data(self, hours: int = 24) -> dict:
        """Сводка за последние hours: кто приходил, машины, незнакомцы.

        Используется утренним дайджестом (см. src/main.py).
        """
        since = time.time() - hours * 3600
        out = {"faces": [], "cars": [], "unknown": 0}
        with self._lock:
            out["faces"] = [dict(r) for r in self._conn.execute(
                "SELECT person, COUNT(*) AS cnt FROM face_events "
                "WHERE ts > ? AND person != 'unknown' GROUP BY person ORDER BY cnt DESC",
                (since,))]
            out["unknown"] = int(self._conn.execute(
                "SELECT COUNT(*) FROM face_events WHERE ts > ? AND person='unknown'",
                (since,)).fetchone()[0])
            out["cars"] = [dict(r) for r in self._conn.execute(
                "SELECT plate, brand, color, COUNT(*) AS cnt FROM car_events "
                "WHERE ts > ? GROUP BY plate ORDER BY cnt DESC",
                (since,))]
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()