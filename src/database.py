"""Хранение событий в SQLite: лица, автомобили, животные и посылки.

Всё хранится локально на сервере в data/db/ml_camera.db.
Таблицы:
  face_events   — кто и когда попадал в кадр
  car_events    — авто: номер, марка, цвет, когда
  object_events — животные (kind='animal') и оставленные вещи (kind='package')

Методы query_events()/stats()/get_media_path() используются веб-панелью:
они не отдают наружу пути к файлам, только идентификаторы событий.
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
                CREATE TABLE IF NOT EXISTS object_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    camera TEXT,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    title TEXT,
                    confidence REAL DEFAULT 0,
                    stationary_seconds REAL DEFAULT 0,
                    image_path TEXT,
                    crop_path TEXT,
                    notified INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_face_ts ON face_events (person, ts);
                CREATE INDEX IF NOT EXISTS idx_car_plate ON car_events (plate, ts);
                CREATE INDEX IF NOT EXISTS idx_object_kind ON object_events (kind, label, ts);
                CREATE INDEX IF NOT EXISTS idx_object_ts ON object_events (ts);
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

    def is_object_in_cooldown(self, kind: str, label: str, cooldown_min: float) -> bool:
        """Кулдаун для животных/посылок: отдельно по виду события и классу."""
        cutoff = time.time() - cooldown_min * 60
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(ts) AS last_ts FROM object_events "
                "WHERE kind=? AND label=? AND ts > ?",
                (kind, label, cutoff),
            ).fetchone()
        return bool(row and row["last_ts"] is not None)

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

    def save_object_event(self, camera: str, kind: str, label: str, title: str = "",
                          confidence: float = 0.0, stationary_seconds: float = 0.0,
                          image_path: str = "", crop_path: str = "") -> int:
        """kind: 'animal' (животное) или 'package' (оставленная вещь)."""
        if kind not in ("animal", "package"):
            raise ValueError(f"неизвестный kind: {kind}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO object_events (ts, camera, kind, label, title, confidence, "
                "stationary_seconds, image_path, crop_path) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), camera, kind, label, title, confidence,
                 stationary_seconds, image_path, crop_path),
            )
            return int(cur.lastrowid)

    # ---------- выборки ----------
    def recent_events(self, limit: int = 50):
        """Последние события для консоли (scripts/show_events.py)."""
        rows = []
        with self._lock:
            rows += [dict(r) for r in self._conn.execute(
                "SELECT ts, camera, person AS title, similarity AS extra, image_path FROM face_events "
                "ORDER BY id DESC LIMIT ?", (limit,))]
            rows += [dict(r) for r in self._conn.execute(
                "SELECT ts, camera, plate AS title, color || ' ' || brand AS extra, image_path FROM car_events "
                "ORDER BY id DESC LIMIT ?", (limit,))]
            rows += [dict(r) for r in self._conn.execute(
                "SELECT ts, camera, COALESCE(NULLIF(title, ''), label) AS title, "
                "CASE kind WHEN 'animal' THEN 'животное' ELSE 'посылка' END AS extra, "
                "image_path FROM object_events ORDER BY id DESC LIMIT ?", (limit,))]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    def digest_data(self, hours: int = 24) -> dict:
        """Сводка за последние hours: кто приходил, машины, звери, вещи.

        Используется утренним дайджестом (см. src/main.py).
        """
        since = time.time() - hours * 3600
        out = {"faces": [], "cars": [], "unknown": 0, "animals": [], "packages": []}
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
            objects = [dict(r) for r in self._conn.execute(
                "SELECT kind, label, COUNT(*) AS cnt FROM object_events "
                "WHERE ts > ? GROUP BY kind, label ORDER BY cnt DESC",
                (since,))]
        out["animals"] = [r for r in objects if r["kind"] == "animal"]
        out["packages"] = [r for r in objects if r["kind"] == "package"]
        return out

    # ---------- выборки для веб-панели ----------
    @staticmethod
    def _since(hours: Optional[float]) -> float:
        return time.time() - float(hours) * 3600 if hours and hours > 0 else 0.0

    def query_events(self, limit: int = 50, offset: int = 0, kind: Optional[str] = None,
                     camera: Optional[str] = None, q: Optional[str] = None,
                     hours: Optional[float] = None) -> tuple[list[dict], bool]:
        """История событий для веб-панели.

        kind: face | car | animal | package | object | all/None.
        Возвращает (события, есть_ли_ещё). Наружу не отдаются пути к файлам —
        только type/id, а картинка запрашивается через get_media_path().
        """
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        fetch = limit + offset + 1          # +1 — чтобы понять, есть ли ещё
        since = self._since(hours)
        if not kind or kind == "all":
            kinds = {"face", "car", "animal", "package"}
        elif kind == "object":
            kinds = {"animal", "package"}
        elif kind in ("face", "car", "animal", "package"):
            kinds = {kind}
        else:
            kinds = {"face", "car", "animal", "package"}
        search = f"%{q.strip()}%" if q and q.strip() else None
        rows: list[dict] = []

        with self._lock:
            if "face" in kinds:
                sql = ("SELECT id, ts, camera, person, similarity, image_path, NULL AS crop_path "
                       "FROM face_events WHERE ts >= ?")
                params: list = [since]
                if camera:
                    sql += " AND camera=?"
                    params.append(camera)
                if search:
                    sql += " AND person LIKE ?"
                    params.append(search)
                sql += " ORDER BY ts DESC LIMIT ?"
                params.append(fetch)
                for r in self._conn.execute(sql, params):
                    person = r["person"] or "unknown"
                    rows.append({
                        "type": "face", "id": r["id"], "ts": r["ts"],
                        "camera": r["camera"] or "",
                        "title": person if person != "unknown" else "Незнакомец",
                        "subtitle": (f"схожесть {r['similarity']:.2f}"
                                     if r["similarity"] is not None else "лицо"),
                        "score": r["similarity"],
                        "label": person,
                        "has_image": bool(r["image_path"]),
                        "has_crop": False,
                    })
            if "car" in kinds:
                sql = ("SELECT id, ts, camera, plate, brand, color, image_path, crop_path "
                       "FROM car_events WHERE ts >= ?")
                params = [since]
                if camera:
                    sql += " AND camera=?"
                    params.append(camera)
                if search:
                    sql += " AND (plate LIKE ? OR brand LIKE ? OR color LIKE ?)"
                    params += [search, search, search]
                sql += " ORDER BY ts DESC LIMIT ?"
                params.append(fetch)
                for r in self._conn.execute(sql, params):
                    brand = r["brand"] or ""
                    color = r["color"] or ""
                    details = " ".join(x for x in (color, brand) if x and x != "unknown")
                    rows.append({
                        "type": "car", "id": r["id"], "ts": r["ts"],
                        "camera": r["camera"] or "",
                        "title": r["plate"] or "Автомобиль",
                        "subtitle": details or "номер не распознан",
                        "score": None,
                        "label": brand or "car",
                        "has_image": bool(r["image_path"]),
                        "has_crop": bool(r["crop_path"]),
                    })
            object_kinds = kinds & {"animal", "package"}
            if object_kinds:
                placeholders = ",".join("?" for _ in object_kinds)
                sql = (f"SELECT id, ts, camera, kind, label, title, confidence, "
                       f"stationary_seconds, image_path, crop_path FROM object_events "
                       f"WHERE ts >= ? AND kind IN ({placeholders})")
                params = [since, *sorted(object_kinds)]
                if camera:
                    sql += " AND camera=?"
                    params.append(camera)
                if search:
                    sql += " AND (label LIKE ? OR title LIKE ?)"
                    params += [search, search]
                sql += " ORDER BY ts DESC LIMIT ?"
                params.append(fetch)
                for r in self._conn.execute(sql, params):
                    title = r["title"] or r["label"]
                    if r["kind"] == "animal":
                        subtitle = "животное в кадре"
                    else:
                        subtitle = "оставленная вещь"
                        if r["stationary_seconds"]:
                            subtitle += f" · без движения {int(r['stationary_seconds'])} с"
                    rows.append({
                        "type": r["kind"], "id": r["id"], "ts": r["ts"],
                        "camera": r["camera"] or "",
                        "title": title, "subtitle": subtitle,
                        "score": r["confidence"], "label": r["label"],
                        "has_image": bool(r["image_path"]),
                        "has_crop": bool(r["crop_path"]),
                    })

        rows.sort(key=lambda r: r["ts"], reverse=True)
        page = rows[offset:offset + limit]
        return page, len(rows) > offset + limit

    def stats(self, hours: float = 24) -> dict:
        """Счётчики для верхних карточек веб-панели."""
        since = self._since(hours)
        with self._lock:
            def count(sql: str, params=()) -> int:
                return int(self._conn.execute(sql, params).fetchone()[0])

            faces = count("SELECT COUNT(*) FROM face_events WHERE ts >= ?", (since,))
            unknown = count("SELECT COUNT(*) FROM face_events WHERE ts >= ? AND person='unknown'",
                            (since,))
            cars = count("SELECT COUNT(*) FROM car_events WHERE ts >= ?", (since,))
            animals = count("SELECT COUNT(*) FROM object_events WHERE ts >= ? AND kind='animal'",
                            (since,))
            packages = count("SELECT COUNT(*) FROM object_events WHERE ts >= ? AND kind='package'",
                             (since,))
            by_camera = [dict(r) for r in self._conn.execute(
                "SELECT camera, COUNT(*) AS cnt FROM ("
                " SELECT camera, ts FROM face_events"
                " UNION ALL SELECT camera, ts FROM car_events"
                " UNION ALL SELECT camera, ts FROM object_events"
                ") WHERE ts >= ? GROUP BY camera ORDER BY cnt DESC", (since,))]
            last = self._conn.execute(
                "SELECT MAX(ts) AS ts FROM ("
                " SELECT ts FROM face_events UNION ALL SELECT ts FROM car_events"
                " UNION ALL SELECT ts FROM object_events)").fetchone()
        return {
            "hours": hours,
            "total": faces + cars + animals + packages,
            "faces": faces,
            "unknown": unknown,
            "cars": cars,
            "animals": animals,
            "packages": packages,
            "by_camera": by_camera,
            "last_ts": last["ts"] if last and last["ts"] is not None else None,
        }

    def timeline(self, hours: float = 24, buckets: int = 24) -> dict:
        """Активность по часам для графика в веб-панели.

        Возвращает ровные корзины (даже пустые), чтобы фронтенду не пришлось
        самому выравнивать время. Суммы по типам событий считает SQLite.
        """
        hours = max(1.0, min(float(hours), 168.0))
        buckets = max(4, min(int(buckets), 96))
        now = time.time()
        since = now - hours * 3600
        span = max(1.0, hours * 3600)
        width = span / buckets
        with self._lock:
            rows = self._conn.execute(
                "SELECT CAST((ts - ?) / ? AS INTEGER) AS b, kind, COUNT(*) AS cnt FROM ("
                " SELECT ts, 'face' AS kind FROM face_events WHERE ts >= ?"
                " UNION ALL SELECT ts, 'car' FROM car_events WHERE ts >= ?"
                " UNION ALL SELECT ts, 'animal' FROM object_events"
                "   WHERE ts >= ? AND kind='animal'"
                " UNION ALL SELECT ts, 'package' FROM object_events"
                "   WHERE ts >= ? AND kind='package'"
                ") GROUP BY b, kind",
                (since, width, since, since, since, since),
            ).fetchall()
        items = [
            {"ts": since + i * width, "total": 0, "face": 0,
             "car": 0, "animal": 0, "package": 0}
            for i in range(buckets)
        ]
        for r in rows:
            idx = max(0, min(buckets - 1, int(r["b"])))
            kind = r["kind"]
            if kind in items[idx]:
                items[idx][kind] += int(r["cnt"])
                items[idx]["total"] += int(r["cnt"])
        return {"hours": hours, "buckets": buckets, "items": items,
                "max": max((item["total"] for item in items), default=0)}

    def camera_stats(self, hours: float = 24) -> dict[str, dict]:
        """По каждой камере: сколько событий за период и когда было последнее."""
        since = self._since(hours)
        with self._lock:
            rows = self._conn.execute(
                "SELECT camera, MAX(ts) AS last_ts,"
                " SUM(CASE WHEN ts >= ? THEN 1 ELSE 0 END) AS cnt "
                "FROM ("
                " SELECT camera, ts FROM face_events"
                " UNION ALL SELECT camera, ts FROM car_events"
                " UNION ALL SELECT camera, ts FROM object_events"
                ") WHERE camera IS NOT NULL AND camera != '' GROUP BY camera",
                (since,),
            ).fetchall()
        return {
            str(r["camera"]): {
                "last_ts": float(r["last_ts"] or 0.0),
                "events": int(r["cnt"] or 0),
            }
            for r in rows
        }

    def get_media_path(self, kind: str, event_id: int, crop: bool = False) -> Optional[str]:
        """Путь к фото события по type/id. kind — как в query_events()."""
        event_id = int(event_id)
        with self._lock:
            if kind == "face":
                row = self._conn.execute(
                    "SELECT image_path FROM face_events WHERE id=?", (event_id,)).fetchone()
                path = row["image_path"] if row else None
            elif kind == "car":
                row = self._conn.execute(
                    "SELECT image_path, crop_path FROM car_events WHERE id=?", (event_id,)).fetchone()
                path = (row["crop_path"] or row["image_path"]) if row else None
                if row and not crop:
                    path = row["image_path"]
            elif kind in ("animal", "package", "object"):
                sql = "SELECT image_path, crop_path FROM object_events WHERE id=?"
                params = [event_id]
                if kind != "object":
                    sql += " AND kind=?"
                    params.append(kind)
                row = self._conn.execute(sql, params).fetchone()
                path = ((row["crop_path"] or row["image_path"]) if crop else row["image_path"]) if row else None
            else:
                return None
        return str(path) if path else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()