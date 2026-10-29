"""Просмотр событий из БД.

Запуск:
  python scripts/show_events.py [кол-во]
  python scripts/show_events.py [кол-во] --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.database import Database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Показать последние события")
    parser.add_argument("limit", nargs="?", type=int, default=50, help="сколько событий (по умолчанию 50)")
    parser.add_argument("--config", default=None,
                        help="путь к config.yaml (напр. config/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db = Database(cfg)
    rows = db.recent_events(args.limit)
    if not rows:
        print("Событий пока нет.")
    for r in rows:
        ts = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        extra = r.get("extra") or ""
        image_path = r.get("image_path") or ""
        print(f"[{ts}] {r['camera'] or '?'}: {r['title']} | {extra} | {image_path}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())