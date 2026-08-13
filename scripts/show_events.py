"""Просмотр событий из БД.

Запуск: python scripts/show_events.py [кол-во]
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.database import Database  # noqa: E402


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    cfg = load_config()
    db = Database(cfg)
    for r in db.recent_events(limit):
        ts = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {r['camera'] or '?'}: {r['title']} | {r.get('extra', '')} | {r.get('image_path', '')}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())