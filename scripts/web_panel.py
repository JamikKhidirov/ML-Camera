"""Запуск веб-панели истории событий отдельным процессом.

Примеры:
    python scripts/web_panel.py --config config/config.yaml
    python scripts/web_panel.py --config config/config.yaml --port 8088
    python scripts/web_panel.py --config config/config.yaml --host 0.0.0.0 --token мой_пароль
    python scripts/web_panel.py --config config/config.yaml --open

Панель читает ту же SQLite-базу, что и камера (data/db/ml_camera.db),
поэтому может работать параллельно с `python main.py`.

Важно: в этом режиме доступен только просмотр. Статусы камер читаются из
data/db/camera_status.json (его пишет процесс камеры), а кнопки
старт/стоп/снимок отключены — ими управляет только панель, запущенная
внутри процесса камеры (`python main.py --web`).
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import init_logging, load_config  # noqa: E402
from src.web_panel import serve_forever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MLCamera — веб-панель истории событий")
    parser.add_argument("--config", default=None,
                        help="конфиг (по умолчанию config/config.yaml через main.py)")
    parser.add_argument("--host", default=None, help="адрес (например 127.0.0.1 или 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="порт (по умолчанию 8088)")
    parser.add_argument("--token", default=None,
                        help="токен доступа (перекрывает web.token в конфиге)")
    parser.add_argument("--open", action="store_true", help="открыть панель в браузере")
    args = parser.parse_args()

    cfg = load_config(args.config or "config/config.yaml")
    init_logging(cfg.logging.level, cfg.logging.resolve(cfg.logging.file))
    if args.token is not None:
        cfg._data.setdefault("web", {})["token"] = args.token

    if args.open:
        host = args.host or str((cfg.get("web", {}) or {}).get("host", "127.0.0.1"))
        port = args.port or int((cfg.get("web", {}) or {}).get("port", 8088))
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = f"http://{shown}:{port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    serve_forever(cfg, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
