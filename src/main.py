"""Точка входа MLCamera.

Примеры запуска:
  python -m src.main                 # все включённые камеры из config
  python -m src.main --self-test     # проверка уведомлений
  python -m src.main --once          # обработать ~30 кадров и выйти (отладка)
  python -m src.main --config my.yaml
  python -m src.main --cameras 0 2   # запустить только камеры с индексами 0 и 2
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from src.config import load_config, init_logging
from src.database import Database
from src.notifier import Notifier
from src.pipeline import Pipeline

log = logging.getLogger(__name__)


def _digest_thread(cfg, db: Database, notifier: Notifier, stop: threading.Event) -> None:
    """Утренний дайджест: раз в сутки шлёт сводку за последние 24 часа."""
    d = cfg.get("digest", None)
    enabled = bool(d.get("enabled", False)) if d is not None else False
    if not enabled:
        return
    hours = int(d.get("hours", 24))
    hh, mm = (int(x) for x in d.get("time", "09:00").split(":")[:2])
    log.info("Дайджест включён: время %02d:%02d, окно %dh", hh, mm, hours)
    while not stop.is_set():
        now = datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        delay = (target - now).total_seconds()
        while delay > 0 and not stop.is_set():
            time.sleep(min(60.0, delay))
            delay -= 60.0
        if stop.is_set():
            break
        data = db.digest_data(hours)
        lines = [f"Сводка камер за последние {hours} ч:"]
        if data["faces"]:
            lines.append("\n👤 Люди:")
            for r in data["faces"]:
                lines.append(f"  • {r['person']} — {r['cnt']} раз")
        if data["unknown"]:
            lines.append(f"\n❓ Незнакомцы — {data['unknown']} раз")
        if data["cars"]:
            lines.append("\n🚗 Машины:")
            for r in data["cars"]:
                plate = r["plate"] or "номер не распознан"
                lines.append(f"  • {plate} ({r['color']} {r['brand']}) — {r['cnt']} раз")
        if not data["faces"] and not data["cars"] and not data["unknown"]:
            lines.append("\nСобытий не было.")
        notifier.send_event("Дайджест", "\n".join(lines))
        log.info("Дайджест отправлен.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLCamera — умная камера")
    p.add_argument("--config", default=None, help="путь к config.yaml "
                    "(например config/config.pc.yaml или config/config.vps.yaml)")
    p.add_argument("--cameras", type=int, nargs="*", default=None,
                   help="индексы камер из config (например 0 1 2)")
    p.add_argument("--once", action="store_true", help="отработать ~30 кадров и выйти")
    p.add_argument("--self-test", action="store_true", help="отправить тестовое уведомление")
    p.add_argument("--register", action="store_true", help="подготовить лица из data/datasets/raw/"
                    " (создать папки, вырезать, запомнить) и выйти")
    p.add_argument("--preview", action="store_true", help="живое окно с рамками (вкл. ui.preview_enabled)")
    p.add_argument("--recording", action="store_true", help="запись клипов событий (вкл. recording.enabled)")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    init_logging(cfg.logging.level, cfg.logging.resolve(cfg.logging.file))

    # флаги переопределяют конфиг
    if args.preview:
        cfg.ui._data["preview_enabled"] = True
    if args.recording:
        cfg.recording._data["enabled"] = True

    db = Database(cfg)
    notifier = Notifier(cfg)

    if args.self_test:
        notifier.self_test()
        db.close()
        return 0

    if args.register:
        from src.face_setup import register_all
        people = register_all(cfg)
        db.close()
        if people:
            print(f"\nПодготовлено людей: {people}. Можно запускать камеру: "
                  f"python main.py --config config/config.pc.yaml")
            return 0
        print("\nНикто не зарегистрирован: положи фото в data/datasets/raw/<ИМЯ>/ "
              "и повтори регистрацию.")
        return 1

    cameras = cfg.get("cameras", [])
    if isinstance(cameras, dict):
        cameras = cameras.get("cameras") or []
    if args.cameras is not None:
        cameras = [cameras[i] for i in args.cameras if 0 <= i < len(cameras)]
    enabled = [c for c in cameras if isinstance(c, dict) and bool(c.get("enabled", True))]
    if not enabled:
        log.error("Нет включённых камер в %s (cameras[].enabled)", args.config or "config.yaml")
        db.close()
        return 1

    max_frames = 30 if args.once else None
    stop = threading.Event()

    def _sig(*_):
        stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # утренний дайджест (отдельный поток)
    digest_t = threading.Thread(target=_digest_thread, args=(cfg, db, notifier, stop), daemon=True)
    if not args.once:
        digest_t.start()

    # Одна камера (самый частый случай, особенно ПК с окном) — запускаем
    # в ГЛАВНОМ потоке: cv2.imshow/waitKey не любят рабочие потоки.
    if len(enabled) == 1 or args.once:
        cam_cfg = enabled[0]
        pipe = Pipeline(cfg, cam_cfg, db, notifier)
        try:
            pipe.run(max_frames)
        except KeyboardInterrupt:
            log.info("Остановлено пользователем")
        except Exception:
            log.exception("Ошибка камеры '%s'", cam_cfg.get("name", "cam"))
            return 1
        finally:
            stop.set()
            db.close()
        return 0

    executor = ThreadPoolExecutor(max_workers=len(enabled))
    futures = []
    for cam_cfg in enabled:
        pipe = Pipeline(cfg, cam_cfg, db, notifier)
        futures.append(executor.submit(pipe.run, max_frames))

    try:
        for fut in futures:
            fut.result()
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    except Exception:
        log.exception("Ошибка в потоке камеры")
    finally:
        stop.set()
        executor.shutdown(wait=True)
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())