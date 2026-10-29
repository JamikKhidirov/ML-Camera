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
from datetime import datetime, timedelta

from src.config import load_config, init_logging
from src.database import Database
from src.notifier import Notifier
from src.camera_manager import CameraManager

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
        if data.get("animals"):
            lines.append("\n🐾 Животные:")
            for r in data["animals"]:
                lines.append(f"  • {r['label']} — {r['cnt']} раз")
        if data.get("packages"):
            lines.append("\n📦 Оставленные вещи:")
            for r in data["packages"]:
                lines.append(f"  • {r['label']} — {r['cnt']} раз")
        if (not data["faces"] and not data["cars"] and not data["unknown"]
                and not data.get("animals") and not data.get("packages")):
            lines.append("\nСобытий не было.")
        notifier.send_event("Дайджест", "\n".join(lines))
        log.info("Дайджест отправлен.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLCamera — умная камера")
    p.add_argument("--config", default=None,
                   help="путь к config.yaml (по умолчанию config/config.yaml)")
    p.add_argument("--cameras", type=int, nargs="*", default=None,
                   help="индексы камер из config (например 0 1 2)")
    p.add_argument("--once", action="store_true", help="отработать ~30 кадров и выйти")
    p.add_argument("--self-test", action="store_true", help="отправить тестовое уведомление")
    p.add_argument("--register", action="store_true", help="подготовить лица из data/datasets/raw/"
                    " (создать папки, вырезать, запомнить) и выйти")
    p.add_argument("--preview", action="store_true", help="живое окно с рамками (вкл. ui.preview_enabled)")
    p.add_argument("--recording", action="store_true", help="запись клипов событий (вкл. recording.enabled)")
    p.add_argument("--web", action="store_true", help="запустить веб-панель истории событий")
    p.add_argument("--no-web", action="store_true",
                   help="не запускать веб-панель, даже если web.autostart: true")
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
                  f"python main.py --config config/config.yaml")
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

    # Резервный источник: веб-камера ноутбука. Если в конфиге нет рабочих
    # камер (или все RTSP-адреса — заглушки), панель всё равно показывает
    # живой поток, а не пустой экран.
    camera_cfg = cfg.get("camera", {}) or {}
    fallback_enabled = (bool(camera_cfg.get("fallback_webcam", True))
                        and args.cameras is None and not args.once)
    fallback_cam = {"name": "Ноутбук (веб-камера)", "source": 0, "enabled": True}
    if not enabled:
        if not fallback_enabled:
            log.error("Нет включённых камер в %s (cameras[].enabled)", args.config or "config.yaml")
            db.close()
            return 1
        cameras = [fallback_cam]
        enabled = cameras
        log.warning("В конфиге нет включённых камер — подключаю веб-камеру ноутбука")

    max_frames = 30 if args.once else None
    stop = threading.Event()

    # Камеры живут в отдельных потоках CameraManager: так веб-панель может
    # показать статус, остановить поток и запустить его снова без перезапуска
    # процесса (модели остаются в памяти).
    manager = CameraManager(cfg, db, notifier, cameras=cameras)

    # Веб-панель истории событий. По умолчанию стартует вместе с камерой,
    # если в конфиге web.enabled + web.autostart; флаг --web включает принудительно.
    web_server = None
    web_cfg = cfg.get("web", None) or {}
    web_enabled = isinstance(web_cfg, dict) and bool(web_cfg.get("enabled", False))
    web_autostart = isinstance(web_cfg, dict) and bool(web_cfg.get("autostart", False))
    if not args.once and not args.no_web and (args.web or (web_enabled and web_autostart)):
        try:
            from src.web_panel import start_background_server
            web_server = start_background_server(cfg, db=db, manager=manager)
        except Exception:
            log.exception("Не удалось запустить веб-панель")

    # Открыть панель в браузере — «одна команда и всё работает».
    if web_server is not None and bool(web_cfg.get("open_browser", True)):
        import webbrowser
        bound_host, bound_port = web_server.server_address[:2]
        shown = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
        url = f"http://{shown}:{bound_port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        log.info("Открываю панель в браузере: %s", url)

    def _sig(*_):
        stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # утренний дайджест (отдельный поток)
    digest_t = threading.Thread(target=_digest_thread, args=(cfg, db, notifier, stop), daemon=True)
    if not args.once:
        digest_t.start()

    manager.start_status_writer()
    started = manager.start_enabled(max_frames)
    log.info("Запущено камер: %d из %d", started, len(manager))

    # Если все настроенные камеры не открылись — добавляем веб-камеру.
    if fallback_enabled and not any(
            cam.get("name") == fallback_cam["name"] for cam in cameras):
        deadline = time.time() + 10.0
        while not stop.is_set() and time.time() < deadline:
            states = [item.status for item in manager]
            if states and all(state == "error" for state in states):
                log.warning("Ни одна камера не открылась — включаю веб-камеру ноутбука")
                webcam = manager.add_camera(fallback_cam["name"], fallback_cam["source"])
                webcam.start(max_frames)
                break
            if any(state in ("running", "starting") for state in states):
                # даём камерам шанс: ошибка RTSP проявляется за доли секунды,
                # а загрузка моделей занимает несколько секунд
                stop.wait(0.5)
                if any(item.status == "running" for item in manager):
                    break
                continue
            stop.wait(0.5)

    try:
        if args.once:
            manager.wait(timeout=600)
        else:
            while not stop.is_set():
                # Если панель работает, процесс не выходит после остановки всех
                # камер: из панели камеру можно запустить снова.
                if web_server is None and not manager.any_alive:
                    break
                stop.wait(0.3)
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    finally:
        stop.set()
        manager.stop_all(timeout=5.0)
        manager.stop_status_writer()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        db.close()

    if args.once:
        states = [item.status for item in manager]
        if states and all(state == "error" for state in states):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())