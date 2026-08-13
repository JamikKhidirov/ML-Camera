"""Одной кнопкой развернуть MLCamera на VPS прямо с компьютера (Windows/Linux).

Что нужно:
  1) вписать host / user / password сервера в файл pc/deploy.yaml
     (скопируй из pc/deploy.example.yaml; пароль НЕ попадает в git);
  2) запустить:  python pc/deploy_vps.py

Что делает скрипт:
  - упаковывает проект в zip (вместе с твоим конфигом и базой лиц);
  - по SSH заливает zip на сервер;
  - запускает scripts/deploy_server.sh (ставит Docker, распаковывает,
    собирает и запускает контейнер).

Требуется:  pip install -r requirements-build.txt  (paramiko)
Запуск с явными данными (не нужен deploy.yaml):
  python pc/deploy_vps.py --host 1.2.3.4 --user root --password СЕКРЕТ
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEPLOY_FILE = ROOT / "pc" / "deploy.yaml"
DEPLOY_EXAMPLE = ROOT / "pc" / "deploy.example.yaml"


def _latest_zip() -> Path:
    zips = sorted((ROOT / "dist").glob("ml_camera_*.zip")) if (ROOT / "dist").exists() else []
    if not zips:
        raise SystemExit("Нет zip. Сначала собери проект:  python scripts/make_release_zip.py")
    return zips[-1]


def _build_zip() -> None:
    print("==> Упаковываю проект (config + база лиц) в zip...")
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "make_release_zip.py"), run_name="__main__")


def _pump(chan, last_banner: str) -> None:
    """Печатает вывод сервера, пока команда не завершится."""
    timer = time.time()
    while True:
        if chan.recv_ready():
            timer = time.time()
            data = chan.recv(4096)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        if chan.exit_status_ready():
            while chan.recv_ready():
                data = chan.recv(4096)
                if data:
                    print(data.decode("utf-8", errors="replace"), end="")
            break
        if time.time() - timer > 40 * 60:  # сборка docker может идти долго
            break
        time.sleep(0.2)
    code = chan.recv_exit_status()
    if code != 0:
        raise SystemExit(f"Команда на сервере завершилась с ошибкой (код {code})")
    print(f"\n[{last_banner} OK]")


def deploy(host: str, user: str, password: str, port: int) -> None:
    import paramiko

    _build_zip()
    zip_path = _latest_zip()

    print(f"==> Подключаюсь к {user}@{host}:{port} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, password=password,
                timeout=30, look_for_keys=False, allow_agent=False)

    try:
        print("==> Заливаю zip на сервер...")
        sftp = ssh.open_sftp()
        remote_zip = f"/root/{zip_path.name}"
        sftp.put(str(zip_path), remote_zip)
        sftp.close()

        print("==> Разворачиваю (Docker, ~5-10 мин)...")
        script = (ROOT / "scripts" / "deploy_server.sh").read_text(encoding="utf-8")
        chan = ssh.get_transport().open_session()
        chan.settimeout(45 * 60)
        chan.exec_command("bash -s")
        chan.sendall(script.encode("utf-8"))
        chan.shutdown_write()
        _pump(chan, "деплой")
    finally:
        ssh.close()

    print("\nВсё готово! Проверка:  ssh root@HOST 'docker ps'")
    print("Логи:                ssh root@HOST 'docker logs -f ml_camera'")


def main() -> int:
    ap = argparse.ArgumentParser(description="Авто-деплой MLCamera на VPS")
    ap.add_argument("--host", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--port", type=int, default=22)
    args = ap.parse_args()

    if args.host and args.password:
        deploy(args.host, args.user or "root", args.password, args.port)
        return 0

    if not DEPLOY_FILE.exists():
        DEPLOY_FILE.write_text((DEPLOY_EXAMPLE.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"Создал {DEPLOY_FILE}. Впиши туда IP и пароль сервера и запусти ещё раз.")
        return 1
    data = yaml.safe_load(DEPLOY_FILE.read_text(encoding="utf-8")) or {}
    host = data.get("host") or args.host
    user = data.get("user") or args.user or "root"
    password = data.get("password")
    port = int(data.get("port") or args.port)
    if not host or not password:
        print("В pc/deploy.yaml не заполнены host/password.")
        return 1
    deploy(host, user, password, port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())