"""Упаковка проекта в zip для ручной загрузки на сервер.

Создаёт dist/ml_camera_<дата>.zip со всем нужным для разворота на VPS:
  src/, scripts/, tests/, config/, docs/, Dockerfile,
  docker-compose.yml, docker-compose.prod.yml, requirements.txt

Запуск: python scripts/make_release_zip.py
Затем: залей zip на сервер и выполни scripts/deploy_server.sh
"""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = [
    "src", "scripts", "tests", "config", "docs",
    "Dockerfile", "docker-compose.yml",
    "requirements.txt", "requirements-pc.txt", "requirements-build.txt",
    "README.md", "main.py",
    # лица и база фотографий уедут на сервер вместе с проектом (если есть)
    "data/datasets", "data/db",
]


def main() -> int:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"ml_camera_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    added = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            p = ROOT / item
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts and ".pytest_cache" not in f.parts:
                        zf.write(f, f.relative_to(ROOT))
                        added += 1
            elif p.is_file():
                zf.write(p, p.relative_to(ROOT))
                added += 1
    print(f"Готово: {out}  ({added} файлов, {out.stat().st_size/1024:.0f} КБ)")
    print("\nДальше на сервере:")
    print("  1) scp dist/ml_camera_*.zip root@IP:/root/")
    print("  2) ssh root@IP 'bash -s' < scripts/deploy_server.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())