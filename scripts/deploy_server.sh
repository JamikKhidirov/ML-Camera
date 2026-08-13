#!/usr/bin/env bash
# =====================================================================
#  deploy_server.sh — разворот MLCamera на сервере ОДНОЙ командой.
#
#  У себя на компьютере:
#    1) python scripts/make_release_zip.py
#    2) scp dist/ml_camera_*.zip root@IP_СЕРВЕРА:/root/
#    3) ssh root@IP_СЕРВЕРА 'bash -s' < scripts/deploy_server.sh
#    4) отредактируй /root/ml_camera/config/config.yaml
#       (IP камер, telegram, ntfy)  ->  nano config/config.yaml
#    5) docker compose -f docker-compose.prod.yml up -d --build
#       (или просто перезапустите службу, см. ниже)
#
#  Скрипт сам: ставит Docker, распаковывает, настраивает сервис systemd.
# =====================================================================
set -euo pipefail

echo "==> 1/5 Ставлю Docker (если нет)..."
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker || true
fi

echo "==> 2/5 Распаковываю проект..."
APP_DIR="$HOME/ml_camera"
mkdir -p "$APP_DIR/data/db" "$APP_DIR/data/models" "$APP_DIR/data/datasets" \
         "$APP_DIR/data/events" "$APP_DIR/data/logs"
ZIP=$(ls "$HOME"/ml_camera_*.zip 2>/dev/null | head -n1)
if [ -n "${ZIP:-}" ]; then
  unzip -o -q "$ZIP" -d "$APP_DIR"
  mv "$ZIP" "$HOME/.ml_camera_last.zip" 2>/dev/null || true
  echo "   распакован $ZIP"
fi

echo "==> 3/5 Готовлю docker-compose.prod.yml (образ собирается из Dockerfile)..."
cat > "$APP_DIR/docker-compose.prod.yml" <<YML
services:
  camera:
    build: .
    image: ml_camera:local
    container_name: ml_camera
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./config/config.yaml:/app/config/config.yaml:ro
    environment:
      - TZ=Europe/Moscow
      - OMP_NUM_THREADS=2
YML

echo "==> 4/5 Собираю и запускаю контейнер (первый раз ~5-10 мин)..."
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml up -d --build

echo "==> 5/5 Готово!"
echo "Проверка:  docker ps  /  docker logs -f ml_camera"
echo "Файлы конфига: $APP_DIR/config/config.yaml"
echo "События:      $APP_DIR/data/events"
echo "База:         $APP_DIR/data/db/ml_camera.db"