#!/usr/bin/env bash
# =====================================================================
#  deploy_server.sh — разворот MLCamera на сервере ОДНОЙ командой.
#
#  У себя на компьютере (ВСЁ ДЕЛАЕТСЯ АВТОМАТИЧЕСКИ):
#    python pc/deploy_vps.py
#  (этот скрипт запускается на сервере той же командой)
#
#  Скрипт сам: ставит Docker, распаковывает проект (вместе с твоим
#  конфигом и базой лиц), собирает и запускает контейнер.
#  После — только проверка: docker ps / docker logs -f ml_camera
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
    command: ["python", "-m", "src.main"]
    ports:
      - "\${MLCAMERA_PORT:-8088}:8088"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - TZ=\${TZ:-Europe/Moscow}
      - OMP_NUM_THREADS=\${OMP_NUM_THREADS:-2}
      - MLCAMERA_CONFIG=/app/config/config.yaml
      - MLCAMERA_WEB_HOST=0.0.0.0
      - MLCAMERA_WEB_PORT=8088
      - MLCAMERA_OPEN_BROWSER=0
      - MLCAMERA_CAMERA_SOURCE=\${MLCAMERA_CAMERA_SOURCE:-}
      - MLCAMERA_CAMERA_NAME=\${MLCAMERA_CAMERA_NAME:-}
      - MLCAMERA_CAMERA_FPS=\${MLCAMERA_CAMERA_FPS:-}
      - MLCAMERA_FALLBACK_WEBCAM=\${MLCAMERA_FALLBACK_WEBCAM:-0}
      - MLCAMERA_WEB_TOKEN=\${MLCAMERA_WEB_TOKEN:-}
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8088/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 90s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
YML

echo "==> 4/5 Собираю и запускаю контейнер (первый раз ~5-10 мин)..."
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml up -d --build

echo "==> 5/5 Готово!"
echo "Проверка:  docker ps  /  docker logs -f ml_camera"
echo "Панель:    http://IP-СЕРВЕРА:8088  (порт и пароль — в .env)"
echo "Конфиг пришёл вместе с проектом (твой config/config.yaml)"
echo "Если надо поменять: $APP_DIR/config/config.yaml  ->  docker compose -f docker-compose.prod.yml up -d --build"
echo "События:      $APP_DIR/data/events"
echo "База:         $APP_DIR/data/db/ml_camera.db"