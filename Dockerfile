# syntax=docker/dockerfile:1
# Образ MLCamera для VPS и домашнего сервера.
# Сборка:  docker compose up -d --build
# Конфиг:  ./config/config.yaml (монтируется в контейнер)
# Данные:  ./data (БД, события, модели) — не теряются при перезапуске.

FROM python:3.11-slim

ENV TZ=Europe/Moscow \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MLCAMERA_CONFIG=/app/config/config.yaml

# Системные зависимости:
#   libgl1/libglib — OpenCV; libgomp1 — onnxruntime;
#   tesseract + rus — номера; ffmpeg — RTSP/видео;
#   build-essential/python3-dev — сборка insightface, если нет wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tesseract-ocr \
        tesseract-ocr-rus \
        ffmpeg \
        curl \
        ca-certificates \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — слой кэшируется и не пересобирается при правке кода.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Код приложения и единственный конфиг.
COPY main.py ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY config ./config

# Папки данных создаём заранее: примонтированный том может быть пустым.
RUN mkdir -p /app/data/db /app/data/events /app/data/models \
             /app/data/datasets /app/data/videos /app/data/logs \
             /app/data/snapshots

VOLUME ["/app/data"]
EXPOSE 8088

# /health отдаётся без токена и ничего не раскрывает — удобно для Docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${MLCAMERA_WEB_PORT:-8088}/health" || exit 1

# По умолчанию: камеры из config.yaml + веб-панель.
CMD ["python", "-m", "src.main"]
