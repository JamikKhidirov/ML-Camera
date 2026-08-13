# Образ для VPS 300₽/мес: 1-2 CPU, ~1-2 ГБ RAM — лёгкая сборка
FROM python:3.10-slim

ENV TZ=Europe/Moscow \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

# Системные зависимости: OpenCV (libgl), tesseract для номеров
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Хотим удобные пути: скрипты и src внутри контейнера
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY config ./config

# Монтируем data с хоста (датасеты, БД, события, модели)
VOLUME ["/app/data"]

# Приложение: все включённые камеры
CMD ["python", "-m", "src.main"]