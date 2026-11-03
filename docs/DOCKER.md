# MLCamera в Docker

Контейнер — это «серверная» сборка MLCamera: камера + веб-панель + база +
уведомления. Внутри Linux-образ с OpenCV, ffmpeg, Tesseract (русский),
InsightFace и YOLO. Всё, что накопилось (фото, события, база лиц, модели),
лежит **на хосте** в `./data`, поэтому пересборка контейнера ничего не стирает.

> Коротко: `copy .env.example .env` → вписать RTSP → `docker compose up -d --build`
> → открыть `http://localhost:8088`.

---

## 1. Что понадобится

| Что | Зачем |
|---|---|
| Docker Engine 24+ или Docker Desktop | сам контейнер |
| Docker Compose v2 (`docker compose`) | запуск одной командой |
| 2+ ГБ RAM, 2 vCPU | 1–2 камеры на разумном FPS |
| RTSP/MJPEG-поток камеры | **веб-камеру ноутбука в Linux-контейнер не пробросить** (см. ниже) |

Проверить, что Docker жив:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Если `hello-world` не запускается — сначала запусти Docker Desktop (Windows)
или `sudo systemctl start docker` (Linux). Без запущенного демона сборка
падает с `failed to connect to the docker API`.

> **Windows:** Docker Desktop использует WSL 2. Если `wsl --status` пишет
> «WSL 2 не найден» или `wsl --update` просит права администратора —
> открой PowerShell **от имени администратора**, выполни `wsl --update`,
> затем перезагрузись. После этого запусти Docker Desktop и дождись
> статуса **Engine running**.

---

## 2. Быстрый старт

```bash
# 1. переменные окружения
cp .env.example .env          # Windows: copy .env.example .env
# открой .env и укажи RTSP-адрес камеры

# 2. сборка и запуск в фоне
docker compose up -d --build

# 3. смотрим, что происходит
docker compose ps
docker compose logs -f camera
```

Панель: **http://localhost:8088** (или `http://IP-СЕРВЕРА:8088`).
Если задан `MLCAMERA_WEB_TOKEN`, добавь `?token=...` или введи пароль в форме.

Остановить / запустить снова:

```bash
docker compose stop            # пауза, контейнер остаётся
docker compose start
docker compose down            # удалить контейнер (данные в ./data остаются)
```

---

## 3. Два способа задать камеру

### Вариант А — через `.env` (одна камера, быстрее всего)

```dotenv
MLCAMERA_CAMERA_SOURCE=rtsp://admin:pass@192.168.1.100:554/stream1
MLCAMERA_CAMERA_NAME=Ворота
MLCAMERA_CAMERA_FPS=30
```

`MLCAMERA_CAMERA_SOURCE` **заменяет весь список камер** из YAML одной камерой.
Подходит для типового сервера с одной камерой.

### Вариант Б — через `config/config.yaml` (несколько камер и тонкая настройка)

Оставь `MLCAMERA_CAMERA_SOURCE` пустым и опиши камеры в YAML — файл
примонтирован в контейнер как `./config:/app/config`:

```yaml
cameras:
  - name: "Ворота"
    source: "rtsp://admin:pass@192.168.1.100:554/stream1"
    enabled: true
  - name: "Двор"
    source: "rtsp://admin:pass@192.168.1.101:554/stream1"
    enabled: true
```

После правки YAML перезапусти контейнер:

```bash
docker compose restart camera
```

Полный список полей — в [`docs/SETUP.md`](SETUP.md) и
[`config/config.yaml`](../config/config.yaml).

---

## 4. Переменные окружения

Читаются в `src/config.py` (`apply_env_overrides`) при каждом старте.

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `MLCAMERA_PORT` | `8088` | **порт на хосте** (проброс `8088` из контейнера) |
| `MLCAMERA_WEB_HOST` | `0.0.0.0` | адрес панели внутри контейнера |
| `MLCAMERA_WEB_PORT` | `8088` | порт панели внутри контейнера (менять не нужно) |
| `MLCAMERA_WEB_TOKEN` | пусто | пароль панели; пусто — без пароля |
| `MLCAMERA_OPEN_BROWSER` | `0` | в контейнере браузера нет, всегда `0` |
| `MLCAMERA_CAMERA_SOURCE` | пусто | RTSP/HTTP-адрес **одной** камеры |
| `MLCAMERA_CAMERA_NAME` | `Камера` | имя этой камеры (папка событий) |
| `MLCAMERA_CAMERA_FPS` | `30` | FPS камеры (`0` — не ограничивать) |
| `MLCAMERA_FALLBACK_WEBCAM` | `0` | в контейнере веб-камеры нет, оставь `0` |
| `MLCAMERA_CONFIG` | `/app/config/config.yaml` | путь к конфигу внутри образа |
| `TZ` | `Europe/Moscow` | часовой пояс (время событий) |
| `OMP_NUM_THREADS` | `2` | потоки ONNX; больше — не всегда быстрее |

> Порт меняется **только** через `MLCAMERA_PORT`. Если поменять
> `MLCAMERA_WEB_PORT`, проброс `8088:8088` перестанет совпадать.

---

## 5. Что примонтировано и где данные

```yaml
volumes:
  - ./data:/app/data       # база, события, модели, датасеты, логи
  - ./config:/app/config   # единственный config.yaml
```

| Что | Где на хосте |
|---|---|
| База SQLite | `data/db/ml_camera.db` |
| События и фото | `data/events/<камера>/` |
| Модели YOLO/InsightFace | `data/models/` |
| Датасеты лиц и машин | `data/datasets/` |
| Логи | `data/logs/camera.log` |
| Конфиг | `config/config.yaml` |

`data/` в образ **не попадает** (`.dockerignore`) — только монтируется.
Бэкап делается обычным копированием папки:

```bash
tar czf ml_camera_backup_$(date +%F).tar.gz data config
```

На Windows проще остановить контейнер и скопировать `data` и `config`
через Проводник.

---

## 6. Healthcheck и статус

В образе есть `curl`, а у сервиса — проверка:

```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8088/health"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 90s
```

`/health` **не требует токена** — это нужно Docker и балансировщикам.
Смотреть статус:

```bash
docker compose ps                    # STATE: running (healthy)
docker inspect --format '{{.State.Health.Status}}' ml_camera
curl -fsS http://localhost:8088/health
```

`start_period: 90s` — первый запуск качает YOLO и грузит InsightFace, поэтому
контейнер может быть `starting` до полутора минут. Это нормально.

---

## 7. Веб-камера и GPU — ограничения

**Веб-камера ноутбука.** Docker Desktop (Windows/macOS) работает в
виртуальной Linux-машине и USB-камеру хоста не видит. В `.env` оставь
`MLCAMERA_FALLBACK_WEBCAM=0`, иначе в логах будут ошибки открытия `0`.
Для проверки камеры без Docker используй `camera.bat` / `python main.py --view`.

**Linux-хост** умеет пробрасывать устройство:

```yaml
services:
  camera:
    devices:
      - /dev/video0:/dev/video0
    environment:
      - MLCAMERA_CAMERA_SOURCE=0
      - MLCAMERA_FALLBACK_WEBCAM=1
```

**GPU.** Образ собирается под CPU (`onnxruntime`); CUDA в него не входит.
На сервере без GPU это ожидаемый режим: `yolov8n` + `buffalo_s` + Tesseract.
Для GPU нужна отдельная сборка образа с `onnxruntime-gpu` и
`--gpus all` — это выходит за рамки текущего `Dockerfile`.

**IP-камера.** Лучший вариант для сервера: RTSP-поток по локальной сети или
через VPN. См. [`docs/VPS_CAMERAS.md`](VPS_CAMERAS.md).

---

## 8. Обновление проекта

```bash
git pull
docker compose up -d --build     # пересобрать и перезапустить
docker image prune -f            # убрать старые слои (необязательно)
```

Данные и конфиг не трогаются: они в `./data` и `./config`.

Только перезапуск после правки конфига:

```bash
docker compose restart camera
```

---

## 9. Если что-то не работает

| Симптом | Причина и решение |
|---|---|
| `failed to connect to the docker API` | Docker Desktop/демон не запущен |
| Docker Desktop не стартует, `wsl --status` без WSL 2 | PowerShell от админа: `wsl --update`, перезагрузка |
| `port is already allocated` | занят `8088`: поменяй `MLCAMERA_PORT=8090` в `.env` |
| Контейнер `unhealthy` | `docker compose logs --tail=100 camera`; чаще всего неверный RTSP или нет модели |
| `Камера не открывается` | проверь RTSP в VLC, логин/пароль, `enabled: true` |
| Панель не открывается с другого ПК | `MLCAMERA_WEB_HOST=0.0.0.0`, порт в firewall, правильный IP |
| Просит токен | `MLCAMERA_WEB_TOKEN` в `.env`; добавь `?token=...` |
| Все лица «Незнакомец» | база пустая: зарегистрируй людей (`python main.py --register`) и положи `data/db` на место |
| Высокий CPU | `MLCAMERA_CAMERA_FPS=15`, `detect_every_n_frames: 3`, `yolov8n` |
| Медленно качает модели | первый старт: YOLO и InsightFace скачиваются в `data/models` |
| Кириллица в логах «кракозябрами» | это вывод Windows-консоли; в `data/logs/camera.log` и панели текст нормальный |

Полезные команды диагностики:

```bash
docker compose logs -f --tail=100 camera
docker compose exec camera python -c "import cv2; print(cv2.__version__)"
docker compose exec camera python scripts/self_check.py
docker stats ml_camera
```

---

## 10. Проверка перед выкладкой

```bash
docker compose config          # синтаксис compose и подстановка .env
docker compose build           # сборка образа
docker compose up -d           # запуск
curl -fsS http://localhost:8088/health
```

Полный чек-лист сервера — в [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).
Автоматический разворот на чистом VPS — `scripts/deploy_server.sh`
(он сам ставит Docker, распаковывает архив и поднимает
`docker-compose.prod.yml`).
