# SETUP — настройка под ключ

Пошаговое руководство: от покупки камеры до уведомлений на телефоне.

---

## 0. Что понадобится

| Вещь | Зачем | Примерная цена |
|---|---|---|
| IP-камера с RTSP и ИК | видео + ночной режим | от 1500 ₽ (Xiaomi, Hikvision, Dahua) |
| Компьютер / сервер | где крутится анализ | твой ПК для отладки, VPS 300 ₽ для постоянной работы |
| Telegram-аккаунт | уведомления | бесплатно |
| Android-телефон | push-уведомления | у тебя уже есть |
| Wi-Fi/провод до камер | передача видео | — |

> Если камеры пока нет — можно отлаживать на видеофайле:
> положи `data/test_video.mp4` и в конфиге `source: "data/test_video.mp4"`.

---

## 1. Установка зависимостей (Windows/компьютер)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Для распознавания номеров есть три варианта:

- **FastALPR (рекомендуется, нейросеть)** — точнее всего, ставится скриптом:
  ```bash
  python scripts/install_alpr.py
  ```
  После установки поставь в `config/config.yaml` (или в панели → Настройки)
  `plate_engine: "fastalpr"`. По умолчанию там `tesseract` — он работает
  без дополнительных моделей.
- **tesseract (лёгкий, для слабого VPS)** — нужен tesseract в системе:

  - **Windows**: скачай установщик https://github.com/UB-Mannheim/tesseract/wiki
    (ставь галочку на Russian language data), затем:
    ```python
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```
    (эту строчку добавь в `src/plate_recognizer.py` после импорта)
  - **Linux/сервер**: `apt install -y tesseract-ocr tesseract-ocr-rus` (в Docker уже есть)

- **easyocr (запасной)** — `plate_engine: "easyocr"` + `pip install easyocr`.
  Тянет torch (~2 ГБ) и медленный на CPU.

> Полный список моделей, их размеры, лицензии и ограничения: `docs/MODELS.md`.

### Животные и оставленные вещи

Отдельно ставить ничего не нужно — используются классы готовой YOLO.
В `config/config.yaml` уже включено:

```yaml
objects:
  animals:
    enabled: true
    classes: ["dog", "cat", "bird"]
  packages:
    enabled: true
    classes: ["backpack", "suitcase", "handbag"]
    stationary_seconds: 45
```

Подробно: [`OBJECTS.md`](OBJECTS.md).

### Своя модель марок для РФ

Готовая модель знает только 49 марок США. Если нужны Lada, Kia, Haval и
другие — обучите свою на фото с камеры:

```bash
pip install torch torchvision
python scripts/collect_car_images.py
# разложить по data/datasets/cars/train/<Марка>/
python scripts/train_brand_classifier.py --dry-run
python scripts/train_brand_classifier.py --epochs 15
```

Подробно: [`BRAND_RU.md`](BRAND_RU.md).

---

## 2. Подключение камер

**Сначала проверь без камеры:** в `config/config.yaml` по умолчанию уже
включена веб-камера ноутбука:

```yaml
cameras:
  - name: "Ноутбук (веб-камера)"
    source: 0
    enabled: true
```

Запусти `python -m src.main` — в панели появится живой поток и «чертёж лица».
Если IP-камера не открылась, программа сама переключится на веб-камеру
(`camera.fallback_webcam: true`).

Своя IP-камера добавляется в тот же список `cameras`:

```yaml
cameras:
  - name: "Камера-1 (двор)"
    source: "rtsp://ЛОГИН:ПАРОЛЬ@IP_КАМЕРЫ:554/stream1"
    enabled: true
  - name: "Камера-2 (ворота)"
    source: "rtsp://ЛОГИН:ПАРОЛЬ@IP_КАМЕРЫ:554/stream1"
    enabled: true
```

Веб-камере при этом поставь `enabled: false`, чтобы не занимать устройство.

Как узнать RTSP-адрес своей камеры:

1. Установи фирменное приложение камеры (Mi Home, Hik-Connect, ...) и найди
   **IP-адрес** камеры в локальной сети (обычно `192.168.1.xxx`).
2. Включи на камере **RTSP** (некоторые требуют "Учётная запись для стороннего доступа").
3. Подставь логин/пароль (администратора камеры).
4. Проверь, что видео идёт: в VLC → «Открыть сетевой поток» → вставь RTSP-ссылку.

Типовые пути RTSP (если `stream1` не работает, попробуй другие):

| Камера | RTSP |
|---|---|
| Xiaomi / большинство китайских | `rtsp://user:pass@IP:554/stream1` |
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| TP-Link | `rtsp://user:pass@IP:554/stream1` |

> Чтобы камеры были доступны с сервера, у тебя на роутере нужен доступ:
> либо камеры и сервер в одной сети, либо проброс портов / VPN (например,
> WireGuard на роутере). На VPS можно даже не пробрасывать порты наружу —
> достаточно чтобы сервер мог «дотянуться» до камер.

Проверь подключение:
```bash
python -m src.main --once
```
(проанализирует 30 кадров и завершится).

---

## 3. Телеграм-уведомления

1. Открой в Telegram бота **@BotFather**, отправь `/newbot`, придумай имя —
   получишь **токен** вида `123456789:AA...`.
2. Открой **@userinfobot**, нажми Start — пришлёт твой **chat_id** (число).
3. Впиши в `config/config.yaml`:
   ```yaml
   notify:
     telegram:
       enabled: true
       bot_token: "123456789:AA..."
       chat_id: "123456789"
   ```
4. Проверь: `python -m src.main --self-test` — в телеграм должно прийти
   тестовое сообщение.

---

## 4. Push-уведомления на Android (ntfy)

Способ бесплатный, без регистрации, без Google:

1. Установи приложение **ntfy** из Play Market (или F-Droid).
2. Придумай **секретный topic** (никому не сообщай), например
   `my_camera_7f3a91` — и впиши в конфиг:
   ```yaml
   push:
     enabled: true
     provider: "ntfy"
     server: "https://ntfy.sh"
     topic: "my_camera_7f3a91"
   ```
3. В приложении ntfy подпишись на этот же topic.
4. Проверь: `python -m src.main --self-test`.

> Придёт и в Telegram, и пушем на телефон — дублирование безопасно.

---

## 5. Сбор датасета и регистрация семьи

Сколько фото и как их снять — в **docs/DATASET.md** (там таблица: мама/папа/ты/2 сестры).

Коротко:
```bash
python scripts/make_dirs.py          # создать папки
# кладёшь фото в data/datasets/raw/<имя>/ (мама, папа, я, сестра1, сестра2)
python scripts/collect_faces.py      # вырезает лица
python scripts/register_faces.py     # считает эмбеддинги
```

---

## 6. Запуск

```bash
# Один конфиг (config/config.yaml): веб-камера + панель + браузер
python -m src.main

# С превью и записью
python -m src.main --preview --recording

# Настройки в окне Tkinter
python scripts/settings.py
```

Полезные флаги:

```bash
python -m src.main --once              # пробный прогон (30 кадров)
python -m src.main --cameras 0 2       # только камеры 0 и 2
python -m src.main --self-test         # проверить уведомления
python -m src.main --preview           # живое окно с рамками (вкл. ui.preview_enabled)
python -m src.main --recording         # записывать клипы событий (вкл. recording.enabled)
python -m src.main --web               # поднять веб-панель вместе с камерой
python -m src.main --no-web            # без панели
python scripts/show_events.py          # что накопилось в БД
```

Снимки событий — в `data/events/<имя камеры>/<дата>/`, клипы — в
`data/events/<имя камеры>/clips/`, события — в `data/db/ml_camera.db`.

### 6.1. Веб-панель истории и управления

В `config/config.yaml` панель стартует вместе с камерой:

**http://127.0.0.1:8088**

Вкладки: **«Камеры»** (live-поток, рамки, «чертёж лица», статус, fps,
старт/стоп, снимок), **«История»** (события, фильтры, график) и
**«Система»** (версия, БД, диск, возможности, таблица камер).

Отдельно (только просмотр, без кнопок управления):

```bash
python scripts/web_panel.py --config config/config.yaml --open
```

> Кнопки камер работают только когда панель запущена вместе с камерой:
> `python main.py --web` или `web.autostart: true`. Отдельный процесс
> читает статусы из `data/db/camera_status.json` и не может управлять
> потоком.

Оверлей настраивается в `web`:

```yaml
web:
  overlay: true       # рамки объектов
  face_mesh: true     # 106 точек лица InsightFace
```

Для доступа с телефона в конфиге:

```yaml
web:
  host: "0.0.0.0"
  token: "длинный-пароль"
  control: true          # кнопки камер
```

и открыть `http://IP-КОМПЬЮТЕРА:8088/?token=длинный-пароль`.
Без токена и HTTPS не выставляйте панель в интернет. Полное руководство:
[`WEB.md`](WEB.md).

## 6.2. Если у тебя мощный ПК

Единый `config/config.yaml` уже работает и на ноутбуке, и на ПК. На мощной
машине можно поднять качество — в веб-панели («🛠 Настройки») или в файле:

- **плавность**: `camera.fps: 30`, `web.stream_fps: 15`;
- **точность ИИ**: `analysis.detect_every_n_frames: 2`;
- **большие модели**: `models.yolov8: "data/models/yolov8m.pt"`,
  `face.model_name: "buffalo_l"` (скачай модель заново
  `python scripts/download_models.py`);
- **запись**: `recording.enabled: true` (клипы вокруг событий);
- **животные и оставленные вещи**: `objects.enabled: true`;
- **веб-панель истории и управления** на http://127.0.0.1:8088;
- **авто-обучение** незнакомцев (`data/datasets/auto_unknown/`).

Установка доп. пакетов для ПК:

```bash
pip install -r requirements-pc.txt
```

> Видеокарта AMD (RX 5700 XT): ускорение лица через **DirectML**.
> По умолчанию не ставится (см. комментарий в requirements-pc.txt),
> при этом всё работает на CPU — Ryzen 5600 тянет с запасом.

---

## 7. Частые проблемы

| Проблема | Решение |
|---|---|
| `Не удалось открыть источник` | проверь RTSP-ссылку в VLC; включи RTSP в приложении камеры |
| Кадры тёмные ночью | включи на камере ИК («Авто»); проверь `night_enabled: true` |
| Лица не распознаются | добавь фото, пересобери датасет; снизь `similarity_threshold` до 0.40 |
| Номер не читается | проверь tesseract (`scripts/download_models.py`); поставь камеру под лучший угол; уменьши `detect_every_n_frames` |
| Много повторов уведомлений | подними `cooldown_minutes` |
| Всё тормозит на VPS | подними `detect_every_n_frames` до 10, `frame_scale` до 0.5; отключи `brand_enabled` |
| Не приходят уведомления о животных | проверь `objects.animals.enabled` и `conf_threshold` |
| Вещь не считается оставленной | подними `objects.packages.stationary_seconds` или проверь, не двигается ли камера |
| Веб-панель не открывается | включи `web.enabled`, проверь порт 8088 и `host` |
| Кнопки камер серые | панель запущена отдельно — нужен `python main.py --web` |
| Веб-панель просит токен | открой `?token=...` из `web.token` или задай пустой токен для localhost |

Полная карта настройки есть в `config/config.yaml` — там всё прокомментировано.