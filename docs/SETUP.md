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
  В `config/config.pc.yaml` уже стоит `plate_engine: "fastalpr"`.
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

---

## 2. Подключение камер

Все камеры прописываются в `config/config.yaml` в списке `cameras`:

```yaml
cameras:
  - name: "Камера-1 (двор)"
    source: "rtsp://ЛОГИН:ПАРОЛЬ@IP_КАМЕРЫ:554/stream1"
    enabled: true
  - name: "Камера-2 (ворота)"
    source: "rtsp://ЛОГИН:ПАРОЛЬ@IP_КАМЕРЫ:554/stream1"
    enabled: true
```

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
# Универсальный конфиг (config/config.yaml)
python -m src.main

# Мощная ПК-версия (Ryzen + RX 5700 XT): превью, клипы, большие модели
python -m src.main --config config/config.pc.yaml

# Лёгкая серверная версия (VPS 300 ₽)
python -m src.main --config config/config.vps.yaml
```

Полезные флаги:

```bash
python -m src.main --once              # пробный прогон (30 кадров)
python -m src.main --cameras 0 2       # только камеры 0 и 2
python -m src.main --self-test         # проверить уведомления
python -m src.main --preview           # живое окно с рамками (вкл. ui.preview_enabled)
python -m src.main --recording         # записывать клипы событий (вкл. recording.enabled)
python scripts/show_events.py          # что накопилось в БД
```

Снимки событий — в `data/events/<имя камеры>/<дата>/`, клипы — в
`data/events/<имя камеры>/clips/`, события — в `data/db/ml_camera.db`.

## 6.1. Версия для ПК (Windows 11, Ryzen, видеокарта)

ПК-конфиг `config/config.pc.yaml` включает:
- **живое окно предпросмотра** с рамками детекции (`--preview`);
- **запись видеоклипов** вокруг каждого события (лицо/машина/незнакомец);
- **детекцию движения** — анализ только когда что-то шевелится;
- **точные модели**: `yolov8m`, InsightFace `buffalo_l`, `easyocr` для номеров;
- **авто-обучение** незнакомцев (`data/datasets/auto_unknown/`).

Установка доп. пакетов для ПК-версии:

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

Полная карта настройки есть в `config/config.yaml` — там всё прокомментировано.