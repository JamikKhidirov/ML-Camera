<div align="center">
  <img src="docs/assets/banner.gif" alt="MLCamera — умная камера для семьи" width="100%">
</div>

<h1 align="center">MLCamera</h1>

<p align="center">
  <b>IP-камера, которая знает родных, читает номера, замечает животных и оставленные вещи, определяет марку машины и пишет в Telegram.</b><br>
  Всё считается локально: без облачных подписок и отправки видео на чужие серверы.
</p>

<p align="center">
  <a href="#-что-умеет">Возможности</a> ·
  <a href="#-как-это-работает">Как работает</a> ·
  <a href="#-камеры">Камеры</a> ·
  <a href="#-веб-панель">Веб-панель</a> ·
  <a href="#-животные-и-оставленные-вещи">Животные и вещи</a> ·
  <a href="#-модели">Модели</a> ·
  <a href="#-быстрый-старт">Быстрый старт</a> ·
  <a href="#-команды">Команды</a> ·
  <a href="#-документация">Документация</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Windows | Linux" src="https://img.shields.io/badge/Windows%20%7C%20Linux-0f172a?style=for-the-badge&logo=linux&logoColor=white">
  <img alt="5 neural models" src="https://img.shields.io/badge/neural%20models-5-42e8a0?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/tests-83%20passed-42e8a0?style=for-the-badge">
  <img alt="Version" src="https://img.shields.io/badge/version-v1.1.0-40d6ff?style=for-the-badge">
  <img alt="Local-first" src="https://img.shields.io/badge/privacy-local--first-966eff?style=for-the-badge">
</p>

---

## 🎯 Что это

MLCamera — домашняя камера, которая не просто пишет видео, а **понимает, что происходит**:

- узнаёт членов семьи по лицу и пишет «Алексей пришёл»;
- отдельно сообщает о незнакомцах и сохраняет их лица для дообучения;
- замечает машину, читает номер, определяет цвет и марку;
- сообщает о собаках/кошках/птицах и оставленных рюкзаках, сумках, чемоданах;
- работает с несколькими камерами сразу — у каждой свой поток и история;
- **из коробки** подключается к веб-камере ноутбука: живой поток, рамки объектов и «чертёж лица» прямо в браузере;
- если IP-камеры не открылись — автоматически переключается на веб-камеру;
- показывает всю историю событий в локальной веб-панели;
- не спамит: одна машина или человек = одно уведомление за проход;
- работает днём и ночью, на ПК и на дешёвом VPS;
- отправляет фото и текст в Telegram или на телефон через ntfy.

> [!NOTE]
> Животные, посылки, лица, номера и готовая модель марки — **без обучения**,
> всё скачивается скриптами. Обучать нужно только **свою модель марок**
> под российский рынок — и это опционально. См.
> [`docs/MODELS.md`](docs/MODELS.md) и [`docs/BRAND_RU.md`](docs/BRAND_RU.md).

---

## ✨ Что умеет

<p align="center">
  <img src="docs/assets/features.gif" alt="Возможности MLCamera" width="760">
</p>

| Возможность | Как работает |
|---|---|
| 👤 **Кто пришёл** | InsightFace строит эмбеддинг лица и сравнивает с базой семьи |
| 👾 **Незнакомец** | Отдельное уведомление + фото в `data/datasets/auto_unknown/` |
| 🚗 **Машина проехала** | YOLO находит автомобиль, трекер ведёт его между кадрами |
| 🔢 **Номер** | FastALPR: детектор YOLOv9 + OCR, голосование по нескольким кадрам |
| 🏷️ **Марка** | ConvNeXt-Tiny, 49 марок, один анализ на машину по лучшему кадру |
| 🇷🇺 **Свои марки** | Обучение на своих фото: Lada, Kia, Haval и любые другие |
| 🎨 **Цвет** | HSV-гистограмма по центральной зоне кузова |
| 🐾 **Животные** | YOLO-классы `dog`, `cat`, `bird` — одно уведомление на трек |
| 📦 **Оставленные вещи** | `backpack`/`suitcase`/`handbag` + проверка неподвижности |
| 🖥️ **Веб-панель** | История событий, фильтры, поиск, фото и управление камерами — без новых зависимостей |
| 🛠 **Настройки из коробки** | FPS камеры/потока/записи и все параметры — в панели или в Tkinter, один `config.yaml` |
| 📷 **Несколько камер** | Список в конфиге: имя, поток, `enabled` — у каждой свои трекеры, папки и уведомления |
| 🎛️ **Управление камерой** | `--cameras`, `--preview`, `--once`, ночной режим и запись — без правки кода |
| 🌙 **Ночное видение** | ИК-режим камеры + усиление тёмных кадров и усреднение |
| ⏱️ **Без спама** | CentroidTracker: 2 минуты перед камерой = 1 уведомление |
| 🟢/🔴 **Вошёл / вышел** | Пересечение линии `tracking.entry_line_y` |
| ⭐ **Свой номер** | Белый список `car.known_plates` → «Своя машина» |
| 🌅 **Утренний дайджест** | Сводка за сутки в 09:00: люди, машины, животные, посылки |
| 🔧 **Мультиразрешение** | Нет движения → анализ мелким кадром, экономия CPU |
| 🎬 **Запись клипов** | Короткие видео вокруг события или непрерывная запись |
| 📨 **Telegram + ntfy** | Фото, текст и подписи в несколько чатов одновременно |

---

## 🧠 Как это работает

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="Схема обработки кадра MLCamera" width="100%">
</p>

1. **Камера** отдаёт RTSP-поток, веб-камеру или видеофайл.
2. **Детектор движения** решает, стоит ли запускать тяжёлый анализ. Если в кадре тихо — CPU почти не тратится.
3. **YOLO** за один прогон находит людей, машины, животных и вещи.
4. Дальше кадр расходится по веткам:
   - **лица** → InsightFace → поиск по базе семьи → событие;
   - **машины** → трекер → номер (ALPR), цвет (HSV), марка (CNN);
   - **животные** → трекер → одно уведомление на появление;
   - **вещи** → трекер → проверка неподвижности → «оставлена».
5. Результат сохраняется в **SQLite + фото**, владельцу уходит **уведомление**
   в Telegram/ntfy, а история открывается в **веб-панели**.
6. Если включена запись — рядом с событием сохраняется короткий **видеоклип**.

> Главная оптимизация: марка машины считается **один раз на трек**, номер —
> голосованием по нескольким кадрам, а животные и вещи фильтруются из того же
> результата YOLO. Поэтому камера не захлёбывается на CPU.

---

## 📷 Камеры

Все камеры живут в одном списке `cameras:` — добавлять, выключать и менять
их можно без правки кода. Каждая камера обрабатывается отдельно: у неё свои
трекеры, папка событий, записи и уведомления.

### Подключить камеру

**По умолчанию** `config/config.yaml` уже смотрит в веб-камеру ноутбука:

```yaml
cameras:
  - name: "Ноутбук (веб-камера)"
    source: 0
    enabled: true
```

Просто запусти `python main.py` — живой поток появится в панели на
`http://127.0.0.1:8088`. Своя IP-камера добавляется так:

```yaml
cameras:
  - name: "Двор"                    # имя в уведомлениях, папках и панели
    source: "rtsp://admin:pass@192.168.1.100:554/stream1"
    enabled: true
  - name: "Ворота"
    source: "rtsp://admin:pass@192.168.1.101:554/stream1"
    enabled: false                  # камера описана, но не запускается
```

Если ни одна камера из списка не открылась, сработает
`camera.fallback_webcam: true` — программа сама подключит веб-камеру `0`.
Отключить: `fallback_webcam: false`.

| `source` | Что это |
|---|---|
| `rtsp://...` | IP-камера (Hikvision, TP-Link, Xiaomi и т.п.) |
| `http://.../video` | MJPEG-поток камеры или телефона |
| `0`, `1` | веб-камера или телефон, подключённый как камера |
| `data/test_video.mp4` | видеофайл — отладка без камеры |

> `name` лучше не менять после первых событий: это имя папки
> `data/events/<имя>/` и подпись камеры в базе и веб-панели.

### Включить только часть камер

```bash
python main.py --cameras 0 2     # только первая и третья из списка
```

Либо просто поставь `enabled: false` у ненужных: камера не открывается,
а её история остаётся в базе.

### Съёмка и ночной режим

```yaml
camera:
  width: 1280
  height: 720
  night_enabled: true               # ИК-кадры: усиление + усреднение
  night_brightness_threshold: 60    # ниже этой яркости включается ночной режим
  frame_averaging_night: true
  frame_averaging_count: 3
```

### Экономия CPU

```yaml
analysis:
  detect_every_n_frames: 1          # 3–5 на слабом CPU
  frame_scale: 1.0                  # 0.5–0.75 — кадр меньше, YOLO быстрее
  multi_resolution: true            # без движения анализируется мелкий кадр
  small_frame_scale: 0.5

motion:
  enabled: true
  threshold: 1500                   # меньше — чувствительнее
  scale: 0.25
  force_interval_frames: 300        # раз в N кадров анализ даже без движения
```

### Превью и запись

```yaml
ui:
  preview_enabled: true             # окно с рамками: люди, машины, звери, вещи
  window_title: "MLCamera — {cam}"

recording:
  enabled: true
  mode: "continuous"                # "events" — только клипы вокруг событий
  output_dir: "data/videos"
  rotate_seconds: 300               # новый файл каждые 5 минут
  fps: 12
```

| Задача | Что менять |
|---|---|
| Камера не запускается | `cameras[].enabled: true`, проверить адрес и логин/пароль |
| Нужна только одна камера | `python main.py --cameras 0` |
| Поток обрывается | переподключение автоматическое через 5 с; проверь RTSP |
| Тормозит на VPS | `detect_every_n_frames: 5`, `frame_scale: 0.5`, `brand_enabled: false` |
| Не открывается окно | `ui.preview_enabled: true` или флаг `--preview` |
| Не пишется видео | `recording.enabled: true` или флаг `--recording` |

Проверить камеру, не оставляя её работать: `python main.py --once` —
обработает ~30 кадров и выйдет.

### Управление камерой из веб-панели

Если панель запущена вместе с камерой (`main.py`), во вкладке **«Камеры»**
есть статус каждой камеры (`running`/`stopped`/`error`), fps, число кадров
и кнопки:

| Кнопка | Что делает |
|---|---|
| ▶ Запустить | поднимает поток остановленной камеры без перезапуска процесса |
| ■ Остановить | мягко завершает поток; модели остаются в памяти |
| 📸 Снимок | показывает текущий кадр камеры в браузере |

Остановка всех камер **не завершает** `main.py`, пока работает панель —
камеру можно запустить снова. Если панель запущена отдельным процессом
(`scripts/web_panel.py`), она показывает только статусы из
`data/db/camera_status.json`, а кнопки отключены.

```yaml
web:
  enabled: true
  autostart: true
  control: true        # false — панель только смотрит, без кнопок
```

Подробнее: [`docs/WEB.md`](docs/WEB.md#get-apicameras).

---

## 🖥️ Веб-панель

Локальная история событий: лица, машины, животные и оставленные вещи с
фото, фильтрами и поиском — плюс управление камерами. Новых зависимостей
нет — обычный `http.server` из стандартной библиотеки Python.

**Открыть: http://127.0.0.1:8088**

| Как запустить | Команда |
|---|---|
| Вместе с камерой | `web.enabled: true`, `web.autostart: true` → `python main.py` |
| Принудительно с камерой | `python main.py --web` |
| Отдельным процессом | `python scripts/web_panel.py --config config/config.yaml --open` |

Что внутри:

- **live-поток** с каждой камеры (MJPEG) и снимок кадра;
- рамки машин/животных/вещей и **«чертёж лица»** по 106 точкам InsightFace;
- карточки событий с фото и вырезкой номера;
- фильтры по типу (лица / машины / животные / вещи), камере и периоду;
- поиск по имени, номеру, марке, цвету и классу;
- счётчики за сутки, график активности, группировка по датам;
- автообновление каждые 5 секунд;
- тёмная адаптивная тема, горячие клавиши `1/2/3`;
- вкладка **«Камеры»**: статус, fps, разрешение, аптайм, события за 24 ч;
- старт/стоп камеры, снимок и скачивание JPEG — без перезапуска процесса;
- бейдж `1/2` на вкладке — сколько камер сейчас работает.

Служебные адреса:

- поток: `/api/cameras/stream?index=0&fps=10&quality=70`;
- снимок: `/api/cameras/snapshot?index=0&quality=90&download=1`;
- статусы: `/api/cameras`, система: `/api/system`, график: `/api/timeline`.

> Кнопки управления доступны только когда панель живёт **в процессе
> камеры** (`main.py` / `main.py --web`). Отдельный `scripts/web_panel.py`
> показывает статусы, но не управляет: два процесса не должны делить один
> поток. POST-запросы защищены заголовком `X-MLCamera` и проверкой
> `Origin`; `web.control: false` отключает управление совсем.

Доступ с телефона:

```yaml
web:
  host: "0.0.0.0"
  token: "длинный-пароль"
```

Открыть `http://IP-КОМПЬЮТЕРА:8088/?token=длинный-пароль` — сервер
запомнит cookie. Без токена и HTTPS панель в интернет не выставляй.
API, безопасность и systemd-юнит: [`docs/WEB.md`](docs/WEB.md).

---

## 🐾 Животные и оставленные вещи

Работают на **той же YOLO**, что и люди с машинами: отдельную модель
скачивать не нужно, лишнего прогона сети нет.

| Событие | Классы COCO | Когда уведомление |
|---|---|---|
| Животное | `dog`, `cat`, `bird` | после `min_frames` кадров, один раз на трек |
| Оставленная вещь | `backpack`, `suitcase`, `handbag` | вещь не двигается `stationary_seconds` |

```yaml
objects:
  enabled: true
  animals:
    enabled: true
    classes: ["dog", "cat", "bird"]
    conf_threshold: 0.35
    min_frames: 2
    cooldown_minutes: 30
  packages:
    enabled: true
    classes: ["backpack", "suitcase", "handbag"]
    conf_threshold: 0.40
    min_frames: 3
    stationary_seconds: 45       # сколько секунд без движения = «оставлена»
    stationary_move_ratio: 0.15  # допустимое смещение от размера вещи
    cooldown_minutes: 120
```

«Оставлена» = центр трека не смещается больше, чем
`max(8 px, 15% размера вещи)`, в течение `stationary_seconds`.
Человек идёт с рюкзаком — таймер сбрасывается; снял и ушёл — рюкзак
даёт событие.

> Отдельного класса `package` в COCO нет: коробка может не детектироваться,
> а сумка на стуле — дать ложное срабатывание. Настройка и диагностика:
> [`docs/OBJECTS.md`](docs/OBJECTS.md).

---

## 🤖 Модели

<p align="center">
  <img src="docs/assets/models.svg" alt="Нейросети внутри MLCamera" width="100%">
</p>

| Задача | Модель | Размер | Лицензия |
|---|---|---|---|
| Люди, машины, животные, вещи | YOLOv8m (ПК) / YOLOv8n (VPS) | 6–50 МБ | AGPL-3.0 |
| Лица и эмбеддинги | InsightFace `buffalo_s` / `buffalo_l` | 158 МБ | код MIT, веса — research |
| Марка автомобиля | ConvNeXt-Tiny, Stanford Cars, 49 марок | 106 МБ | Apache-2.0 |
| Своя марка (РФ) | MobileNetV3 / ConvNeXt, обучение на ваших фото | 10–110 МБ | ваша |
| Поиск номера | YOLOv9-t 384 license-plate end2end | 7.4 МБ | MIT |
| Чтение номера | European plates MobileViT v2 | 14.5 МБ | MIT |

**Команды для скачивания:**

```bash
python scripts/download_models.py --config config/config.yaml
python scripts/download_car_brand_model.py   # только марка, если нужно отдельно
python scripts/install_alpr.py               # детектор + OCR номеров
```

**Своя модель марок для РФ:**

Датасет — обычные папки, по одной на марку:

```text
data/datasets/cars/
├── train/
│   ├── Lada/       150 фото
│   ├── Kia/        120 фото
│   └── Haval/       90 фото
├── val/            # необязательно; иначе 20% уйдёт из train
└── inbox/          # сюда собирает collect_car_images.py
```

```bash
python scripts/collect_car_images.py                 # собрать фото машин из событий
# разложить по data/datasets/cars/train/<Марка>/
python scripts/train_brand_classifier.py --dry-run
python scripts/train_brand_classifier.py --epochs 15
```

После обучения появляются `car_brand.onnx`, `car_brand_labels.json`,
`car_brand_meta.json` и backup старой модели. Подключается в конфиге:

```yaml
car:
  brand_enabled: true
models:
  brand_onnx: "data/models/car_brand_ru.onnx"
  brand_labels: "data/models/car_brand_labels.json"
```

Подробно: [`docs/BRAND_RU.md`](docs/BRAND_RU.md).

**Что важно знать:**

- Марка определяется по лучшей вырезке за трек, а не каждый кадр — так точнее и дешевле.
- Номер собирается голосованием, а не по одному случайному кадру.
- `brand_min_conf` не даёт модели выдать случайную марку: ниже порога будет `unknown`.
- Животные и вещи используют классы COCO готовой YOLO — отдельная модель не нужна.
- Полные пути, кэши и ограничения — в [`docs/MODELS.md`](docs/MODELS.md).

---

## 🚀 Быстрый старт

**Один конфиг — `config/config.yaml`.** Ничего копировать и выбирать не нужно:
и камера, и веб-панель, и Tkinter-настройки читают и пишут этот файл.

### Вариант 1 — Windows, двойной клик

1. **`pc\setup.bat`** — один раз: создаст окружение, поставит библиотеки,
   скачает YOLO, лица, марку и ALPR.
2. **`pc\run_pc.bat`** (или `start_pc.bat`) — запускает камеру и веб-панель.
3. Через пару секунд браузер откроется сам: **http://127.0.0.1:8088**
   (если не открылся — открой вручную).
4. Вкладка **🛠 Настройки** — все параметры: FPS камеры, FPS live-потока,
   качество JPEG, камеры, ИИ, запись, уведомления. Нажми
   **«Сохранить и применить»** — камеры перезапустятся с новыми значениями.
5. **`settings.bat`** — те же настройки в окне Tkinter (без браузера).

### Вариант 2 — вручную (Windows / Linux)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-pc.txt      # easyocr-резерв + опциональный DirectML

python scripts/download_models.py --config config/config.yaml
python scripts/install_alpr.py

python main.py                      # камера + панель + браузер
```

Полезные режимы:

```bash
python main.py --once               # 30 кадров и выход — проверка
python main.py --preview            # окно с рамками
python main.py --no-web             # только камера, без панели
python scripts/settings.py          # настройки в Tkinter
```

### Проверка без камеры

```bash
python scripts/make_test_video.py
python main.py --config config/config.yaml --once
```

### Самопроверка

```bash
python scripts/self_check.py
python -m pytest -q
```

Ожидаемо: `83 passed`, самопроверка без ошибок, `--once` завершается с кодом `0`.

---

## 🧰 Команды

### Запуск

| Команда | Что делает |
|---|---|
| `python main.py` | камера + панель по `config/config.yaml` (по умолчанию веб-камера) |
| `python main.py --config ПУТЬ` | другой конфиг (по умолчанию `config/config.yaml`) |
| `python main.py --cameras 0 2` | только выбранные камеры из списка |
| `python main.py --once` | ~30 кадров и выход — быстрая проверка |
| `python main.py --preview` | окно с рамками |
| `python main.py --recording` | запись видео |
| `python main.py --web` | поднять веб-панель вместе с камерой (старт/стоп и снимок доступны) |
| `python main.py --no-web` | без веб-панели |
| `python scripts/settings.py` | настройки в окне Tkinter |
| `python main.py --register` | подготовить лица из `data/datasets/raw/` |
| `python main.py --self-test` | тестовое уведомление в Telegram/ntfy |

### Скрипты

| Скрипт | Зачем |
|---|---|
| `scripts/download_models.py` | скачать YOLO, лица и готовую модель марки |
| `scripts/download_car_brand_model.py` | только модель марки |
| `scripts/install_alpr.py` | детектор и OCR номеров (FastALPR) |
| `scripts/web_panel.py` | веб-панель отдельным процессом (статусы камер — только просмотр) |
| `scripts/settings.py` | настройки в окне Tkinter (все поля, как в панели) |
| `settings.bat` | то же в один двойной клик (Windows) |
| `scripts/collect_car_images.py` | собрать фото машин в датасет марок |
| `scripts/train_brand_classifier.py` | обучить свою модель марок |
| `scripts/show_events.py` | последние события в консоли |
| `scripts/self_check.py` | самопроверка окружения и моделей |
| `scripts/make_test_video.py` | тестовое видео без камеры |
| `scripts/add_person.py` | добавить человека в датасет лиц |
| `scripts/register_faces.py` | пересчитать базу лиц |

### Тесты

```bash
python -m pytest -q
```

---

## ⚙️ Настройка

**Один файл — `config/config.yaml`.** Его читают камера, веб-панель и Tkinter.
Менять можно тремя способами:

1. **Веб-панель → вкладка «🛠 Настройки»** — формы по разделам, кнопка
   «Сохранить и применить» перезапускает камеры сама.
2. **`settings.bat` / `python scripts/settings.py`** — то же окно Tkinter;
   если камера запущена, настройки уйдут в неё по HTTP.
3. **Вручную** — открой `config/config.yaml` в блокноте.

Главные «крутилки»:

| Параметр | Что задаёт |
|---|---|
| `camera.fps` | FPS камеры (0 — максимум камеры, 30 — плавно) |
| `camera.width` / `camera.height` | разрешение кадра |
| `web.stream_fps` | FPS live-потока в браузере (1–30) |
| `web.stream_quality` | качество JPEG live-потока (30–95) |
| `recording.fps` | FPS записи видео |
| `analysis.detect_every_n_frames` | как часто запускать ИИ |
| `web.overlay` / `web.face_mesh` | рамки и «чертёж лица» |

Минимальный конфиг — `config/config.yaml` (он же используется по умолчанию):

```yaml
cameras:
  - name: "Ноутбук (веб-камера)"
    source: 0
    enabled: true
  # своя IP-камера — просто добавь второй блок:
  # - name: "Двор"
  #   source: "rtsp://user:password@192.168.1.100:554/stream1"
  #   enabled: true

notify:
  telegram:
    enabled: true
    bot_token: "PASTE_TELEGRAM_BOT_TOKEN"
    chat_ids:
      - "111111111"

car:
  brand_enabled: true
  brand_min_conf: 0.25
  plate_engine: "fastalpr"
  known_plates:
    - "A123BC 777"

objects:
  animals:
    enabled: true              # dog / cat / bird
  packages:
    enabled: true              # backpack / suitcase / handbag
    stationary_seconds: 45     # когда вещь считать оставленной

web:
  enabled: true
  autostart: true
  host: "127.0.0.1"
  port: 8088
  token: ""                    # задай пароль для доступа с телефона
  control: true                # false — панель без кнопок управления
```

<details>
<summary><b>Все конфиги проекта</b></summary>

| Файл | Для чего |
|---|---|
| `config/config.yaml` | **единственный конфиг**: камеры, FPS, ИИ, запись, панель, уведомления |
| `config/config.yaml.bak` | автоматическая резервная копия, создаётся при сохранении из панели/Tkinter |

> Других конфигов нет. Для второго профиля (например «сервер») скопируй
> `config.yaml` в свой файл и запусти `python main.py --config мой.yaml`,
> но по умолчанию всегда читается `config/config.yaml`.

</details>

<details>
<summary><b>Три движка распознавания номеров</b></summary>

| Движок | Точность на РФ | Скорость | Что нужно |
|---|---|---|---|
| `fastalpr` | лучшая | средняя | `python scripts/install_alpr.py` |
| `easyocr` | средняя | медленно на CPU | `pip install easyocr` |
| `tesseract` | слабая | быстрый | tesseract в системе |

Переключение — одна строка: `car.plate_engine: "fastalpr"`.

</details>

---

## 🗂️ Архитектура проекта

```text
MLCamera/
├── main.py                  # точка входа (по умолчанию config/config.yaml)
├── src/
│   ├── config.py            # загрузка и проверка config/*.yaml
│   ├── camera.py            # RTSP / видеофайл / веб-камера + переподключение
│   ├── motion.py            # детектор движения — не грузим CPU зря
│   ├── night_vision.py      # ИК-кадры: усиление и усреднение
│   ├── pipeline.py          # главный цикл: детекция → трекинг → события
│   ├── detector.py          # YOLO: люди, машины, животные, вещи
│   ├── face_recognizer.py   # InsightFace: лица и имена
│   ├── car_analyzer.py      # марка (CNN) + цвет (HSV)
│   ├── plate_recognizer.py  # FastALPR / easyocr / tesseract
│   ├── tracker.py           # CentroidTracker и голосование
│   ├── notifier.py          # Telegram + ntfy
│   ├── recorder.py          # клипы вокруг событий и непрерывная запись
│   ├── database.py          # SQLite: события, кулдауны, запросы для панели
│   ├── camera_manager.py    # камеры в потоках: старт/стоп, статусы, снимок
│   ├── camera_status.py     # общий camera_status.json для отдельной панели
│   └── web_panel.py         # веб-панель: история + управление камерами
├── config/                  # pc / vps / local / universal
├── scripts/                 # установка, модели, лица, обучение, проверки
├── tests/                   # pytest: пайплайн, БД, панель, вещи
├── docs/                    # документация и картинки
└── data/                    # фото, видео, база, модели, логи (не в git)
```

### Где лежат данные

```text
data/
├── events/<камера>/<дата>/   # фото событий, вырезки номеров, клипы
├── videos/<камера>/          # непрерывная запись
├── db/ml_camera.db           # SQLite: лица, машины, животные, вещи
├── models/                   # YOLO, марка, labels и meta
├── datasets/raw/<Имя>/       # фото людей для распознавания
├── datasets/auto_unknown/    # незнакомцы для дообучения
├── datasets/cars/            # датасет марок (train / val / inbox)
└── logs/camera.log
```

Всё содержимое `data/` в git не попадает — это твои фото и база.

---

## 💻 ПК или VPS

| | ПК дома | VPS в облаке |
|---|---|---|
| **CPU** | Ryzen 5 5600 и лучше | 1–2 vCPU на 1–2 камеры |
| **RAM** | от 8 ГБ | 1–2 ГБ |
| **Модели** | `yolov8m`, `buffalo_s`, `fastalpr`, марка | `yolov8n`, `buffalo_s`, `tesseract` |
| **Плюсы** | быстрее, превью, запись, марка | работает 24/7, не зависит от домашнего ПК |
| **Минусы** | должен быть включён | слабее CPU, марку лучше отключить |

Подробно: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) и [`docs/VPS_CAMERAS.md`](docs/VPS_CAMERAS.md).

---

## 📚 Документация

| Документ | О чём |
|---|---|
| [`docs/ЗАПУСК.md`](docs/ЗАПУСК.md) | **подробная инструкция запуска**: от установки до FPS и настроек |
| [`docs/НАЧАЛО.md`](docs/НАЧАЛО.md) | пошагово для новичка: от установки до первого уведомления |
| [`docs/SETUP.md`](docs/SETUP.md) | настройка под ключ: камеры, бот, ntfy |
| [`docs/MODELS.md`](docs/MODELS.md) | все модели: размеры, лицензии, ограничения |
| [`docs/OBJECTS.md`](docs/OBJECTS.md) | животные и оставленные вещи: как работает и как настроить |
| [`docs/WEB.md`](docs/WEB.md) | веб-панель: запуск, токен, API, безопасность |
| [`docs/BRAND_RU.md`](docs/BRAND_RU.md) | своя модель марок для российского рынка |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | все команды, флаги и скрипты |
| [`docs/DATASET.md`](docs/DATASET.md) | как фотографировать людей для точного распознавания |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | деплой на VPS |
| [`docs/VPS_CAMERAS.md`](docs/VPS_CAMERAS.md) | как соединить домашнюю камеру и сервер |

---

## 🩺 Если что-то не работает

| Симптом | Что проверить |
|---|---|
| Камера не открывается | `cameras[].source`, логин/пароль в RTSP, `enabled: true` |
| Нет уведомлений | `notify.telegram.enabled`, токен, `chat_ids` → `python main.py --self-test` |
| Все лица — «Незнакомец» | не выполнена регистрация: `python main.py --register` |
| Нет событий о животных/вещах | `objects.enabled`, классы и `conf_threshold` |
| Вещь не считается оставленной | `stationary_seconds`; камера не должна поворачиваться (PTZ) |
| Панель пустая | `database.path` у камеры и панели один и тот же |
| Панель просит токен | добавить `?token=...` из `web.token` |
| Кнопки камер серые | панель запущена отдельно — нужен `python main.py --web` |
| `403` на кнопке | нет заголовка `X-MLCamera` (обычно только в curl) |
| Всё тормозит | `detect_every_n_frames`, `frame_scale`, отключить `brand_enabled` |
| Марка постоянно `unknown` | поднять/опустить `brand_min_conf` или обучить свою модель |

Полная диагностика: [`docs/RUNBOOK.md`](docs/RUNBOOK.md) и
[`docs/OBJECTS.md`](docs/OBJECTS.md).

---

## ⚠️ Ограничения (честно)

- **Марка автомобиля.** Готовая модель обучена на Stanford Cars — рынок США
  2012 года. В списке нет Kia, Lada, Skoda, Renault, Peugeot, Citroen.
  Поставь `brand_min_conf: 0.25–0.4` или обучи свою модель:
  [`docs/BRAND_RU.md`](docs/BRAND_RU.md).
- **Оставленные посылки.** Отдельного класса `package` в YOLO нет: работаем
  по `backpack`/`suitcase`/`handbag` + неподвижность. Коробка может не
  детектироваться, а сумка на стуле — дать ложное срабатывание.
  Подробности и настройка — [`docs/OBJECTS.md`](docs/OBJECTS.md).
- **Животные.** Уверенность ниже ночью и на мелких объектах; классы можно
  расширить в `objects.animals.classes` (например, `horse`, `cow`).
- **Веб-панель.** Без токена и HTTPS не выставляй её в интернет; для телефона
  задай `web.token` и используй локальную сеть или SSH-туннель.
  Управление камерами работает только в режиме `main.py`; отдельный
  `scripts/web_panel.py` — только просмотр статусов.
  См. [`docs/WEB.md`](docs/WEB.md).
- **Номера.** OCR может ошибаться на мелких, размытых и наклонных номерах.
  Помогают голосование (`plate_vote_frames`), правильный угол камеры и FastALPR.
- **Лица.** Веса InsightFace разрешены только для некоммерческих исследований.
  Код проекта можно использовать свободно, но для коммерческого продукта нужно
  отдельное разрешение InsightFace.
- **YOLO.** Ultralytics YOLOv8 распространяется под AGPL-3.0. Для коммерческого
  использования нужна коммерческая лицензия Ultralytics.
- **Ночь.** Ночью точность ниже: нужна ИК-подсветка и 2–5 ч/б фото человека в датасете.
- **База лиц.** Пока не выполнена регистрация, все лица будут «Незнакомец».

---

## 🗺️ Идеи развития

- [x] Готовая модель марки без обучения
- [x] Нейросетевые номера FastALPR
- [x] Документация по всем моделям
- [x] Уведомления о животных и оставленных посылках (на базе YOLO)
- [x] Своя модель марок для российского рынка
- [x] Веб-панель с историей событий
- [x] Несколько камер, выбор камер флагом `--cameras`
- [x] Сбор фото машин из событий в датасет
- [x] Управление камерами из веб-панели: старт/стоп и снимок кадра
- [ ] Настройка зон и линий прямо в веб-панели
- [ ] Просмотр архива записей из панели

---

<div align="center">
  <sub>
    MLCamera v1.1.0 · обработка локально · данные остаются у тебя<br>
    Сделано для дома, а не для облака.
  </sub>
</div>
