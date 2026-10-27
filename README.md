<div align="center">
  <img src="docs/assets/banner.gif" alt="MLCamera — умная камера для семьи" width="100%">
</div>

<h1 align="center">MLCamera</h1>

<p align="center">
  <b>IP-камера, которая знает родных, читает номера, определяет марку машины и пишет в Telegram.</b><br>
  Всё считается локально: без облачных подписок и отправки видео на чужие серверы.
</p>

<p align="center">
  <a href="#-что-умеет">Возможности</a> ·
  <a href="#-как-это-работает">Как работает</a> ·
  <a href="#-модели">Модели</a> ·
  <a href="#-быстрый-старт">Быстрый старт</a> ·
  <a href="#-документация">Документация</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Windows | Linux" src="https://img.shields.io/badge/Windows%20%7C%20Linux-0f172a?style=for-the-badge&logo=linux&logoColor=white">
  <img alt="5 neural models" src="https://img.shields.io/badge/neural%20models-5-42e8a0?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/tests-34%20passed-42e8a0?style=for-the-badge">
  <img alt="Version" src="https://img.shields.io/badge/version-v1.0.0-40d6ff?style=for-the-badge">
  <img alt="Local-first" src="https://img.shields.io/badge/privacy-local--first-966eff?style=for-the-badge">
</p>

---

## 🎯 Что это

MLCamera — домашняя камера, которая не просто пишет видео, а **понимает, что происходит**:

- узнаёт членов семьи по лицу и пишет «Алексей пришёл»;
- отдельно сообщает о незнакомцах и сохраняет их лица для дообучения;
- замечает машину, читает номер, определяет цвет и марку;
- не спамит: одна машина или человек = одно уведомление за проход;
- работает днём и ночью, на ПК и на дешёвом VPS;
- отправляет фото и текст в Telegram или на телефон через ntfy.

> [!NOTE]
> Все нейросети — **готовые**. Ничего обучать вручную не нужно: модели
> скачиваются скриптами. Полный список с размерами и лицензиями —
> в [`docs/MODELS.md`](docs/MODELS.md).

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
| 🎨 **Цвет** | HSV-гистограмма по центральной зоне кузова |
| 🌙 **Ночное видение** | ИК-режим камеры + усиление тёмных кадров и усреднение |
| ⏱️ **Без спама** | CentroidTracker: 2 минуты перед камерой = 1 уведомление |
| 🟢/🔴 **Вошёл / вышел** | Пересечение линии `tracking.entry_line_y` |
| ⭐ **Свой номер** | Белый список `car.known_plates` → «Своя машина» |
| 🌅 **Утренний дайджест** | Сводка за сутки в 09:00: люди, машины, незнакомцы |
| 🔧 **Мультиразрешение** | Нет движения → анализ мелким кадром, экономия CPU |
| 🎬 **Запись клипов** | Короткие видео вокруг события или непрерывная запись |
| 📨 **Telegram + ntfy** | Фото, текст и подписи в несколько чатов одновременно |

---

## 🧠 Как это работает

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="Схема обработки кадра MLCamera" width="100%">
</p>

1. **Камера** отдаёт RTSP-поток или видеофайл.
2. **Детектор движения** решает, стоит ли запускать тяжёлый анализ. Если в кадре тихо — CPU почти не тратится.
3. **YOLO** находит людей и машины.
4. Дальше кадр расходится по двум веткам:
   - **лица** → InsightFace → поиск по базе семьи → событие;
   - **машины** → трекер → номер (ALPR), цвет (HSV), марка (CNN).
5. Результат сохраняется в **SQLite + фото**, а владельцу уходит **уведомление** в Telegram/ntfy.
6. Если включена запись — рядом с событием сохраняется короткий **видеоклип**.

> Главная оптимизация: марка машины считается **один раз на трек**, а номер —
> голосованием по нескольким кадрам. Поэтому камера не захлёбывается на CPU.

---

## 🤖 Модели

<p align="center">
  <img src="docs/assets/models.svg" alt="Нейросети внутри MLCamera" width="100%">
</p>

| Задача | Модель | Размер | Лицензия |
|---|---|---|---|
| Люди и машины | YOLOv8m (ПК) / YOLOv8n (VPS) | 6–50 МБ | AGPL-3.0 |
| Лица и эмбеддинги | InsightFace `buffalo_s` / `buffalo_l` | 158 МБ | код MIT, веса — research |
| Марка автомобиля | ConvNeXt-Tiny, Stanford Cars, 49 марок | 106 МБ | Apache-2.0 |
| Поиск номера | YOLOv9-t 384 license-plate end2end | 7.4 МБ | MIT |
| Чтение номера | European plates MobileViT v2 | 14.5 МБ | MIT |

**Команды для скачивания:**

```bash
python scripts/download_models.py --config config/config.pc.yaml
python scripts/download_car_brand_model.py   # только марка, если нужно отдельно
python scripts/install_alpr.py               # детектор + OCR номеров
```

**Что важно знать:**

- Марка определяется по лучшей вырезке за трек, а не каждый кадр — так точнее и дешевле.
- Номер собирается голосованием, а не по одному случайному кадру.
- `brand_min_conf` не даёт модели выдать случайную марку: ниже порога будет `unknown`.
- Полные пути, кэши и ограничения — в [`docs/MODELS.md`](docs/MODELS.md).

---

## 🚀 Быстрый старт

### Вариант 1 — Windows, двойной клик

1. Запусти **`pc\setup.bat`**.
   Он создаст окружение, поставит библиотеки, скачает YOLO, лица, марку и ALPR.
2. Открой **`config\config.pc.yaml`** и замени:
   - RTSP-адреса камер;
   - токен Telegram-бота и `chat_ids`;
   - при желании — свой номер в `known_plates`.
3. Запусти **`pc\run_pc.bat`** (или `python main.py`).

### Вариант 2 — вручную (Windows / Linux)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-pc.txt      # easyocr-резерв + опциональный DirectML

python scripts/download_models.py --config config/config.pc.yaml
python scripts/install_alpr.py
python main.py --config config/config.pc.yaml
```

### Проверка без камеры

```bash
python scripts/make_test_video.py
python main.py --config config/config.local.yaml --once
```

### Самопроверка

```bash
python scripts/self_check.py
python -m pytest -q
```

Ожидаемо: `34 passed`, самопроверка без ошибок, `--once` завершается с кодом `0`.

---

## ⚙️ Настройка

Минимальный конфиг — `config/config.pc.yaml`:

```yaml
cameras:
  - name: "Двор"
    source: "rtsp://user:password@192.168.1.100:554/stream1"
    enabled: true

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
```

<details>
<summary><b>Все конфиги проекта</b></summary>

| Файл | Для чего |
|---|---|
| `config/config.yaml` | универсальный, по умолчанию |
| `config/config.pc.yaml` | мощный ПК: превью, запись, `yolov8m`, `fastalpr` |
| `config/config.vps.yaml` | слабый сервер: `yolov8n`, `tesseract`, без марки |
| `config/config.local.yaml` | отладка на видеофайле без камеры |
| `config/config.vps.example.yaml` | шаблон для VPS |

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
├── main.py                  # точка входа (по умолчанию ПК-конфиг)
├── src/
│   ├── camera.py            # RTSP / видеофайл / веб-камера
│   ├── pipeline.py          # главный цикл: детекция → трекинг → события
│   ├── detector.py          # YOLO: люди и машины
│   ├── face_recognizer.py   # InsightFace: лица и имена
│   ├── car_analyzer.py      # марка (CNN) + цвет (HSV)
│   ├── plate_recognizer.py  # FastALPR / easyocr / tesseract
│   ├── tracker.py           # CentroidTracker и голосование
│   ├── notifier.py          # Telegram + ntfy
│   ├── recorder.py          # клипы вокруг событий
│   └── db.py                # SQLite: события и кулдауны
├── config/                  # pc / vps / local / universal
├── scripts/                 # установка, модели, лица, проверки
├── docs/                    # документация и картинки
└── data/                    # фото, видео, база, логи (не в git)
```

---

## 🖥️ ПК или VPS

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
| [`docs/НАЧАЛО.md`](docs/НАЧАЛО.md) | пошагово для новичка: от установки до первого уведомления |
| [`docs/SETUP.md`](docs/SETUP.md) | настройка под ключ: камеры, бот, ntfy |
| [`docs/MODELS.md`](docs/MODELS.md) | все модели: размеры, лицензии, ограничения |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | все команды, флаги и скрипты |
| [`docs/DATASET.md`](docs/DATASET.md) | как фотографировать людей для точного распознавания |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | деплой на VPS |
| [`docs/VPS_CAMERAS.md`](docs/VPS_CAMERAS.md) | как соединить домашнюю камеру и сервер |

---

## ⚠️ Ограничения (честно)

- **Марка автомобиля.** Модель обучена на Stanford Cars — рынок США 2012 года.
  В списке нет Kia, Lada, Skoda, Renault, Peugeot, Citroen. На типичных
  российских дворовых ракурсах возможны ошибки. Поставь `brand_min_conf: 0.25–0.4`
  или отключи марку: `brand_enabled: false`.
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
- [ ] Уведомления о животных и оставленных посылках (на базе YOLO)
- [ ] Своя модель марок для российского рынка
- [ ] Веб-панель с историей событий

---

<div align="center">
  <sub>
    MLCamera v1.0.0 · обработка локально · данные остаются у тебя<br>
    Сделано для дома, а не для облака.
  </sub>
</div>
