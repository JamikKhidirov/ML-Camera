# RUNBOOK — справочник запуска: что, откуда и как

## 1. Какой файл запускать?

**Точка входа одна** — `src/main.py`. Запускается несколькими способами:

| Способ | Команда | Что делает |
|---|---|---|
| ⭐ Проще всего | `python main.py` | ПК-версия (config/config.pc.yaml) + превью + запись видео + уведомления |
| ⭐ Двойной клик | `start_pc.bat` | то же, что `python main.py --preview` |
| Вручную (ПК) | `python main.py --config config/config.pc.yaml --preview` | то же самое явно |
| Вручную (сервер) | `python main.py --config config/config.vps.yaml` | лёгкая версия без видео |
| Через модуль | `python -m src.main --config ...` | идентично `python main.py ...` |
| В Docker | `docker compose -f docker-compose.prod.yml up -d` | авто-разворот на VPS |

> `main.py` в корне — просто обёртка над `src/main.py`: сам подставляет
> `--config config/config.pc.yaml`, если не указан другой. Внутри main.py
> находит все включённые камеры из конфига и запускает для каждой отдельный
> поток (в котором уже вызываются `camera.py` → `pipeline.py` → ...).

## 2. Все команды запуска (`python main.py ...`)

| Флаг | Назначение |
|---|---|
| `--config ПУТЬ` | какой конфиг брать (pc/vps/local) |
| `--cameras 0 2` | запустить только камеры с индексами 0 и 2 |
| `--once` | прогнать ~30 кадров и выйти (отладка) |
| `--preview` | включить живое окно с видео и рамками (перекрывает конфиг) |
| `--recording` | включить запись видео (перекрывает конфиг) |
| `--self-test` | отправить тестовое уведомление в Telegram/ntfy |

Примеры:

```bash
python main.py --self-test                                # проверить, что уведомления доходят
python main.py                                            # ПК: превью + запись + уведомления
python main.py --config config/config.vps.yaml            # серверная версия (без видео)
python main.py --config config/config.local.yaml --once   # отладка на тест-видео
python main.py --cameras 0                                # только первая камера
```

## 3. Скрипты (`python scripts/...`)

| Скрипт | Что делает |
|---|---|
| `make_dirs.py` | создать все папки проекта (данные, модели, видео...) |
| `add_person.py Имя` | добавить нового человека (создать папки) |
| `organize_folders.py` | переименовать папки на настоящие имена + подписи «кто есть кто» |
| `collect_faces.py` | вырезать лица из фото в `data/datasets/raw/<имя>/` |
| `register_faces.py` | запомнить лица (эмбеддинги) — после этого камера узнаёт людей |
| `download_models.py` | скачать/проверить модели (YOLO, InsightFace, марка авто, OCR) |
| `download_car_brand_model.py` | скачать готовую модель марки (49 марок) и собрать ONNX |
| `install_alpr.py` | поставить FastALPR (детектор + OCR номеров) и починить OpenCV GUI |
| `make_test_video.py` | сгенерить тестовое видео без камеры |
| `make_readme_assets.py` | пересобрать анимации и схемы для README (`docs/assets/`) |
| `self_check.py` | ПОЛНАЯ самопроверка всех модулей (что всё работает) |
| `show_events.py` | вывести последние события из базы |
| `make_release_zip.py` | упаковать проект в zip для загрузки на VPS |
| `deploy_server.sh` | развернуть проект на сервере одной командой |
| `train_brand_classifier.py` | (опция) обучить свою модель марок на своих фото |

## 4. Всё одним махом на ПК (по шагам)

```bash
# 1) зависимости (один раз)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-pc.txt        # easyocr (резерв) + (опц.) DirectML

# 2) папки и люди
python scripts/make_dirs.py
python scripts/organize_folders.py        # имена папок = имена в уведомлениях
python scripts/add_person.py Бабушка      # если нужен ещё человек

# 3) фото в data/datasets/raw/<имя>/ (см. docs/DATASET.md) и обучение:
python scripts/collect_faces.py
python scripts/register_faces.py

# 4) настройка: config/config.pc.yaml  (IP камер, токен, chat_ids, ntfy topic)
# 5) модели и проверки:
python scripts/download_models.py --config config/config.pc.yaml
python scripts/install_alpr.py             # нейросетевые номера
python main.py --self-test
python scripts/self_check.py

# 6) запуск:
start_pc.bat        # или: python main.py
```

## 5. Что ещё можно сделать (функции и как их включить)

| Идея | Как включить | Где смотреть результат |
|---|---|---|
| Точные номера через FastALPR (нейросеть) | `python scripts/install_alpr.py`, в конфиге `plate_engine: "fastalpr"` | уведомления с номером |
| Запасной OCR: easyocr | в конфиге `plate_engine: "easyocr"` + `pip install -r requirements-pc.txt` | уведомления с номером |
| Марки машин (Toyota, BMW...) | `python scripts/download_car_brand_model.py`, `brand_enabled: true` | в уведомлениях: «Марка: Toyota» |
| Свои марки, которых нет в модели | `python scripts/train_brand_classifier.py` на своих фото | «Марка: ...» |
| Клипы вокруг событий вместо сплошной записи | `recording.mode: "events"` | `data/videos/<камера>/` |
| Дольше хранить видео | поднять `recording.rotate_seconds` | — |
| Следить за незнакомцами | `face.notify_unknown: true` (уже стоит) | «Незнакомец у камеры» + фото |
| Авто-обучение новых людей | `face.auto_capture_unknown: true` | `data/datasets/auto_unknown/` |
| Ускорение лица видеокартой AMD | `pip install onnxruntime-directml` (см. requirements-pc.txt) | — |
| Больше камер | добавить блок в `cameras:` (любое число) | — |
| Слать не только тебе | добавить id в `notify.telegram.chat_ids` | Telegram всем сразу |
| Push на Android без Telegram | подписка в приложении ntfy на свой `topic` | пуш на телефон |
| Ночной режим | `camera.night_enabled: true` + ИК на камере | — |
| НЕ хранить фото после отправки | `notify.delete_images_after_send: true` (на VPS стоит) | диск свободен |
| 1 уведомление за проход вместо спама | `tracking.enabled: true` (уже стоит) | человек стоял 2 мин = 1 сообщение |
| «Вошёл / Вышел» | `tracking.entry_line_y: 0.5` (подгони под свою камеру) | «Алексей вошёл», «Олег вышел» |
| Свой номер = «Своя машина» | вписать номера в `car.known_plates` | «Своя машина A123BC 777» |
| Номер читается точнее | поднять `car.plate_vote_frames: 5` (голосование по кадрам) | в уведомлении стабильный номер |
| Утренний дайджест за сутки | `digest.enabled: true`, `time: "09:00"` | в 09:00 сводка: кто был, какие машины |
| Экономия CPU без движения | `analysis.multi_resolution: true` (уже стоит) | нет шевеления → анализ мелким кадром |
| Ночью номер с трудом | ночное усреднение `camera.frame_averaging_night: true` + `frame_averaging_count: 3` | меньше шума ночью |

## 6. Куда смотреть результаты

- **Видео с камер (ПК):** `data/videos/<камера>/`
- **Фото событий и вырезки номеров:** `data/events/<камера>/<дата>/`
- **Лица для авто-обучения:** `data/datasets/auto_unknown/`
- **База событий:** `data/db/ml_camera.db` → `python scripts/show_events.py`
- **Логи:** `data/logs/camera.log`