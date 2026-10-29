# RUNBOOK — справочник запуска: что, откуда и как

## 1. Какой файл запускать?

**Точка входа одна** — `src/main.py`. Запускается несколькими способами:

| Способ | Команда | Что делает |
|---|---|---|
| ⭐ Проще всего | `python main.py` | `config/config.yaml`: веб-камера ноутбука + live-поток + панель |
| ⭐ Двойной клик | `start_pc.bat` | ПК-версия: камера, панель, превью и запись |
| Настройки (клик) | `settings.bat` | окно Tkinter: все параметры |
| Настройки (команда) | `python scripts/settings.py` | то же без ярлыка |
| С превью | `python main.py --preview` | окно с рамками + панель |
| Через модуль | `python -m src.main` | идентично `python main.py` |
| В Docker | `docker compose -f docker-compose.prod.yml up -d` | авто-разворот на VPS |

> `main.py` в корне — просто обёртка над `src/main.py`: без `--config`
> читается `config/config.yaml`. В нём по умолчанию включена веб-камера `0`,
> поэтому программа работает «из коробки». Если IP-камера не открылась,
> сработает `camera.fallback_webcam: true` и подключится веб-камера.

## 2. Все команды запуска (`python main.py ...`)

| Флаг | Назначение |
|---|---|
| `--config ПУТЬ` | какой конфиг брать (по умолчанию `config/config.yaml`) |
| `--cameras 0 2` | запустить только камеры с индексами 0 и 2 |
| `--once` | прогнать ~30 кадров и выйти (отладка) |
| `--preview` | включить живое окно с видео и рамками (перекрывает конфиг) |
| `--recording` | включить запись видео (перекрывает конфиг) |
| `--web` | поднять веб-панель вместе с камерой |
| `--no-web` | не запускать веб-панель (даже если `web.autostart: true`) |
| `--self-test` | отправить тестовое уведомление в Telegram/ntfy |

Примеры:

```bash
python main.py --self-test        # проверить, что уведомления доходят
python main.py                    # веб-камера + панель (config.yaml)
python main.py --preview          # превью + панель
python main.py --no-web           # только камера
python main.py --once             # отладка: 30 кадров и выход
python main.py --cameras 0        # только первая камера
python scripts/settings.py        # настройки в Tkinter
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
| `web_panel.py` | веб-панель истории событий (отдельный процесс, без управления камерами) |
| `settings.py` | настройки в окне Tkinter — те же поля, что в веб-панели |
| `collect_car_images.py` | собрать фото машин из событий в датасет для разметки |
| `train_brand_classifier.py` | (опция) обучить свою модель марок на своих фото |
| `make_release_zip.py` | упаковать проект в zip для загрузки на VPS |
| `deploy_server.sh` | развернуть проект на сервере одной командой |

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

# 4) настройка: config/config.yaml  (IP камер, токен, chat_ids, ntfy topic)
# 5) модели и проверки:
python scripts/download_models.py --config config/config.yaml
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
| Свои марки для РФ | `python scripts/collect_car_images.py` → разложить → `python scripts/train_brand_classifier.py` | «Марка: Lada» |
| Животные (собака/кошка/птица) | `objects.animals.enabled: true` (уже стоит) | уведомление «Собака в кадре» |
| Оставленные вещи | `objects.packages.enabled: true` + `stationary_seconds` | «Оставлен рюкзак» |
| Веб-панель истории | `web.enabled: true`, `autostart: true` или `python scripts/web_panel.py` | http://127.0.0.1:8088 |
| Управление камерами из панели | панель внутри процесса: `python main.py --web` (или `autostart: true`) | вкладка «Камеры»: статус, старт/стоп, снимок |
| Доступ к панели с телефона | `web.host: "0.0.0.0"`, `web.token: "пароль"` | `http://IP:8088/?token=...` |
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
- **Животные и вещи:** `data/events/<камера>/<дата>/animal_*.jpg`, `package_*.jpg`
- **Лица для авто-обучения:** `data/datasets/auto_unknown/`
- **Фото машин для датасета марок:** `data/datasets/cars/inbox/`
- **База событий:** `data/db/ml_camera.db` → `python scripts/show_events.py`
- **Веб-панель:** http://127.0.0.1:8088 (или `python scripts/web_panel.py --open`)
- **Логи:** `data/logs/camera.log`

---

## 7. Управление камерами из панели

Открой http://127.0.0.1:8088 → вкладка **«Камеры»**.

В карточке видно live-поток с рамками объектов и «чертежом лица»
(106 точек InsightFace), разрешение, fps обработки, аптайм и события за 24 ч.

| Статус | Что значит |
|---|---|
| `running` | поток идёт, кадры обрабатываются |
| `starting` | камера открывается / грузятся модели |
| `stopping` | мягкая остановка, ждём завершения потока |
| `stopped` | камера не работает (в конфиге или остановлена вручную) |
| `error` | не удалось открыть поток — смотри текст ошибки в карточке |

Кнопки:

- **▶ Запустить** — поднимает поток без перезапуска `main.py`; модели уже в памяти.
- **■ Остановить** — завершает поток, процесс и панель продолжают работать.
- **📸 Снимок** — текущий кадр с рамками и «чертежом лица»; можно скачать JPEG.
- клик по превью — камера целиком, live и последние события.

> Управление доступно, только если панель запущена **в процессе камеры**
> (`python main.py --web` или `web.autostart: true`). Отдельный
> `scripts/web_panel.py` читает `data/db/camera_status.json` и показывает
> статусы, но кнопки отключены.

Проверка из консоли:

```bash
curl "http://127.0.0.1:8088/api/cameras"
curl -X POST -H "X-MLCamera: 1" "http://127.0.0.1:8088/api/cameras/stop?index=0"
curl -X POST -H "X-MLCamera: 1" "http://127.0.0.1:8088/api/cameras/start?index=0"
curl -o frame.jpg "http://127.0.0.1:8088/api/cameras/snapshot?index=0"
```

| Симптом | Причина / решение |
|---|---|
| Кнопки серые, в карточке «только просмотр» | панель запущена отдельным процессом — используй `python main.py --web` |
| `403` на POST | нет заголовка `X-MLCamera: 1` или чужой `Origin` |
| `409` при старте | камера уже запущена; обнови страницу |
| `409` на всех кнопках | `web.control: false` в конфиге |
| `404` у снимка | камера ещё не дала кадров или стоит неверный `index` |
| Камера сразу `error` | проверь RTSP-адрес, логин/пароль и сеть; смотри `data/logs/camera.log` |
| Все IP-камеры `error` | включи `camera.fallback_webcam: true` — подставится веб-камера `0` |
| Нет live-потока в карточке | камера `stopped`/`error` или панель открыта отдельно от камеры |
| «Чертёж лица» не появляется | проверь `web.overlay: true`, `web.face_mesh: true` и что в кадре есть лицо; модель InsightFace скачана |
| Поток тормозит | уменьши `analysis.detect_every_n_frames` → увеличь число, `frame_scale: 0.5`; закрой лишние камеры |
| После остановки всех камер процесс жив | так и задумано: панель должна дать запустить камеру снова |
| Статусы в отдельной панели не меняются | процесс камеры не пишет `data/db/camera_status.json` — проверь, что он запущен; на экране будет предупреждение «нет свежих данных» |

Подробнее про API и CSRF: [`WEB.md`](WEB.md).
