# MLCamera — умная камера для семьи

Распознаёт членов семьи по лицу (по **именам**), следит за машинами
(**фотографирует номер, марку, цвет**), пишет всё в базу и шлёт уведомления в
**Telegram** (на несколько аккаунтов сразу) и **push на Android** (ntfy).
Работает на одной или нескольких камерах, есть **ночное видение** (ИК +
усиление тёмного кадра) и **запись видео** на ПК в отдельную папку.

---

## 🚀 Впервые здесь? Начни отсюда

> **Если ты никогда не программировал** — читай **`docs/НАЧАЛО.md`**:
> там ВСЁ по шагам: что скачать, куда положить фотографии родных, где
> вписать токен бота и как всё запустить, даже если ты «полный чайник».

Самый короткий путь к работающей камере:

1. 📥 Двойной клик **`pc\setup.bat`** (поставит всё сам, ~10 минут);
2. ✏️ Открой **`config\config.pc.yaml`** → впиши **токен бота**, **id**
   аккаунтов (кому писать), **IP/логин/пароль камер**;
3. 📷 Положи **фото родных** в `data\datasets\raw\ИМЯ\` (папка = имя, 10–25 фото);
4. 🧠 `python main.py --register` — камера запомнит лица;
5. ▶️ `python main.py` — камера работает!

| Быстрые ссылки | Что внутри |
|---|---|
| **[docs/НАЧАЛО.md](docs/НАЧАЛО.md)** | ★ главная инструкция для всех |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | справочник всех команд |
| [docs/DATASET.md](docs/DATASET.md) | как фотографировать людей |
| [docs/SETUP.md](docs/SETUP.md) | настройка под ключ |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | развёртывание на VPS |
| [docs/VPS_CAMERAS.md](docs/VPS_CAMERAS.md) | камера + сервер |
| [pc/build_exe.bat](pc/build_exe.bat) | собрать `.exe` с иконкой (двойной клик) |
| [pc/deploy_vps.py](pc/deploy_vps.py) | авто-деплой на сервер одной командой |
| [data/README.txt](data/README.txt) | куда что класть в `data/` |

---

## 🗺 Схема проекта: где что лежит и куда нажимать

Всё по-простому: что хочешь сделать → **какой файл/папка** → **ссылка**.

### ⚙️ НАСТРОЙКИ (что заполнять перед запуском)
| Что вписать | Файл | Ссылка |
|---|---|---|
| Токен Telegram-бота, id аккаунтов (кому писать) | `config/config.pc.yaml` | [открыть](config/config.pc.yaml) |
| Логин/пароль/IP камер | `config/config.pc.yaml` → `cameras:` | [открыть](config/config.pc.yaml) |
| Пуши на телефон (ntfy topic) | `config/config.pc.yaml` → `notify.push:` | [открыть](config/config.pc.yaml) |
| Свои машины (номера) | `config/config.pc.yaml` → `car.known_plates` | [открыть](config/config.pc.yaml) |
| Имена людей | `config/config.pc.yaml` → `people.names` | [открыть](config/config.pc.yaml) |
| ПК-версия / серверная | `config/config.pc.yaml` / `config/config.vps.yaml` | [pc](config/config.pc.yaml) · [vps](config/config.vps.yaml) |

### 📷 ФОТОГРАФИИ ЛЮДЕЙ (куда класть)
| Что сделать | Папка | Ссылка |
|---|---|---|
| Кинуть 10–25 фото родных (папка = имя) | `data/datasets/raw/ИМЯ/` | [открыть папку](data/datasets/raw) |
| Фото незнакомцев для авто-обучения | `data/datasets/auto_unknown/` | [открыть папку](data/datasets/auto_unknown) |
| Запомнить лица после добавления фото | команда `python main.py --register` | [main.py](main.py) |

### ▶️ ЗАПУСК (что нажать)
| Хочу | Что | Ссылка |
|---|---|---|
| Запустить камеру (окно + видео + уведомления) | `python main.py` | [main.py](main.py) |
| Запустить двойным кликом на ПК | `start_pc.bat` | [start_pc.bat](start_pc.bat) |
| Первая настройка проекта (ставит всё само) | `pc/setup.bat` | [setup.bat](pc/setup.bat) |
| Собрать `.exe` с иконкой | `pc/build_exe.bat` → `pc/dist/MLCamera/MLCamera.exe` | [build_exe.bat](pc/build_exe.bat) |
| Проверить, что уведомления доходят | `python main.py --self-test` | [main.py](main.py) |
| Прогнать тесты и самопроверку | `run_tests.bat` | [run_tests.bat](run_tests.bat) |

### ☁️ СЕРВЕР VPS (чтобы работало без ПК, 24/7)
| Хочу | Что | Ссылка |
|---|---|---|
| Вписать IP/пароль сервера | `pc/deploy.yaml` (копия: [deploy.example.yaml](pc/deploy.example.yaml)) | [открыть](pc/deploy.example.yaml) |
| Автоматически развернуть проект на сервере | `python pc/deploy_vps.py` | [deploy_vps.py](pc/deploy_vps.py) |
| Деплой через GitHub Actions (вариант Б) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | [открыть](docs/DEPLOYMENT.md) |
| Как соединить камеру и сервер (WireGuard и т.п.) | [docs/VPS_CAMERAS.md](docs/VPS_CAMERAS.md) | [открыть](docs/VPS_CAMERAS.md) |

### 📁 ГДЕ СМОТРЕТЬ РЕЗУЛЬТАТ
| Что | Папка | Ссылка |
|---|---|---|
| Фото событий (что пришло в Telegram) | `data/events/<камера>/<дата>/` | [открыть папку](data/events) |
| Видео с камер (ПК) | `data/videos/<камера>/` | [открыть папку](data/videos) |
| База событий | `data/db/ml_camera.db` + `python scripts/show_events.py` | [show_events.py](scripts/show_events.py) |
| Логи (если что-то не работает) | `data/logs/camera.log` | [открыть папку](data/logs) |

### 📚 ДОКУМЕНТАЦИЯ (все файлы)
| Документ | О чём | Ссылка |
|---|---|---|
| ★ **НАЧАЛО.md** | главный гайд «с нуля до работающей камеры» | [открыть](docs/НАЧАЛО.md) |
| RUNBOOK.md | все команды, скрипты, что ещё можно сделать | [открыть](docs/RUNBOOK.md) |
| DATASET.md | как правильно фотографировать людей | [открыть](docs/DATASET.md) |
| SETUP.md | настройка под ключ (камеры, бот, пуши) | [открыть](docs/SETUP.md) |
| DEPLOYMENT.md | развёртывание на VPS | [открыть](docs/DEPLOYMENT.md) |
| VPS_CAMERAS.md | камера + сервер (сети, проброс портов) | [открыть](docs/VPS_CAMERAS.md) |
| data/README.txt | куда что класть в папке `data/` | [открыть](data/README.txt) |

### 🧰 СКРИПТЫ (папка `scripts/`, вызываются `python scripts/...`)
| Скрипт | Что делает | Ссылка |
|---|---|---|
| `make_dirs.py` | создать все папки проекта | [открыть](scripts/make_dirs.py) |
| `add_person.py Имя` | добавить нового человека | [открыть](scripts/add_person.py) |
| `collect_faces.py` | вырезать лица из фото | [открыть](scripts/collect_faces.py) |
| `register_faces.py` | запомнить лица (эмбеддинги) | [открыть](scripts/register_faces.py) |
| `download_models.py` | скачать/проверить модели ИИ | [открыть](scripts/download_models.py) |
| `make_test_video.py` | сгенерировать тестовое видео без камеры | [открыть](scripts/make_test_video.py) |
| `self_check.py` | полная самопроверка всех модулей | [открыть](scripts/self_check.py) |
| `show_events.py` | показать последние события из базы | [открыть](scripts/show_events.py) |
| `make_icon.py` | пересобрать иконку `pc/MLCamera.ico` | [открыть](scripts/make_icon.py) |
| `make_release_zip.py` | упаковать проект в zip для сервера | [открыть](scripts/make_release_zip.py) |
| `deploy_server.sh` | развернуть на сервере (запускается `pc/deploy_vps.py`) | [открыть](scripts/deploy_server.sh) |
| `train_brand_classifier.py` | (опция) обучить распознавание марок | [открыть](scripts/train_brand_classifier.py) |

### 📦 ПАПКА `pc/` (всё для запуска с компьютера)
| Файл | Что это | Ссылка |
|---|---|---|
| `setup.bat` | первый запуск (ставит всё само) | [открыть](pc/setup.bat) |
| `build_exe.bat` | собрать `.exe` с иконкой | [открыть](pc/build_exe.bat) |
| `MLCamera.ico` | иконка программы | [открыть](pc/MLCamera.ico) |
| `deploy_vps.py` | авто-деплой на VPS | [открыть](pc/deploy_vps.py) |
| `deploy.example.yaml` | шаблон: IP/пароль сервера | [открыть](pc/deploy.example.yaml) |

---

## 1. Что умеет

| Возможность | Как работает |
|---|---|
| «Кто пришёл» — по имени | распознавание лиц InsightFace. Уведомление: «**Аня у камеры**» + фото |
| Незнакомец | отдельное уведомление + лицо сохраняется для обучения |
| Авто-обучение | лица незнакомцев копятся в `data/datasets/auto_unknown/` — переложил в папку человека → камера запомнила |
| Машина проехала | детекция YOLO |
| **Фото номера машины** | вырезка нижней части авто + OCR → «А123ВС 777» |
| **Цвет машины** | по гистограмме HSV → «белый, серебристый, чёрный...» |
| Марка машины (опция) | классификатор MobileNetV3 (обучается сам, см. `scripts/train_brand_classifier.py`) |
| Фото сохраняется | полный кадр + вырезка номера, потом можно **авто-удалить** после отправки |
| Ночное видение | ИК-режим камеры + CLAHE/шумоподавление в коде |
| **Трекинг (без спама)** | человек стоит 2 мин = **1 уведомление** за проход; машина — 1 за проезд |
| **«Вошёл / вышел»** | пересечение линии в кадре (`tracking.entry_line_y`) |
| **Свой номер vs чужой** | белый список `car.known_plates` → «Своя машина» / «Чужая машина» |
| **Голосование по номеру** | номер собирается с нескольких кадров, берётся самый частый OCR (`plate_vote_frames`) |
| **Утренний дайджест** | в 09:00 сводка за сутки: кто приходил, какие машины, незнакомцы |
| **Мультиразрешение** | нет движения → анализ мелким кадром (экономия CPU); есть → полным |
| **Усреднение ночных кадров** | N ночных кадров усредняются, номер читается даже в темноте |
| Уведомления | Telegram **нескольким аккаунтам** + пуш ntfy на Android |
| Запись видео (ПК) | в папку `data/videos/<камера>/` в mp4 |
| Несколько камер | любые 1-100 камер в `cameras:` |

---

## 2. Что нужно — требования

### Для запуска на компьютере (вариант «дома»)

| Компонент | Минимально / рекомендуемо |
|---|---|
| ОС | Windows 10/11, Linux |
| CPU | любой 4-ядерный; твой **Ryzen 5 5600** — с запасом |
| RAM | от 8 ГБ |
| Видеокарта | не обязательна (всё на CPU). **RX 5700 XT** ускорит распознавание лиц через DirectML |
| Диск | 5 ГБ (модели + события + видео) |
| Камера | IP с RTSP и ИК-подсветкой (Xiaomi, Hikvision, Dahua, TP-Link...) |

### Для запуска на сервере VPS (вариант «в облаке»)

| Параметр | Какой нужен |
|---|---|
| **CPU** | **1-2 vCPU** для 1-2 камер; **4 vCPU** для 3+ камер |
| **RAM** | **1-2 ГБ** (1-2 камеры); **4 ГБ** (3+ камер) |
| **Диск** | от 10 ГБ SSD |
| **ОС** | Ubuntu 22.04 / 24.04 |
| **Сеть** | 1 Гбит; нужен доступ до камер (см. ниже) |
| Где брать | Timeweb Cloud, VDSina, Aeza, Selectel — от ~300 ₽/мес |

> ⚠️ На VPS 1-2 ГБ **не** включай: `easyocr`, `preview`, `recording`, `brand_enabled`.
> Конфиг `config/config.vps.yaml` уже настроен правильно.

---

## 3. Быстрый старт на компьютере (Windows 11 + Ryzen + RX 5700 XT)

### Шаг 1. Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# (опционально) точные номера + ускорение лица через видеокарту AMD:
pip install -r requirements-pc.txt
```

### Шаг 2. Папки для фотографий людей

```bash
python scripts/make_dirs.py              # создаёт все папки (и data/videos)
python scripts/add_person.py Аня         # добавить человека (можно сколько угодно)
```

Положи по 20-25 фото каждого в `data/datasets/raw/<имя>/` и выполни:

```bash
python scripts/collect_faces.py          # вырезает лица
python scripts/register_faces.py         # запоминает (эмбеддинги)
```

### Шаг 3. Подключить камеры

Открой **`config/config.pc.yaml`** → раздел `cameras:` → замени ссылки:

```yaml
cameras:
  - name: "Камера-1 (двор)"
    source: "rtsp://ЛОГИН:ПАРОЛЬ@IP_КАМЕРЫ:554/stream1"   # <-- сюда
    enabled: true
```

- `ЛОГИН` / `ПАРОЛЬ` — учётка администратора камеры (найди в приложении камеры);
- `IP_КАМЕРЫ` — адрес камеры в домашней сети (Mi Home / Hik-Connect покажут);
- ссылку можно проверить в **VLC**: «Открыть сетевой поток»;
- без камеры — тест на видео: положи `data/test_video.mp4` или создай
  `python scripts/make_test_video.py` и поставь `source: "data/test_video.mp4"`.

### Шаг 4. Уведомления (куда писать)

В том же `config/config.pc.yaml`, раздел `notify`:

```yaml
notify:
  telegram:
    bot_token: "123456789:AA...свой_токен"   # от @BotFather (/newbot)
    chat_ids:                                 # КОМУ присылать — список аккаунтов
      - "111111111"                           # твой id (узнай у @userinfobot)
      - "222222222"                           # id мамы/сестры — людям тоже можно
  push:
    topic: "мой_секретный_topic"              # подпишись в приложении ntfy
```

### Шаг 5. Запуск

```bash
# ПРОЩЕ ВСЕГО — двойной клик по start_pc.bat  ИЛИ:
python main.py
#   main.py в корне сам берёт ПК-версию (config/config.pc.yaml),
#   открывает окно с видео и рамками и пишет видео в data/videos\

# Или вручную, любой конфиг:
python main.py --config config/config.vps.yaml     # серверная (без видео)
python main.py --self-test                          # проверка уведомлений
python main.py --preview                            # живое окно с рамками
python main.py --once                               # прогнать ~30 кадров

# Проверка, что ВСЁ работает (прогоняет все модули):
python scripts/self_check.py
```

**Где смотреть видео:** твоя версия пишет поток в папку
**`data/videos/Камера-1 (двор)/`** — там mp4-файлы (новый каждые 5 минут,
`recording.mode: "continuous"`). Хочешь клипы **только вокруг событий** —
поставь `recording.mode: "events"`.

**Где смотреть события:** `data/events/<камера>/<дата>/` — фото, которые
пришли в Telegram (и вырезки номеров). Достать из базы: `python scripts/show_events.py`.

---

## 4. Запуск на сервере VPS (в облаке, без видео)

### 4.1 Какие характеристики сервера

Купи VPS: **Ubuntu 22.04, 1 vCPU, 1 ГБ RAM, 10 ГБ SSD** (≈300 ₽/мес) на
Timeweb / VDSina / Aeza. Для 3+ камер — 4 vCPU / 4 ГБ.

### 4.2 Куда писать данные сервера и камер

Всё заполняется в **одном файле** — `config/config.vps.yaml`:

| Что вписать | Строка в конфиге | Откуда взять |
|---|---|---|
| IP и пароль камер | `cameras[].source` | приложение камеры (IP в сети) + VLC для проверки |
| Имена людей | `people.names` | сам |
| Токен бота | `notify.telegram.bot_token` | @BotFather |
| Аккаунты-получатели | `notify.telegram.chat_ids` | @userinfobot (ты, мама, сёстры...) |
| Push-topic | `notify.push.topic` | придумай + подпишись в ntfy |
| Удалять фото после отправки | `delete_images_after_send: true` | уже стоит в VPS-конфиге |

**IP/пароль самого сервера** — не в конфиге, а в GitHub-секретах (для авто-деплоя)
или в `scp/ssh` (для ручного): `SERVER_IP`, `SERVER_USER`, `SERVER_PASSWORD`.

> **Как сервер из интернета достучится до домашних камер?**
> 1. Самый простой старт: камера в той же домашней сети, что и сервер-комп —
>    но комп не всегда включён. Лучше: **WireGuard** на роутере/ПК дома —
>    тогда сервер и камеры «в одной сети», в RTSP пишется домашний IP камеры.
> 2. Либо проброс порта `554` на роутере на камеру → в RTSP пишется внешний
>    IP дома (⚠️ поставь надёжный пароль камеры).
> 3. Либо вообще без VPS: запусти **ПК-версию** дома (раздел 3).

### 4.3 Вариант А — автоматический деплой через GitHub (проще всего)

1. ```bash
   git init && git add . && git commit -m "MLCamera v1"
   git remote add origin https://github.com/ТВОЙ_ЛОГИН/ml_camera.git
   git push -u origin main
   ```
2. В GitHub: **Settings → Secrets and variables → Actions → New secret**:

   | Секрет | Значение |
   |---|---|
   | `SERVER_IP` | IP сервера (напр. `185.23.45.67`) |
   | `SERVER_USER` | `root` |
   | `SERVER_PASSWORD` | пароль от сервера |
   | `CONFIG_YAML` | **весь текст** `config/config.vps.yaml` (Ctrl+A, Ctrl+C, вставить) |

3. Пуш в `main` → само: тесты → сборка → деплой. Проверка:
   ```bash
   ssh root@IP_СЕРВЕРА 'docker ps && docker logs -f ml_camera'
   ```

### 4.4 Вариант Б — руками

```bash
python scripts/make_release_zip.py                      # dist/ml_camera_*.zip
scp dist/ml_camera_*.zip root@IP_СЕРВЕРА:/root/
ssh root@IP_СЕРВЕРА 'bash -s' < scripts/deploy_server.sh
# потом на сервере: nano ~/ml_camera/config/config.yaml  (вставь свои значения)
cd ~/ml_camera && docker compose -f docker-compose.prod.yml up -d --build
```

### 4.5 Заметки по VPS

- Видео на VPS **не пишется** (`recording.enabled: false`) — только события,
  фото и уведомления.
- Фото после отправки **удаляются** (`delete_images_after_send: true`) — диск не забивается.
- События: `~/ml_camera/data/events/`, база: `~/ml_camera/data/db/ml_camera.db`.

---

## 5. Ночное видение — как настроить

1. На камере включи **ИК-подсветку / режим «Авто»** (ночью картинка ч/б).
2. В конфиге стоит `camera.night_enabled: true` — тёмные кадры автоматически
   усиливаются (выравнивание гистограммы + шумоподавление), и лицо с номером
   распознаются в темноте.
3. Порог, с которого кадр считается «ночным» — `camera.night_brightness_threshold: 60`.
4. Для надёжности ночного распознавания добавь в датасет **2-5 ч/б фото** каждого
   члена семьи (см. `docs/DATASET.md`).

---

## 6. Структура папок

```
data/
├── datasets/raw/<имя>/      ← фото членов семьи (класть сюда)
├── datasets/faces/<имя>/    ← вырезанные лица (делает collect_faces)
├── datasets/auto_unknown/   ← лица незнакомцев (авто-обучение)
├── models/                  ← веса YOLO/марки
├── videos/<камера>/         ← ВИДЕО с камер (ПК-версия), mp4
├── events/<камера>/<дата>/  ← фото событий + вырезки номеров
├── db/ml_camera.db          ← база событий
└── logs/camera.log          ← логи
```

## 7. Документация

> 🚀 **Начни с `docs/НАЧАЛО.md`** — пошаговая инструкция для новичков.

- **docs/НАЧАЛО.md** — САМАЯ ГЛАВНАЯ: по шагам от нуля до работающей камеры
- **docs/RUNBOOK.md** — САМОЕ ВАЖНОЕ: что запускать, какие команды, что ещё можно сделать
- **docs/VPS_CAMERAS.md** — запуск с VPS: как найти IP камеры, будет ли камера доступна с сервера, как проверить
- **docs/SETUP.md** — настройка под ключ (RTSP, камеры, telegram, ntfy, фото)
- **docs/DATASET.md** — сколько фото каждого человека и как снимать
- **docs/DEPLOYMENT.md** — разворот на VPS: где какой секрет, команды
- **config/config.vps.example.yaml** — конфиг с подсказками «замени на...»
- **config/config.pc.yaml** / **config/config.vps.yaml** — две готовые версии
- **data/README.txt** — куда что класть в папке `data/`

## 8. Частые вопросы

| Вопрос | Ответ |
|---|---|
| «Не открывается источник» | проверь RTSP в VLC, включи RTSP в приложении камеры |
| Ночью тёмные снимки | включи ИК на камере, добавь ч/б фото в датасет |
| Не узнаёт человека | добавь фото, снизь `similarity_threshold` до 0.40 |
| Номер не читается | подними камеру под угол лучше, в ПК-версии включи `easyocr` |
| «Как добавить ещё человека?» | `python scripts/add_person.py Имя` + люди в `people.names` |
| «Куда пропали отправленные фото?» | на VPS они удаляются, `delete_images_after_send: true` |
| Спам-уведомления | подними `cooldown_minutes`; повторный проход того же человека/машины уже глушится трекингом (`tracking.enabled`) |
| «Номер то читается, то нет» | подними `car.plate_vote_frames` (голосование по кадрам) |
| «Как пометить свою машину?» | допиши номер в `car.known_plates` — будет «Своя машина» |
| «Что было ночью?» | утренний дайджест (`digest.enabled: true`) или `python scripts/show_events.py` |