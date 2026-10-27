# MODELS — все нейросети проекта: что, откуда, сколько весит

Коротко: **ничего обучать не нужно**. Все модели готовые, скачиваются скриптами.
Ниже — полный список, лицензии, размеры и ограничения.

---

## 1. Сводная таблица

| Задача | Модель | Лицензия | Размер | Где лежит |
|---|---|---|---|---|
| Люди и машины (детекция) | YOLOv8m (ПК) / YOLOv8n (VPS) | AGPL-3.0 | 49.7 / 6.2 МБ | `data/models/` |
| Лица: детекция + эмбеддинги | InsightFace `buffalo_s` (ПК/слабый CPU) или `buffalo_l` | код MIT, **веса — только некоммерческие исследования** | 158 МБ (`buffalo_s`) | `~/.insightface/models/` |
| Марка автомобиля | ConvNeXt-Tiny, Stanford Cars (49 марок) | Apache-2.0 | 106 МБ (ONNX) | `data/models/` |
| Поиск номера на фото | YOLOv9-t 384 license-plate end2end | MIT (пакет), веса из open-image-models | 7.4 МБ | `~/.cache/open-image-models/` |
| Чтение номера (OCR) | European plates MobileViT v2 | MIT | 14.5 МБ | `~/.cache/fast-plate-ocr/` |
| Движение в кадре | OpenCV MOG2 (не нейросеть) | Apache-2.0 | — | — |
| Цвет кузова | HSV-гистограмма (не нейросеть) | — | — | — |

Итого моделей на диске: **~340 МБ** (ПК-конфиг с `yolov8m` и `buffalo_s`).

---

## 2. Как скачать всё одной командой

```bash
# 1) зависимости
pip install -r requirements.txt
pip install -r requirements-pc.txt

# 2) YOLO + лица + марка + проверка OCR
python scripts/download_models.py --config config/config.pc.yaml

# 3) нейросетевые номера (FastALPR)
python scripts/install_alpr.py
```

На Windows всё это уже делает `pc\setup.bat` (5 шагов).

Проверить, что всё на месте:

```bash
python scripts/self_check.py
python scripts/download_models.py --config config/config.pc.yaml
```

---

## 3. Детекция людей и машин — YOLOv8

- **Что делает:** находит `person`, `car`, `truck`, `bus`, `motorcycle`.
- **Модель:** Ultralytics YOLOv8. На ПК — `yolov8m.pt` (точнее), на VPS — `yolov8n.pt` (быстрее).
- **Лицензия:** AGPL-3.0. Для домашнего/личного использования — ок; для продажи
  или закрытого сервиса нужна коммерческая лицензия Ultralytics.
- **Скачивание:** автоматически при первом запуске, либо
  `python scripts/download_models.py`.
- **Настройка:**

```yaml
car:
  detect_classes: ["car", "truck", "bus", "motorcycle"]
  conf_threshold: 0.35
models:
  yolov8: "data/models/yolov8m.pt"   # или yolov8n.pt
```

> Класс называется именно `motorcycle` (не `motorbike`).

---

## 4. Лица — InsightFace

- **Что делает:** находит лицо, строит эмбеддинг, сравнивает с базой семьи.
- **Модель:** `buffalo_s` (быстрая) или `buffalo_l` (точнее, тяжелее).
- **Лицензия:** код InsightFace — MIT. **Веса моделей — только для
  некоммерческих исследований.** Для домашней камеры это подходит; для
  коммерческого продукта нужно отдельное разрешение InsightFace.
- **Где лежит:** `~/.insightface/models/buffalo_s/` (скачивается сама).
- **Настройка:**

```yaml
face:
  model_name: "buffalo_s"     # или buffalo_l
  similarity_threshold: 0.45
  detection_size: [640, 640]
```

- **База лиц:** `data/db/embeddings_<model_name>.json`. У каждого размера
  модели своя база — нельзя смешивать `buffalo_s` и `buffalo_l`.

---

## 5. Марка автомобиля — ConvNeXt-Tiny (Stanford Cars)

- **Источник:** `ryosiswand/convnext_tiny-stanford-cars-brand` (Hugging Face).
- **Лицензия:** Apache-2.0.
- **Классов:** 49 марок. Вход 288×288, обучение с `Resize(288×1.14) → CenterCrop(288)`,
  нормализация ImageNet, TTA по горизонтальному отражению.
- **Точность:** 92.7% на тестовом наборе Stanford Cars.
- **Скачивание и экспорт в ONNX:**

```bash
python scripts/download_car_brand_model.py
# пересобрать: python scripts/download_car_brand_model.py --force
```

Скрипт кладёт в `data/models/`:

| Файл | Что внутри |
|---|---|
| `car_brand.onnx` | сама модель (106 МБ) |
| `car_brand_labels.json` | 49 марок в порядке выходов |
| `car_brand_meta.json` | размер входа, resize_ratio, mean/std, TTA |

- **Как работает в камере:** марка определяется **один раз на машину**, когда
  трек финализируется (машина уехала или набрала `plate_vote_frames` кадров).
  Используется лучшая по размеру и уверенности вырезка за трек — это и точнее,
  и дешевле, чем считать CNN каждый кадр.
- **Настройка:**

```yaml
car:
  brand_enabled: true
  brand_tta: true          # усреднение с отражением, +точность
  brand_min_conf: 0.25     # ниже порога -> "unknown", а не случайная марка
```

### Важное ограничение (честно)

Stanford Cars — это **рынок США 2012 года**. Модель отлично работает на
«своих» данных (Hummer, Acura, Ford и т.п.), но на типичных российских
дворовых фото и ракурсах ошибается чаще.

Чего в списке **нет вообще**: Kia, Lada/ВАЗ, Skoda, Renault, Peugeot,
Citroen, Dacia, УАЗ, ГАЗ. Такие машины получат `unknown` или неверную марку.

Что делать:

1. Оставить `brand_min_conf: 0.25–0.4` — тогда при неуверенности будет
   `unknown`, а не выдуманная марка.
2. Отключить марку: `brand_enabled: false` (на VPS так и стоит).
3. Для своих марок обучить маленький классификатор на своих фото:
   `scripts/train_brand_classifier.py` (см. `docs/DATASET.md`).

Цвет и номер от марки не зависят и работают всегда.

---

## 6. Номера — FastALPR (детектор + OCR)

Два нейросетевых шага:

1. **Детектор** `yolo-v9-t-384-license-plate-end2end` — находит прямоугольник
   номера в области машины.
2. **OCR** `european-plates-mobile-vit-v2-model` — читает символы.
   Европейская модель лучше понимает кириллические буквы, похожие на латиницу,
   чем глобальная `cct-s-v2-global-model`.

- **Лицензии:** `fast-alpr`, `open-image-models`, `fast-plate-ocr` — MIT.
- **Установка:**

```bash
python scripts/install_alpr.py            # поставить и скачать веса
python scripts/install_alpr.py --check    # только проверить
```

- **Почему не `pip install fast-alpr` напрямую:** пакет тянет
  `opencv-python-headless`, которая затирает GUI-сборку OpenCV и ломает окно
  предпросмотра (`cv2.imshow`). Скрипт ставит пакеты с `--no-deps` и при
  необходимости чинит OpenCV.
- **Постобработка под РФ:** результат OCR приводится к виду `A123BC 777`.
  Похожие символы исправляются по позициям формата:
  `CO51K119 → C051KT 19`, `C051KO199 → C051KO 199`.
- **Настройка:**

```yaml
car:
  plate_enabled: true
  plate_engine: "fastalpr"
  alpr_detector: "yolo-v9-t-384-license-plate-end2end"
  alpr_ocr: "european-plates-mobile-vit-v2-model"
  alpr_conf: 0.35
  alpr_providers: ["CPUExecutionProvider"]   # или DmlExecutionProvider
  plate_vote_frames: 5
```

Номер собирается голосованием по нескольким кадрам, а вырезка лучшего номера
сохраняется рядом с фото события.

---

## 7. Три движка номеров: что выбрать

| Движок | Точность на РФ | Скорость | Что нужно | Когда использовать |
|---|---|---|---|---|
| `fastalpr` | **лучшая** | средняя | `python scripts/install_alpr.py` | ПК, домашний сервер |
| `easyocr` | средняя | медленно на CPU | torch ~2 ГБ, `pip install easyocr` | запасной вариант |
| `tesseract` | слабая | быстрый | tesseract-ocr в системе | слабый VPS, Docker |

Переключение — одна строка:

```yaml
car:
  plate_engine: "fastalpr"   # или "easyocr", или "tesseract"
```

Если движок не установлен, камера не падает: номер просто не читается,
а в лог пишется подсказка, что поставить.

---

## 8. Что не является нейросетью

- **Движение** — OpenCV MOG2: сравнивает кадры, решает, стоит ли запускать
  тяжёлый анализ. Настраивается в `motion:`.
- **Цвет кузова** — гистограмма HSV по центральной зоне вырезки. Быстро и
  предсказуемо, но на ночных ч/б кадрах цвет будет `unknown`.
- **Трекинг** — CentroidTracker: связывает боксы между кадрами по расстоянию.
  Благодаря нему одна машина = одно уведомление, а не поток.

---

## 9. Куда какие веса кладутся

```
data/models/
├── yolov8m.pt              # YOLO (ПК)
├── yolov8n.pt              # YOLO (VPS)
├── car_brand.onnx          # марка
├── car_brand_labels.json
└── car_brand_meta.json

~/.insightface/models/buffalo_s/       # лица
~/.cache/open-image-models/            # детектор номеров
~/.cache/fast-plate-ocr/               # OCR номеров
```

`data/models/` и кэши моделей не коммитятся в git — они скачиваются скриптами.

---

## 10. Быстрая диагностика

| Симптом | Причина / что делать |
|---|---|
| `Марка: unknown` всегда | нет `data/models/car_brand.onnx` → `python scripts/download_car_brand_model.py` |
| Марка врёт на своих машинах | домен Stanford Cars, см. раздел 5 → поднять `brand_min_conf` или отключить |
| `Нет fast-alpr` | `python scripts/install_alpr.py` |
| `cv2.imshow` не работает | удалить `opencv-python-headless`, поставить `opencv-python` (это делает `install_alpr.py`) |
| Номер пустой | движок не установлен, номер мелкий/размытый, или `alpr_conf` слишком высокий |
| Номер читается по-разному | поднять `plate_vote_frames` (голосование), приблизить камеру |
