# BRAND_RU — своя модель марок для российского рынка

## 1. Почему нужна своя модель

Готовая модель `data/models/car_brand.onnx` обучена на **Stanford Cars** —
рынок США 2012 года, 49 марок. В ней нет:

- Lada / ВАЗ, УАЗ, ГАЗ, Москвич;
- Kia, Hyundai, Skoda, Renault, Peugeot, Citroen, Dacia;
- большинства китайских брендов: Haval, Chery, Geely, Exeed, Omoda.

На типичных российских дворовых фото модель часто отвечает `unknown` или
ошибается. Поэтому для своего рынка нужен **свой маленький классификатор**,
обученный на фото именно ваших машин и ракурсов.

> Готовой качественной модели «марки РФ» с открытой лицензией, которую можно
> просто скачать, нет. Путь проекта — обучить свою на 10–30 марках, которые
> реально встречаются у вас во дворе. Это занимает меньше времени, чем кажется.

---

## 2. Что понадобится

| Компонент | Зачем |
|---|---|
| Python + `torch`, `torchvision` | обучение и экспорт ONNX |
| 100–300 фото на марку | качество |
| 10–30 марок | реалистичный набор для двора |
| 30–90 минут на CPU | обучение MobileNetV3-Small |

Установка обучения (отдельно от основных зависимостей):

```bash
pip install torch torchvision
```

Проверить GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Можно обучать и на CPU — модель маленькая.

---

## 3. Сбор датасета

### 3.1. Из уже сохранённых событий

Камера складывает фото машин в `data/events/<камера>/<дата>/car_*.jpg`.
Соберите их в одну папку:

```bash
python scripts/collect_car_images.py
# или:
python scripts/collect_car_images.py --source data/events --out data/datasets/cars/inbox
```

Скрипт убирает дубли по хешу и складывает фото в
`data/datasets/cars/inbox/`.

### 3.2. Разложить по маркам

```text
data/datasets/cars/
├── train/
│   ├── Lada/         100–300 фото
│   ├── Kia/
│   ├── Hyundai/
│   ├── Toyota/
│   ├── Renault/
│   ├── Skoda/
│   ├── Haval/
│   └── ...
├── val/              # необязательно
│   ├── Lada/
│   └── ...
└── inbox/            # сюда собирает collect_car_images.py
```

Правила:

- **одна папка = одна марка**, название папки попадает в уведомление;
- нужны разные ракурсы: перед, три четверти, бок, зад;
- разные цвета и освещение (день/ночь);
- машина должна занимать заметную часть кадра;
- не кладите в датасет чужие марки «на всякий случай» — это шум;
- избегайте сильного дисбаланса: если Lada 500 фото, а Kia 30 — Kia будет
  угадываться плохо.

> Если папки `val/` нет, скрипт сам отложит 20% фото на валидацию.
> Если есть — использует её как есть.

### 3.3. Сколько марок выбрать

Начните с 5–10 самых частых марок у вас во дворе. Меньше классов —
выше точность. Добавлять новые марки можно позже, переобучив модель.

---

## 4. Обучение

### 4.1. Проверить датасет

```bash
python scripts/train_brand_classifier.py --dry-run
```

Вывод покажет классы и количество фото. Если чего-то не хватает — скрипт
подскажет.

### 4.2. Быстрый старт (MobileNetV3-Small)

```bash
python scripts/train_brand_classifier.py --epochs 15
```

Результат:

```text
data/models/car_brand.onnx          # новая модель
data/models/car_brand_labels.json   # марки в порядке выходов
data/models/car_brand_meta.json     # preprocessing + метрики
data/models/car_brand.onnx.bak      # backup старой модели
data/models/car_brand.pt            # лучший checkpoint
```

Старая модель автоматически сохраняется в `.bak`.

### 4.3. Точнее (ConvNeXt-Tiny)

```bash
python scripts/train_brand_classifier.py \
  --arch convnext_tiny --epochs 20 --batch-size 8 --input-size 224
```

Медленнее, но обычно точнее на сложных ракурсах.

### 4.4. Полезные флаги

| Флаг | Значение |
|---|---|
| `--data` | папка датасета (по умолчанию `data/datasets/cars`) |
| `--arch` | `mobilenet_v3_small`, `resnet18`, `convnext_tiny`, `efficientnet_b0` |
| `--epochs` | число эпох |
| `--batch-size` | размер батча |
| `--lr` | learning rate (по умолчанию `3e-4`) |
| `--input-size` | вход (по умолчанию `224`) |
| `--val-split` | доля валидации (по умолчанию `0.2`) |
| `--device` | `auto`, `cpu`, `cuda` |
| `--no-pretrained` | обучение с нуля без весов ImageNet |
| `--output` | другой путь ONNX, чтобы не перезаписывать готовую модель |
| `--dry-run` | только показать датасет |

Пример «не трогать готовую модель»:

```bash
python scripts/train_brand_classifier.py \
  --output data/models/car_brand_ru.onnx
```

Тогда в конфиге:

```yaml
models:
  brand_onnx: "data/models/car_brand_ru.onnx"
  brand_labels: "data/models/car_brand_labels.json"
```

> Если задаёте свой `--output`, labels/meta по умолчанию кладутся **рядом**
> с ONNX. `car_analyzer.py` ищет `car_brand_labels.json` по пути
> `models.brand_labels`, а meta — рядом с ONNX.

---

## 5. Как это работает внутри

- **Архитектура:** MobileNetV3-Small / ResNet18 / ConvNeXt-Tiny / EfficientNet-B0
  с весами ImageNet, заменён последний слой на число марок.
- **Train-аугментации:** RandomResizedCrop, горизонтальное отражение,
  ColorJitter. Это важно для камеры: ракурсы и свет меняются.
- **Валидация:** Resize(224×1.14) → CenterCrop(224), ImageNet-нормализация.
- **Экспорт:** ONNX, dynamic batch, opset 17.
- **Meta:** `input_size`, `resize: shortest_center_crop`, `resize_ratio: 1.14`,
  `mean/std`, список классов, точность.

`src/car_analyzer.py` читает meta и применяет **то же самое** препроцессинг,
поэтому обученная модель работает без правок кода.

Марка по-прежнему считается **один раз на трек** по лучшей вырезке, а не
каждый кадр.

---

## 6. Настройка камеры

```yaml
car:
  brand_enabled: true
  brand_tta: true          # усреднение с отражением
  brand_min_conf: 0.35     # для своей модели можно 0.4–0.5

models:
  brand_onnx: "data/models/car_brand.onnx"
  brand_labels: "data/models/car_brand_labels.json"
```

Если модель часто отвечает `unknown`, снизьте `brand_min_conf` до `0.25`.
Если путает марки — поднимите до `0.45`.

Проверка:

```bash
python scripts/download_models.py --config config/config.yaml
python main.py --config config/config.yaml --once
```

В логе должно быть:

```text
Модель марки готова: 12 марок, вход 224x224
```

---

## 7. Как улучшить точность

1. **Больше фото редких марок.** 100+ на марку.
2. **Разные ракурсы.** Модель должна видеть машину не только сбоку.
3. **Реальные ночные кадры.** Добавьте 10–20% ночных фото.
4. **Чистые вырезки.** Если в кадре полмашины и много фона — уберите.
5. **Меньше классов на старте.** 8 марок по 150 фото лучше, чем 30 по 30.
6. **Проверяйте val accuracy.** Если < 0.7 — данных мало или классы
   визуально похожи (например, Haval и Chery в тёмное время).
7. **Дообучение.** Добавили новые фото — просто запустите скрипт снова.

---

## 8. Ограничения

- Модель знает **только те марки**, которые были в датасете. Всё остальное
  должно получать `unknown` — для этого и нужен `brand_min_conf`.
- Одна фотография = один автомобиль. Если в кадре две машины, YOLO даёт
  два трека и две вырезки — это нормально.
- Сильно тонированные/грязные машины, ночь, дождь — точность падает.
- Марка не влияет на распознавание номера и цвета: они работают отдельно.

---

## 9. Быстрая шпаргалка

```bash
# 1. собрать фото из событий
python scripts/collect_car_images.py

# 2. разложить по data/datasets/cars/train/<Марка>/

# 3. проверить
python scripts/train_brand_classifier.py --dry-run

# 4. обучить
python scripts/train_brand_classifier.py --epochs 15

# 5. включить в конфиге
#    car.brand_enabled: true
#    models.brand_onnx: "data/models/car_brand.onnx"

# 6. проверить на видео
python main.py --config config/config.yaml --once
```
