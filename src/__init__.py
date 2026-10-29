"""MLCamera — умная камера: люди, машины, животные и оставленные вещи.

Пакет src содержит все модули:
- config         загрузка конфига config/config.yaml
- camera         захват кадра (RTSP / веб-камера / файл)
- night_vision   улучшение ночного кадра
- detector       детекция объектов YOLO (люди, машины, животные, вещи)
- face_recognizer распознавание лиц (InsightFace / ArcFace)
- plate_recognizer OCR автомобильных номеров
- car_analyzer   цвет и марка автомобиля
- database       хранение событий (SQLite)
- notifier       уведомления (Telegram + ntfy push)
- pipeline       главный конвейер обработки
- web_panel      веб-панель истории событий
- main           точка входа
"""
__version__ = "1.1.0"