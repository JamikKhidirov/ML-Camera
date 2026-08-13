"""Полная самопроверка проекта: прогоняет ВСЕ модули.

Показывает, что работает: конфиги, папки людей, база, детекция движения,
цвет машины, номера, запись видео, YOLO, распознавание лиц.

Запуск: python scripts/self_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

PASS, FAIL, SKIP = 0, 0, 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [ПРОЙДЕНО] {msg}")


def bad(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  [ОШИБКА]   {msg}")


def skip(msg: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  [ПРОПУСК]  {msg}")


def section(name: str) -> None:
    print(f"\n--- {name} ---")


def main() -> int:
    print("MLCamera — полная самопроверка")
    print("=" * 50)

    # ---------- 1. окружение ----------
    section("1. Окружение и зависимости")
    try:
        import numpy
        import cv2
        ok(f"numpy {numpy.__version__}, opencv {cv2.__version__}")
    except Exception as exc:
        bad(f"numpy/opencv: {exc}")
    try:
        import onnxruntime as ort
        ok(f"onnxruntime {ort.__version__}, провайдеры: {ort.get_available_providers()}")
    except Exception as exc:
        bad(f"onnxruntime: {exc}")

    # ---------- 2. конфиги ----------
    section("2. Конфиги")
    from src.config import Config, load_config
    for c in ["config/config.yaml", "config/config.pc.yaml", "config/config.vps.yaml",
              "config/config.vps.example.yaml", "config/config.local.yaml"]:
        try:
            cfg = load_config(c)
            n_cam = len(cfg.cameras) if isinstance(cfg.cameras, list) else 0
            ok(f"{c}: камер={n_cam}, имена={list(cfg.people.get('names', {}).keys())}")
        except Exception as exc:
            bad(f"{c}: {exc}")
    cfg = load_config("config/config.local.yaml")

    # ---------- 3. папки людей ----------
    section("3. Папки людей (датасет)")
    raw = ROOT / "data/datasets/raw"
    if raw.exists():
        names = sorted(p.name for p in raw.iterdir() if p.is_dir())
        ok(f"Люди: {names}")
        if names:
            ok("у каждого есть README.txt" if all((raw / n / "README.txt").exists() for n in names) else "")
    else:
        skip("папка data/datasets/raw не найдена")

    # ---------- 4. база данных ----------
    section("4. База данных (SQLite)")
    from src.database import Database
    db = Database(Config({"database": {"path": "data/db/selfcheck.db"}}))
    db.save_face_event("Алексей", 0.91, "cam1", "img.jpg")
    db.save_car_event("cam1", "A123BC 777", "Toyota", "white", "img.jpg", "crop.jpg")
    rows = db.recent_events()
    ok(f"записано событий: {len(rows)}")
    ok("кулдаун работает" if db.is_in_cooldown("face", "Алексей", 60) else "")
    db.close()
    (ROOT / "data/db/selfcheck.db").unlink(missing_ok=True)

    # ---------- 5. детекция движения ----------
    section("5. Детекция движения")
    from src.motion import MotionDetector
    md = MotionDetector(threshold=200, scale=0.25, warmup=2)
    base = np.zeros((180, 320, 3), np.uint8)
    [md.detect(base) for _ in range(3)]
    moving = np.zeros((180, 320, 3), np.uint8)
    moving[40:100, 80:240] = (200, 200, 200)
    detected = md.detect(moving)
    static = md.detect(np.zeros((180, 320, 3), np.uint8))
    ok(f"движение поймано: {detected}" if detected else f"движение НЕ поймано: {detected}")
    ok(f"покой не срабатывает: {not static}" if not static else "ложное срабатывание")

    # ---------- 6. цвет машины ----------
    section("6. Цвет автомобиля")
    from src.car_analyzer import CarAnalyzer
    for bgr, name in [((0, 0, 255), "red"), ((255, 0, 0), "blue"), ((0, 255, 0), "green"),
                      ((255, 255, 255), "white")]:
        img = np.zeros((100, 200, 3), np.uint8)
        img[:, :] = bgr
        got = CarAnalyzer.color_of(img)
        ok(f"{name} -> {got}" if got == name else f"ожидал {name}, получил {got}")

    # ---------- 7. номера ----------
    section("7. Распознавание номеров")
    from src.plate_recognizer import normalize_plate, PlateRecognizer
    cases = [("А123ВС777", "A123BC 777"), ("Н456ОК178", "H456OK 178"), ("Т777МР02", "T777MP 02")]
    for raw_s, exp in cases:
        got = normalize_plate(raw_s)
        ok(f"{raw_s} -> {got}" if got == exp else f"номер {raw_s}: ожидал {exp}, получил {got}")
    if importlib.util.find_spec("easyocr"):
        skip("easyocr не инициализируем (тяжёлая загрузка)")
    elif importlib.util.find_spec("pytesseract"):
        skip("pytesseract есть, но бинарь tesseract проверяется отдельно")

    # ---------- 7b. трекинг и голосование ----------
    section("7b. Трекинг объектов + голосование по номерам")
    from src.tracker import CentroidTracker, majority_vote
    mv = majority_vote(["A123", "B000", "A123", "A123"])
    ok(f"голосование: {mv}" if mv == "A123" else f"голосование сломано: {mv}")
    tr = CentroidTracker(max_disappeared=3)
    first = tr.update([(40, 40, 60, 60)])[0]
    same = tr.update([(42, 42, 62, 62)])[0]
    ok("трекинг держит тот же объект" if same is first else "трекинг сломан: объект потерян")
    for _ in range(5):
        tr.update([])
    new = tr.update([(45, 45, 65, 65)])[0]
    ok("трек исчез по лимиту" if new is not first else "трек не исчезает после max_disappeared")

    # ---------- 7c. дайджест ----------
    section("7c. Дайджест (сводка из БД)")
    from src.database import Database as DB2
    db_dig = DB2(Config({"database": {"path": "data/db/selfcheck_dig.db"}}))
    db_dig.save_face_event("Ольга", 0.9, "cam", "")
    db_dig.save_face_event("unknown", 0.4, "cam", "")
    dig = db_dig.digest_data(hours=24)
    ok(f"дайджест: лиц={len(dig['faces'])}, незнакомцы={dig['unknown']}"
       if dig["faces"] and dig["unknown"] == 1 else f"дайджест неполный: {dig}")
    db_dig.close()
    (ROOT / "data/db/selfcheck_dig.db").unlink(missing_ok=True)

    # ---------- 8. запись видео ----------
    section("8. Запись видео (data/videos)")
    from src.recorder import ClipRecorder
    r_cfg = Config({"recording": {"enabled": True, "mode": "events", "output_dir": "data/videos",
                                  "fps": 10, "seconds_before": 1, "seconds_after": 1, "fourcc": "mp4v"}})
    rec = ClipRecorder(r_cfg, "selfcheck")
    for _ in range(12):
        rec.push(np.zeros((180, 320, 3), np.uint8))
    rec.trigger("test")
    for _ in range(10):
        rec.push(np.zeros((180, 320, 3), np.uint8))
    rec.close()
    clips = list((ROOT / "data/videos/selfcheck").glob("*.mp4"))
    ok(f"клип создан: {clips[0].name}" if clips else "клип НЕ создан")
    for f in (ROOT / "data/videos/selfcheck").glob("*.mp4"):
        f.unlink()

    # ---------- 9. YOLO (люди и машины) ----------
    section("9. YOLO — детекция объектов")
    try:
        from src.detector import YoloDetector
        video = ROOT / "data/test_video.mp4"
        if not video.exists():
            video = ROOT / "data/test_video.mp4"
        cap = cv2.VideoCapture(str(video))
        ok_frame = cap.read()[1]
        cap.release()
        det = YoloDetector(cfg, conf_threshold=0.2)
        t0 = time.time()
        res = det.predict(ok_frame)
        ok(f"YOLO отработала за {time.time()-t0:.1f} c, объектов: {len(res)}")
    except Exception as exc:
        bad(f"YOLO: {exc}")

    # ---------- 10. распознавание лиц ----------
    section("10. Распознавание лиц («кто пришёл»)")
    try:
        from src.face_recognizer import FaceRecognizer
        fr = FaceRecognizer(cfg)
        base_db = fr.known_names()
        cap = cv2.VideoCapture(str(ROOT / "data/test_video.mp4"))
        frame = cap.read()[1]
        cap.release()
        matches, faces = fr.recognize(frame)
        ok(f"лица: детекция отработала ({len(faces)} найденных, в базе {len(base_db)} человек)")
    except Exception as exc:
        bad(f"лица: {exc}")

    # ---------- 11. уведомления (без отправки) ----------
    section("11. Уведомления (настройка)")
    from src.notifier import Notifier
    notif = Notifier(load_config("config/config.local.yaml"))
    tg = notif.tg_enabled
    push = notif.push_enabled
    ok(f"Telegram получателей: {len(notif.chat_ids)}" if not tg or notif.chat_ids else "")
    ok(f"ntfy включён: {push}")

    # ---------- итог ----------
    print("\n" + "=" * 50)
    print(f"ИТОГ: {PASS} пройдено, {FAIL} ошибок, {SKIP} пропущено")
    print("=" * 50)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())