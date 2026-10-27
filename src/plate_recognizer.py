"""Распознавание российских автомобильных номеров.

Регионы/серии используют кириллические буквы, выглядящие как латинские:
    А В Е К М Н О Р С Т У Х  ->  A B E K M H O P C T Y X

Поддерживаются три движка:
  fastalpr  — нейросеть: детектор номера (YOLOv9) + OCR (европейская модель). Точнее всего,
              ставится отдельно: python scripts/install_alpr.py
  easyocr   — точнее tesseract, но требует torch (~2 ГБ) и медленный на CPU
  tesseract — лёгкий (рекомендован для слабого VPS)
"""
from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from src.config import Config

log = logging.getLogger(__name__)

RU_TO_LATIN = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}
CYR = "".join(RU_TO_LATIN.keys())
# паттерн РФ: БУКВА 3 цифры БУКВА2 2-3 цифры (регион)
PLATE_RE = re.compile(r"([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX]{2})\s?(\d{2,3})")
_RF_FULL_RE = re.compile(r"^([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX]{2})(\d{2,3})$")

# Похожие символы: нейросеть часто путает 0/O, 1/T, 8/B и т.п.
# Буквы в номерах РФ — только ABEKMHOPCTYX.
_TO_LETTER = {"0": "O", "1": "T", "4": "A", "6": "G", "8": "B"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
             "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6", "T": "7",
             "B": "8", "P": "9"}


def _fix_rf_plate(text: str) -> str:
    """Пытается привести почти-номер к формату РФ, исправляя похожие символы.

    Пример: "CO51K119" -> "C051KT 19" (O->0 в цифровой позиции, 1->T в буквенной).
    """
    for length in (9, 8):
        for start in range(0, len(text) - length + 1):
            chunk = text[start:start + length]
            fixed = list(chunk)
            for i, ch in enumerate(chunk):
                if i in (0, 4, 5):          # буквенные позиции
                    if ch.isdigit():
                        fixed[i] = _TO_LETTER.get(ch, ch)
                else:                        # цифровые позиции
                    if ch.isalpha():
                        fixed[i] = _TO_DIGIT.get(ch, ch)
            candidate = "".join(fixed)
            m = _RF_FULL_RE.match(candidate)
            if m:
                return f"{m.group(1)}{m.group(2)}{m.group(3)} {m.group(4)}"
    return ""



def normalize_plate(text: str) -> str:
    """Чистит распознанный текст и приводит к виду A123BC 777."""
    text = re.sub(r"[^A-Za-z0-9А-Яа-я]", "", text).upper()
    out_chars = []
    for ch in text:
        if ch.isdigit() or ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            out_chars.append(ch)
        elif ch in RU_TO_LATIN:
            out_chars.append(RU_TO_LATIN[ch])
        else:
            continue
    out = "".join(out_chars)
    m = PLATE_RE.search(out)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)} {m.group(4)}"
    return _fix_rf_plate(out)


class PlateRecognizer:
    """Распознаёт номер по вырезке автомобиля (или прямо по кадру)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = str(cfg.car.plate_engine).lower()
        self._readers: dict = {}
        self._alpr = None
        self.uses_frame = False       # True: движок сам ищет номер в кадре
        self.last_crop: np.ndarray | None = None
        if self.engine in ("fastalpr", "fast-alpr", "alpr"):
            self.engine = "fastalpr"
            self._init_fastalpr()
        elif self.engine == "easyocr":
            self._init_easyocr()
        elif self.engine == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Неизвестный plate_engine: {self.engine}")

    def _init_fastalpr(self):
        try:
            from fast_alpr import ALPR
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Нет fast-alpr. Установи: python scripts/install_alpr.py "
                "(или оставь car.plate_engine: tesseract/easyocr)"
            ) from exc
        detector = str(self.cfg.car.get("alpr_detector", "yolo-v9-t-384-license-plate-end2end"))
        # европейская OCR-модель лучше читает кириллические буквы, похожие на латиницу
        ocr = str(self.cfg.car.get("alpr_ocr", "european-plates-mobile-vit-v2-model"))
        providers = list(self.cfg.car.get("alpr_providers") or ["CPUExecutionProvider"])
        # YOLOv9-детектор с динамическими осями сыпет безобидными warning'ами
        # MergeShapeInfo — гасим их только для сессий ALPR, не глобально.
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self._alpr = ALPR(
            detector_model=detector,
            ocr_model=ocr,
            detector_conf_thresh=float(self.cfg.car.get("alpr_conf", 0.35)),
            detector_providers=providers,
            detector_sess_options=sess_options,
            ocr_providers=providers,
            ocr_sess_options=sess_options,
        )
        self.uses_frame = True
        log.info("ALPR готов: детектор %s + OCR %s", detector, ocr)

    def _init_tesseract(self):
        try:
            import pytesseract
            self._readers["pytesseract"] = pytesseract
        except ImportError:
            raise RuntimeError(
                "Нет pytesseract. Установи: pip install pytesseract, "
                "и в систему tesseract-ocr + tesseract-ocr-rus"
            )

    def _init_easyocr(self):
        try:
            import easyocr
            self._readers["easyocr"] = easyocr.Reader(["en", "ru"], gpu=False)
        except ImportError:
            raise RuntimeError("Нет easyocr. Установи: pip install easyocr (тяжёлый!)")

    # ---------- подготовка вырезки ----------
    @staticmethod
    def car_crop(frame: np.ndarray, bbox, pad: float = 0.08) -> np.ndarray | None:
        """Область машины с небольшим запасом — для нейросетевого детектора номеров."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        # мелкий номер детектору труднее найти — слегка увеличим
        ch, cw = crop.shape[:2]
        if cw < 320:
            scale = 320.0 / max(1, cw)
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def plate_crop_from_car(frame, bbox, size_factor: float = 0.35) -> np.ndarray | None:
        """Вырезает нижнюю центральную часть автомобиля, где обычно номер."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        ch = y2 - y1
        zone_h = int(ch * size_factor)
        zone_y1 = y2 - zone_h
        cx1 = x1 + int((x2 - x1) * 0.15)
        cx2 = x2 - int((x2 - x1) * 0.15)
        crop = frame[zone_y1:y2, cx1:cx2]
        if crop is None or crop.size == 0:
            return None
        # увеличиваем, чтобы OCR лучше видел мелкие символы
        scale = max(2.0, 640.0 / max(crop.shape[:2]))
        new_w, new_h = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def preprocess_plate(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 30, 30)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # ---------- OCR ----------
    def _ocr_text(self, img: np.ndarray) -> str:
        if self.engine == "tesseract":
            tesseract = self._readers["pytesseract"]
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + CYR
            config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
            return tesseract.image_to_string(img, lang="rus+eng", config=config).strip()
        elif self.engine == "easyocr":
            reader = self._readers["easyocr"]
            res = reader.readtext(img, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + CYR)
            return "".join(t for _, t, conf in res if conf > 0.3)

    def read_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """Находит номер в кадре/области машины и читает его (FastALPR).

        Возвращает (номер, уверенность) и складывает вырезку номера в self.last_crop.
        Для tesseract/easyocr это эквивалент read() по нижней полосе машины.
        """
        self.last_crop = None
        if frame is None or frame.size == 0:
            return "", 0.0

        if self._alpr is None:
            h, w = frame.shape[:2]
            return self.read(PlateRecognizer.plate_crop_from_car(frame, (0, 0, w, h)))

        results = self._alpr.predict(frame)
        best: tuple[str, float, np.ndarray] | None = None
        h, w = frame.shape[:2]
        for res in results:
            text = normalize_plate(res.ocr.text if res.ocr is not None else "")
            if not text:
                continue
            conf_raw = getattr(res.ocr, "confidence", 0.0)
            if isinstance(conf_raw, (list, tuple, np.ndarray)):
                conf = float(np.mean(conf_raw)) if len(conf_raw) else 0.0
            else:
                conf = float(conf_raw or 0.0)
            if best is not None and conf <= best[1]:
                continue
            bb = res.detection.bounding_box
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = min(w, int(bb.x2)), min(h, int(bb.y2))
            crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            if crop is not None and crop.size:
                best = (text, conf, crop.copy())
        if best is None:
            return "", 0.0
        self.last_crop = best[2]
        return best[0], best[1]

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """Возвращает (номер, уверенность). Пустая строка — номер не найден."""
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        if self._alpr is not None:
            return self.read_frame(plate_crop)
        for version in (plate_crop, self.preprocess_plate(plate_crop)):
            text = self._ocr_text(version)
            cleaned = normalize_plate(text)
            if cleaned:
                return cleaned, 1.0
        return "", 0.0