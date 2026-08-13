"""Распознавание российских автомобильных номеров.

Регионы/серии используют кириллические буквы, выглядящие как латинские:
    А В Е К М Н О Р С Т У Х  ->  A B E K M H O P C T Y X

Поддерживаются два движка:
  tesseract — лёгкий (рекомендован для слабого VPS)
  easyocr   — точнее, но требует torch (~2 ГБ) и очень медленный на CPU
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
    return ""


class PlateRecognizer:
    """Распознаёт номер по вырезке автомобиля (или прямо по кадру)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = str(cfg.car.plate_engine).lower()
        self._readers: dict = {}
        if self.engine == "easyocr":
            self._init_easyocr()
        elif self.engine == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Неизвестный plate_engine: {self.engine}")

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

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """Возвращает (номер, уверенность). Пустая строка — номер не найден."""
        for version in (plate_crop, self.preprocess_plate(plate_crop)):
            text = self._ocr_text(version)
            cleaned = normalize_plate(text)
            if cleaned:
                return cleaned, 1.0
        return "", 0.0