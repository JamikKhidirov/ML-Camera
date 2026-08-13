"""Уведомления: Telegram-бот + пуш-уведомления на Android (ntfy.sh).

- В Telegram уведомления можно слать СРАЗУ НЕСКОЛЬКИМ аккаунтам:
    notify.telegram.chat_ids: [111111, 222222]   (или chat_id для одного)
- Фото события отправляется вместе с текстом.
- Если notify.delete_images_after_send: true — после успешной отправки
  локальный файл фото удаляется (полезно на VPS, где места мало).
- ntfy — бесплатные push на Android: установи приложение и подпишись на topic.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from src.config import Config

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        token = str(cfg.notify.telegram.bot_token)
        self.tg_token = token if token and token != "PASTE_TELEGRAM_BOT_TOKEN" else ""
        self.chat_ids: list[str] = []
        chat_ids = cfg.notify.telegram.get("chat_ids", None)
        if isinstance(chat_ids, list) and chat_ids:
            self.chat_ids = [str(c) for c in chat_ids]
        elif str(cfg.notify.telegram.chat_id) not in ("", "PASTE_TELEGRAM_CHAT_ID"):
            self.chat_ids = [str(cfg.notify.telegram.chat_id)]
        self.tg_enabled = bool(self.tg_token and self.chat_ids)
        self.push_enabled = bool(cfg.notify.push.enabled and cfg.notify.push.topic)
        self.delete_images = bool(cfg.notify.get("delete_images_after_send", False))
        self._last_sent_at = 0.0
        self._min_interval = float(cfg.notify.min_interval_seconds)
        self._session = requests.Session()

    # ---------- общий гейт (анти-спам) ----------
    def _allowed(self) -> bool:
        now = time.time()
        if now - self._last_sent_at < self._min_interval:
            return False
        self._last_sent_at = now
        return True

    def send_event(self, title: str, message: str, image_path: Optional[str] = None,
                   extra_images: Iterable[str] = ()) -> bool:
        """Отправляет событие во все каналы. Возвращает True, если хоть один канал сработал."""
        if not self._allowed():
            return False
        if not (self.tg_enabled or self.push_enabled):
            return False
        log.info("Уведомление: %s — %s", title, message)
        sent = False
        if self.tg_enabled:
            try:
                self._send_telegram(title, message, image_path)
                sent = True
            except Exception as exc:
                log.error("Telegram: %s", exc)
        if self.push_enabled:
            try:
                self._send_ntfy(title, message, image_path)
                sent = True
            except Exception as exc:
                log.error("ntfy: %s", exc)
        # авто-удаление отправленных фото (чтобы не забивали диск)
        if sent and self.delete_images:
            self._delete_images([image_path] + list(extra_images))
        return sent

    # ---------- Telegram ----------
    def _send_telegram(self, title: str, message: str, image_path: Optional[str]) -> None:
        caption = f"<b>{title}</b>\n{message}"
        if image_path and Path(image_path).exists():
            for chat_id in self.chat_ids:
                with open(image_path, "rb") as f:
                    self._session.post(
                        f"{TELEGRAM_API}/sendPhoto".format(token=self.tg_token),
                        data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                        files={"photo": f},
                        timeout=30,
                    ).raise_for_status()
        else:
            for chat_id in self.chat_ids:
                self._session.post(
                    f"{TELEGRAM_API}/sendMessage".format(token=self.tg_token),
                    json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"},
                    timeout=30,
                ).raise_for_status()

    # ---------- ntfy push (Android) ----------
    def _send_ntfy(self, title: str, message: str, image_path: Optional[str]) -> None:
        server = str(self.cfg.notify.push.server).rstrip("/")
        topic = str(self.cfg.notify.push.topic)
        attach_url = None
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                r = self._session.put(
                    f"{server}/{topic}",
                    data=f,
                    headers={"Filename": Path(image_path).name},
                    timeout=60,
                )
                r.raise_for_status()
                try:
                    attach_url = r.json().get("url")
                except Exception:
                    attach_url = None
        payload = {"topic": topic, "title": title, "message": message, "priority": 5}
        if attach_url:
            payload["attach"] = attach_url
        self._session.post(server, json=payload, timeout=30).raise_for_status()

    # ---------- удаление фото ----------
    @staticmethod
    def _delete_images(paths: Iterable[Optional[str]]) -> None:
        for p in paths:
            if p and Path(p).exists():
                try:
                    os.remove(p)
                    log.info("Фото удалено после отправки: %s", p)
                except Exception as exc:
                    log.warning("Не удалось удалить фото %s: %s", p, exc)

    # ---------- проверка ----------
    def self_test(self) -> None:
        """Отправляет тестовое уведомление (запускается с --self-test)."""
        self.send_event("MLCamera — тест", "Привет! Уведомления настроены и работают 🎥")
        log.info("Тестовое уведомление отправлено в %d получателей (ТГ).", len(self.chat_ids))