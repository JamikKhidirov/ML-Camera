"""Переменные окружения для Docker: конфиг, панель и камера."""
import yaml

from src.config import load_config


def _write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "camera": {"fps": 30, "fallback_webcam": True},
        "web": {"host": "127.0.0.1", "port": 8088, "token": "",
                "open_browser": True},
        "cameras": [{"name": "Дом", "source": 0, "enabled": True}],
    }, allow_unicode=True), encoding="utf-8")
    return path


def test_web_and_camera_overrides(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("MLCAMERA_CONFIG", str(path))
    monkeypatch.setenv("MLCAMERA_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("MLCAMERA_WEB_PORT", "9090")
    monkeypatch.setenv("MLCAMERA_WEB_TOKEN", "secret")
    monkeypatch.setenv("MLCAMERA_OPEN_BROWSER", "0")
    monkeypatch.setenv("MLCAMERA_CAMERA_FPS", "15")
    monkeypatch.setenv("MLCAMERA_FALLBACK_WEBCAM", "0")

    cfg = load_config()
    assert str(cfg.config_path) == str(path)
    assert cfg.web.host == "0.0.0.0"
    assert cfg.web.port == 9090
    assert cfg.web.token == "secret"
    assert cfg.web.open_browser is False
    assert cfg.camera.fps == 15
    assert cfg.camera.fallback_webcam is False


def test_camera_source_replaces_list(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("MLCAMERA_CONFIG", str(path))
    monkeypatch.setenv("MLCAMERA_CAMERA_SOURCE",
                       "rtsp://admin:pass@192.168.1.10:554/stream1")
    monkeypatch.setenv("MLCAMERA_CAMERA_NAME", "Двор")

    cameras = load_config().get("cameras")
    assert len(cameras) == 1
    assert cameras[0]["name"] == "Двор"
    assert cameras[0]["source"].startswith("rtsp://")
    assert cameras[0]["enabled"] is True


def test_digit_source_becomes_webcam_index(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("MLCAMERA_CONFIG", str(path))
    monkeypatch.setenv("MLCAMERA_CAMERA_SOURCE", "1")

    cameras = load_config().get("cameras")
    assert cameras[0]["source"] == 1


def test_empty_env_is_ignored(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("MLCAMERA_CONFIG", str(path))
    monkeypatch.setenv("MLCAMERA_WEB_HOST", "")
    monkeypatch.setenv("MLCAMERA_CAMERA_FPS", "")

    cfg = load_config()
    assert cfg.web.host == "127.0.0.1"
    assert cfg.camera.fps == 30
