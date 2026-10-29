"""Единые настройки: схема, проверка, сохранение в YAML и API панели."""
import json
import threading
import urllib.error
import urllib.request

import pytest
import yaml

from src.config import load_config
from src.database import Database
from src.settings import (FIELDS, SettingsError, normalize_cameras,
                          save_settings, settings_payload, validate_values)
from src.web_panel import create_server


def _write_cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "people": {"names": {"Тест": "Тест"}},
        "cameras": [{"name": "Ноутбук", "source": 0, "enabled": True}],
        "camera": {"width": 1280, "height": 720},
        "web": {"host": "127.0.0.1", "port": 8088},
        "database": {"path": str(tmp_path / "db.sqlite")},
        "storage": {"events_dir": str(tmp_path / "events")},
    }, allow_unicode=True), encoding="utf-8")
    return load_config(str(path))


def test_payload_has_values_for_every_field(tmp_path):
    cfg = _write_cfg(tmp_path)
    payload = settings_payload(cfg)
    assert set(payload["values"]) == {f["path"] for f in FIELDS}
    assert payload["config_path"].endswith("config.yaml")
    assert payload["stream"]["fps"] >= 1
    assert payload["cameras"][0]["source"] == 0


def test_validate_clamps_and_normalizes():
    clean = validate_values({
        "camera.fps": "999",
        "analysis.frame_scale": 0.01,
        "web.host": "  0.0.0.0 ",
        "notify.telegram.chat_ids": "1, 2;3",
        "digest.time": "7:05",
        "face.model_name": "buffalo_l",
    })
    assert clean["camera.fps"] == 60
    assert clean["analysis.frame_scale"] == 0.1
    assert clean["web.host"] == "0.0.0.0"
    assert clean["notify.telegram.chat_ids"] == ["1", "2", "3"]
    assert clean["digest.time"] == "07:05"
    assert clean["face.model_name"] == "buffalo_l"


def test_validate_rejects_bad_select_and_time():
    with pytest.raises(SettingsError):
        validate_values({"face.model_name": "нет-такой"})
    with pytest.raises(SettingsError):
        validate_values({"digest.time": "25:99"})


def test_normalize_cameras():
    cams = normalize_cameras([
        {"name": "", "source": "0", "enabled": "да"},
        {"source": "rtsp://cam/1"},
        "мусор",
    ])
    assert cams == [
        {"name": "Камера 1", "source": 0, "enabled": True},
        {"name": "Камера 2", "source": "rtsp://cam/1", "enabled": True},
    ]


def test_save_settings_writes_yaml_and_updates_memory(tmp_path):
    cfg = _write_cfg(tmp_path)
    save_settings(cfg, {"camera.fps": 12, "web.stream_fps": 5},
                  [{"name": "USB", "source": "0", "enabled": True}])

    raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    assert raw["camera"]["fps"] == 12
    assert raw["web"]["stream_fps"] == 5
    assert raw["cameras"][0]["source"] == 0
    # В памяти тот же объект — камеры увидят новые значения без перезагрузки.
    assert cfg.camera.fps == 12
    # Резервная копия старого файла.
    assert (tmp_path / "config.yaml.bak").exists()
    # Скалярные значения не потерялись.
    assert raw["people"]["names"]["Тест"] == "Тест"


def _post_json(url, payload, token=""):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-MLCamera": "1"}
    if token:
        url += "?token=" + token
        headers["X-Token"] = token
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


@pytest.fixture
def panel(tmp_path):
    cfg = _write_cfg(tmp_path)
    db = Database(cfg)
    server = create_server(cfg, db=db, host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield cfg, base
    server.shutdown()
    server.server_close()
    db.close()


def test_api_settings_get(panel):
    _, base = panel
    with urllib.request.urlopen(base + "/api/settings", timeout=5) as r:
        data = json.loads(r.read().decode("utf-8"))
    assert data["values"]["camera.fps"] == 30       # значение по умолчанию
    assert data["can_apply"] is False               # панель без камеры


def test_api_settings_post_saves_and_validates(panel):
    cfg, base = panel
    status, data = _post_json(base + "/api/settings", {
        "values": {"camera.fps": 24, "web.stream_fps": 6},
        "cameras": [{"name": "Ноутбук", "source": 0, "enabled": True}],
        "restart": False,
    })
    assert status == 200 and data["ok"] is True
    assert data["settings"]["values"]["camera.fps"] == 24
    assert cfg.camera.fps == 24

    status, data = _post_json(base + "/api/settings", {
        "values": {"digest.time": "99:99"},
    })
    assert status == 400 and data["ok"] is False
    assert "ЧЧ:ММ" in data["error"]


def test_api_settings_requires_csrf_header(panel):
    _, base = panel
    body = json.dumps({"values": {}}).encode()
    req = urllib.request.Request(base + "/api/settings", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 403
