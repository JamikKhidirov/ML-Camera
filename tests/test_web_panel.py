"""Веб-панель: API, отдача фото, токен, защита путей и управление камерами."""
import json
import re
import threading
import time
import urllib.error
import urllib.request

import pytest

from src.camera_status import write_status
from src.config import Config
from src.database import Database
from src.web_panel import _HTML_PAGE, create_server


def _cfg(tmp_path, token="", control=True):
    return Config({
        "database": {"path": str(tmp_path / "web.db")},
        "storage": {"events_dir": str(tmp_path / "events")},
        "web": {"host": "127.0.0.1", "port": 0, "token": token,
                "title": "Тестовая панель", "control": control},
    })


def _start(cfg, manager=None):
    db = Database(cfg)
    server = create_server(cfg, db=db, host="127.0.0.1", port=0, manager=manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, db, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers, r.read()


def _post(url, headers=None):
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


class FakeManager:
    """Подмена CameraManager: запоминает вызовы и отдаёт готовые статусы."""

    def __init__(self):
        self.calls = []
        self.cameras = [{
            "index": 0, "name": "Двор", "enabled": True, "status": "running",
            "alive": True, "fps": 25.0, "proc_fps": 12.5, "frames": 10,
            "last_frame_ts": time.time(), "has_image": True, "error": "",
            "width": 1280, "height": 720, "source_kind": "rtsp",
            "recording": True, "recording_mode": "events",
            "started_at": time.time() - 60, "uptime": 60.0,
        }]

    def list(self):
        return self.cameras

    def start(self, index, max_frames=None):
        self.calls.append(("start", index))
        return True, ""

    def stop(self, index, timeout=5.0):
        self.calls.append(("stop", index))
        return True, ""

    def start_all(self, max_frames=None):
        self.calls.append(("start_all", None))

    def stop_all(self, timeout=5.0):
        self.calls.append(("stop_all", None))

    def snapshot(self, index, quality=85):
        if index != 0:
            return None
        return b"\xff\xd8\xff" + str(quality).encode() + b"x" * 1024


@pytest.fixture
def panel(tmp_path):
    cfg = _cfg(tmp_path)
    events = tmp_path / "events" / "cam1"
    events.mkdir(parents=True)
    (events / "dog.jpg").write_bytes(b"\xff\xd8\xffdogjpeg")
    (events / "car.jpg").write_bytes(b"\xff\xd8\xffcarjpeg")
    (events / "plate.jpg").write_bytes(b"\xff\xd8\xffplatejpeg")

    server, db, base = _start(cfg)
    db.save_face_event("Аня", 0.91, "cam1", str(events / "dog.jpg"))
    db.save_car_event("cam1", "A123BC 777", "Toyota", "white",
                      str(events / "car.jpg"), str(events / "plate.jpg"))
    db.save_object_event("cam1", "animal", "dog", "Собака", 0.88, 0,
                         str(events / "dog.jpg"), str(events / "dog.jpg"))
    db.save_object_event("cam1", "package", "backpack", "Рюкзак", 0.9, 65,
                         str(events / "dog.jpg"))
    yield base, db, tmp_path
    server.shutdown()
    server.server_close()
    db.close()


def test_html_and_health(panel):
    base, _, _ = panel
    status, headers, body = _get(base + "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert "Тестовая панель".encode() in body
    assert headers.get("Content-Security-Policy")

    status, _, body = _get(base + "/health")
    assert status == 200 and json.loads(body)["ok"] is True


def test_events_api_and_filters(panel):
    base, _, _ = panel
    _, _, body = _get(base + "/api/events?kind=animal&hours=24")
    data = json.loads(body)
    assert len(data["events"]) == 1
    ev = data["events"][0]
    assert ev["type"] == "animal" and ev["title"] == "Собака"
    assert ev["has_image"] is True
    assert "image_path" not in ev, "наружу не должны уходить пути к файлам"

    _, _, body = _get(base + "/api/events?kind=package")
    ev = json.loads(body)["events"][0]
    assert ev["type"] == "package"
    assert "65" in ev["subtitle"]

    _, _, body = _get(base + "/api/events?q=A123")
    assert json.loads(body)["events"][0]["type"] == "car"

    _, _, body = _get(base + "/api/events?kind=car&camera=cam1")
    assert json.loads(body)["events"][0]["title"] == "A123BC 777"


def test_stats_api(panel):
    base, _, _ = panel
    _, _, body = _get(base + "/api/stats?hours=24")
    stats = json.loads(body)
    assert stats["total"] == 4
    assert stats["faces"] == 1 and stats["cars"] == 1
    assert stats["animals"] == 1 and stats["packages"] == 1


def test_media_by_event_id(panel):
    base, _, _ = panel
    status, headers, body = _get(base + "/media?type=animal&id=1")
    assert status == 200
    assert headers["Content-Type"].startswith("image/")
    assert body.startswith(b"\xff\xd8\xff")

    status, _, body = _get(base + "/media?type=car&id=1&crop=1")
    assert status == 200 and b"platejpeg" in body

    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base + "/media?type=../../etc&id=1")
    assert err.value.code == 400

    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base + "/media?type=animal&id=999")
    assert err.value.code == 404


def test_media_cannot_escape_events_dir(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg)
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"secret")
    db.save_object_event("cam", "animal", "dog", "Собака", image_path=str(outside))
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/media?type=animal&id=1")
        assert err.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_token_auth(tmp_path):
    cfg = _cfg(tmp_path, token="s3cret")
    server, db, base = _start(cfg)
    db.save_object_event("cam", "animal", "cat", "Кошка")
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/events")
        assert err.value.code == 401

        status, headers, _ = _get(base + "/api/events?token=s3cret")
        assert status == 200
        assert "mlc_token=s3cret" in headers.get("Set-Cookie", "")

        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/events?token=wrong")
        assert err.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_cameras_api_with_manager(tmp_path):
    cfg = _cfg(tmp_path)
    manager = FakeManager()
    server, db, base = _start(cfg, manager=manager)
    try:
        _, _, body = _get(base + "/api/cameras")
        data = json.loads(body)
        assert data["controllable"] is True
        assert data["cameras"][0]["name"] == "Двор"
        assert data["cameras"][0]["status"] == "running"
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_camera_post_requires_header_and_origin(tmp_path):
    cfg = _cfg(tmp_path)
    manager = FakeManager()
    server, db, base = _start(cfg, manager=manager)
    try:
        status, body = _post(base + "/api/cameras/start?index=0")
        assert status == 403
        assert manager.calls == []

        status, body = _post(base + "/api/cameras/start?index=0",
                             {"X-MLCamera": "1",
                              "Origin": "http://evil.example"})
        assert status == 403
        assert manager.calls == []

        status, body = _post(base + "/api/cameras/start?index=0",
                             {"X-MLCamera": "1"})
        assert status == 200
        assert json.loads(body)["ok"] is True
        assert manager.calls == [("start", 0)]

        status, _ = _post(base + "/api/cameras/stop?index=0",
                          {"X-MLCamera": "1", "Origin": base})
        assert status == 200
        assert manager.calls[-1] == ("stop", 0)

        status, _ = _post(base + "/api/cameras/start_all",
                          {"X-MLCamera": "1"})
        assert status == 200 and manager.calls[-1] == ("start_all", None)

        status, _ = _post(base + "/api/cameras/stop_all",
                          {"X-MLCamera": "1"})
        assert status == 200 and manager.calls[-1] == ("stop_all", None)

        status, _ = _post(base + "/api/cameras/unknown?index=0",
                          {"X-MLCamera": "1"})
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_camera_snapshot_from_manager(tmp_path):
    cfg = _cfg(tmp_path)
    manager = FakeManager()
    server, db, base = _start(cfg, manager=manager)
    try:
        status, headers, body = _get(base + "/api/cameras/snapshot?index=0")
        assert status == 200
        assert headers["Content-Type"].startswith("image/jpeg")
        assert body.startswith(b"\xff\xd8\xff")

        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/cameras/snapshot?index=5")
        assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_cameras_read_only_without_manager(tmp_path):
    cfg = _cfg(tmp_path)
    write_status(cfg, {"updated_at": time.time(), "cameras": [{
        "index": 0, "name": "Двор", "enabled": True, "status": "running",
        "alive": True, "fps": 12.0, "frames": 3,
        "last_frame_ts": time.time(), "has_image": True, "error": "",
    }]})
    server, db, base = _start(cfg)
    try:
        _, _, body = _get(base + "/api/cameras")
        data = json.loads(body)
        assert data["controllable"] is False
        assert data["stale"] is False
        assert data["cameras"][0]["name"] == "Двор"
        assert data["cameras"][0]["has_image"] is False, \
            "отдельный процесс не может отдать кадр"

        status, body = _post(base + "/api/cameras/start?index=0",
                             {"X-MLCamera": "1"})
        assert status == 409
        assert "вместе с камерой" in json.loads(body)["error"]

        status, _ = _post(base + "/api/cameras/start_all",
                          {"X-MLCamera": "1"})
        assert status == 409

        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/cameras/snapshot?index=0")
        assert err.value.code == 409

        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/cameras/stream?index=0")
        assert err.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_camera_control_can_be_disabled(tmp_path):
    cfg = _cfg(tmp_path, control=False)
    manager = FakeManager()
    server, db, base = _start(cfg, manager=manager)
    try:
        _, _, body = _get(base + "/api/cameras")
        assert json.loads(body)["controllable"] is False

        status, _ = _post(base + "/api/cameras/stop?index=0",
                          {"X-MLCamera": "1"})
        assert status == 409
        assert manager.calls == []
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_cameras_stale_without_fresh_status(tmp_path):
    cfg = _cfg(tmp_path)
    write_status(cfg, {"updated_at": time.time() - 60, "cameras": []})
    server, db, base = _start(cfg)
    try:
        _, _, body = _get(base + "/api/cameras")
        data = json.loads(body)
        assert data["controllable"] is False
        assert data["stale"] is True
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_camera_snapshot_download_headers(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg, manager=FakeManager())
    try:
        _, headers, body = _get(
            base + "/api/cameras/snapshot?index=0&quality=70&download=1")
        assert body.startswith(b"\xff\xd8\xff") and b"70" in body[:8]
        assert headers["Content-Disposition"].startswith("attachment;")
        assert "mlcamera-0-" in headers["Content-Disposition"]

        _, headers, _ = _get(base + "/api/cameras/snapshot?index=0")
        assert headers["Content-Disposition"].startswith("inline;")
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_camera_stream_mjpeg(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg, manager=FakeManager())
    try:
        with urllib.request.urlopen(
                base + "/api/cameras/stream?index=0&fps=15", timeout=10) as r:
            assert r.headers["Content-Type"].startswith(
                "multipart/x-mixed-replace; boundary=")
            chunk = r.read(4096)
        assert b"Content-Type: image/jpeg" in chunk
        assert chunk.count(b"\xff\xd8\xff") >= 2
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_timeline_api(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg)
    try:
        db.save_face_event("Аня", 0.9, "Двор", "f.jpg")
        db.save_car_event("Двор", "A123BC 777", "Lada", "white", "c.jpg", "")
        db.save_object_event("Двор", "animal", "dog", "Собака", 0.9, 0, "d.jpg")

        _, _, body = _get(base + "/api/timeline?hours=24&buckets=12")
        data = json.loads(body)
        assert data["buckets"] == 12 and len(data["items"]) == 12
        assert sum(i["total"] for i in data["items"]) == 3
        assert data["max"] >= 1
        assert data["hours"] == 24
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_system_api(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg, manager=FakeManager())
    try:
        _, _, body = _get(base + "/api/system")
        data = json.loads(body)
        assert data["version"]
        assert data["uptime"] >= 0
        assert data["db_bytes"] > 0
        assert data["disk_total"] > 0
        assert data["cameras"] == {"total": 1, "running": 1, "errors": 0}
        assert data["controllable"] is True
        assert {"face", "car", "objects", "recording", "preview",
                "plate_engine"} <= set(data["features"])
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_cameras_include_ui_stats(tmp_path):
    cfg = _cfg(tmp_path)
    server, db, base = _start(cfg, manager=FakeManager())
    try:
        db.save_car_event("Двор", "A123BC", "Lada", "white", "c.jpg", "")
        _, _, body = _get(base + "/api/cameras")
        cam = json.loads(body)["cameras"][0]
        assert cam["proc_fps"] == 12.5
        assert cam["width"] == 1280 and cam["height"] == 720
        assert cam["source_kind"] == "rtsp"
        assert cam["recording"] is True
        assert cam["uptime"] == 60.0
        assert cam["events_24h"] == 1
        assert cam["last_event_ts"] > 0
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_html_js_selectors_are_valid():
    """Каждый $("#id") в скрипте должен указывать на существующий элемент.

    Составные селекторы вида $("#setCams .cam-row") проверяются по первому
    id — сам контейнер должен быть в разметке или создаваться в JS.
    """
    script = _HTML_PAGE.split("<script>", 1)[1]
    args = re.findall(r'\$\("([^"]*)"\)', script)
    bad = [arg for arg in args if not arg.startswith(("#", "."))]
    assert not bad, f"селектор без # или .: {bad}"
    ids = {arg[1:].split()[0].split(">")[0] for arg in args if arg.startswith("#")}
    # Часть контейнеров создаётся в JS: list.id = "setCams".
    dynamic = set(re.findall(r'\.id\s*=\s*"([^"]+)"', script))
    missing = sorted(i for i in ids
                     if f'id="{i}"' not in _HTML_PAGE and i not in dynamic)
    assert not missing, f"нет элементов с id: {missing}"

