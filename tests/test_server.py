import time

from fastapi.testclient import TestClient

from cursbreaker.server import app

client = TestClient(app)


def _wait_done(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_settings_hides_key_and_roundtrips():
    r = client.get("/api/settings").json()
    assert r["api_key_set"] is False
    assert "api_key" not in r
    assert r["api_key_hint"] == ""
    assert r["api_key_source"] is None

    r2 = client.post("/api/settings", json={"use_mock": True, "mode": "one_pass"}).json()
    assert r2["use_mock"] is True
    assert r2["mode"] == "one_pass"


def test_settings_exposes_key_hint_without_revealing_value():
    # Save a key
    r = client.post("/api/settings", json={"api_key": "AIzaSyABCDEFGHIJ_pretend_key_XYZ34"}).json()
    assert r["api_key_set"] is True
    assert "api_key" not in r          # raw key never leaves the server
    assert r["api_key_hint"].startswith("••••")
    assert r["api_key_hint"].endswith("XYZ34"[-4:])   # last 4 of the stored key
    assert r["api_key_source"] == "config"


def test_clear_api_key_endpoint():
    client.post("/api/settings", json={"api_key": "AIzaSy_keytoremove_ABCD"})
    assert client.get("/api/settings").json()["api_key_set"] is True
    r = client.delete("/api/settings/api_key").json()
    assert r["api_key_set"] is False
    assert r["api_key_hint"] == ""


def test_env_var_overrides_and_is_reported_as_source(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-set-key-ENDS")
    r = client.get("/api/settings").json()
    assert r["api_key_set"] is True
    assert r["api_key_source"] == "env"
    assert r["api_key_hint"].endswith("ENDS")


def test_full_flow_with_mock(png_path):
    client.post("/api/settings", json={"use_mock": True, "mode": "two_pass"})

    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    assert up["files"][0]["pages"] == 1

    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])
    assert status["status"] == "done"

    result = status["results"][0]
    assert result["error"] is None
    assert result["n_lines"] == 4

    txt = client.get(result["txt"])
    assert txt.status_code == 200
    assert b"mock transcription" in txt.content

    hocr = client.get(result["hocr"])
    assert hocr.status_code == 200
    assert b"ocr_line" in hocr.content

    preview = client.get(result["images"][0]["preview"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"

    # Searchable PDF download
    assert result["pdf"], "searchable PDF URL missing from job result"
    pdf = client.get(result["pdf"])
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"

    zipped = client.get(f"/api/download/{started['job_id']}.zip")
    assert zipped.status_code == 200
    assert zipped.content[:2] == b"PK"


def test_upload_rejects_unsupported_types(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_bytes(b"nope")
    with open(bad, "rb") as fh:
        r = client.post(
            "/api/upload", files={"files": ("notes.docx", fh, "application/octet-stream")}
        )
    assert r.status_code == 400


def test_process_requires_key_when_not_mock(png_path):
    client.post("/api/settings", json={"use_mock": False})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/process", json={"file_ids": [file_id]})
    assert r.status_code == 400


def test_download_unknown_job_is_404():
    r = client.get("/api/download/does-not-exist/whatever.txt")
    assert r.status_code == 404


def test_index_credits_mark_humphries_and_authorship():
    html = client.get("/").text
    assert "Mark Humphries" in html
    assert "Generative History" in html
    assert "generativehistory.substack.com" in html
    assert "John O'Connor" in html
    assert "Charlotte Mecklenburg Library" in html


def test_favicon_route_never_500s():
    # 200 when a favicon file is present; 204 when it isn't — never a 404/500.
    r = client.get("/favicon.ico")
    assert r.status_code in (200, 204)


def test_tesseract_status_endpoint_reports_availability():
    r = client.get("/api/tesseract")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body and isinstance(body["available"], bool)
    assert "languages" in body and isinstance(body["languages"], list)


def test_content_type_round_trips_through_settings_api():
    r = client.post("/api/settings", json={"content_type": "mixed", "tesseract_language": "eng"}).json()
    assert r["content_type"] == "mixed"
    assert r["tesseract_language"] == "eng"


def test_index_has_content_type_selector_and_tesseract_status():
    html = client.get("/").text
    # Three content-type radios are present.
    assert 'name="content_type"' in html
    for v in ("handwriting", "mixed", "text"):
        assert f'value="{v}"' in html
    # A visible status block + a place to pick a Tesseract language.
    assert 'id="tesseract-info"' in html
    assert 'id="tesseract_language"' in html


def test_heartbeat_endpoint_updates_timestamp():
    from cursbreaker import server
    server._LAST_PING_AT = None
    r = client.post("/api/heartbeat")
    assert r.status_code == 200
    assert server._LAST_PING_AT is not None


def test_heartbeat_bye_pulls_timestamp_back():
    from cursbreaker import server
    import time

    server._LAST_PING_AT = None
    before = time.time()
    client.post("/api/heartbeat?bye=true")
    # The bye signal moves the last-seen time into the past so the watchdog
    # fires soon, while still leaving a few seconds for a refresh.
    assert server._LAST_PING_AT is not None
    assert server._LAST_PING_AT < before + 0.5


def test_should_shutdown_predicate():
    from cursbreaker.server import _should_shutdown
    # No ping yet -> never shut down
    assert _should_shutdown(None, 10, now=100) is False
    # Recent ping -> stay up
    assert _should_shutdown(95, 10, now=100) is False
    # Stale ping -> shut down
    assert _should_shutdown(50, 10, now=100) is True
    # Job in flight pins the server alive even past the grace period
    assert _should_shutdown(50, 10, now=100, jobs_running=True) is False


def test_access_log_filter_drops_heartbeat_keeps_others():
    import logging

    from cursbreaker.server import install_access_log_filter

    install_access_log_filter()
    logger = logging.getLogger("uvicorn.access")
    our = next(
        f for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )

    fmt = '%s - "%s %s HTTP/%s" %d'
    hb = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/heartbeat", "1.1", 200),
        None,
    )
    hb_bye = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/heartbeat?bye=true", "1.1", 200),
        None,
    )
    upload = logger.makeRecord(
        "uvicorn.access", logging.INFO, "", 0, fmt,
        ("127.0.0.1:12345", "POST", "/api/upload", "1.1", 200),
        None,
    )

    assert our.filter(hb) is False
    assert our.filter(hb_bye) is False
    assert our.filter(upload) is True

    # Idempotent: calling twice doesn't stack duplicate filters.
    before = sum(
        1 for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )
    install_access_log_filter()
    after = sum(
        1 for f in logger.filters
        if getattr(f, "_cursbreaker_heartbeat", False)
    )
    assert before == after == 1
