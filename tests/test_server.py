import time

from fastapi.testclient import TestClient

from cursebreaker.server import app

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


def test_favicon_route_never_500s():
    # 200 when a favicon file is present; 204 when it isn't — never a 404/500.
    r = client.get("/favicon.ico")
    assert r.status_code in (200, 204)
