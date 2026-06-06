import time

import pytest
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


@pytest.fixture
def run_with_mock(monkeypatch):
    """Run real jobs with the deterministic MockProvider -- no network or live
    key. Demo mode no longer exists, so tests opt into the mock explicitly by
    stubbing the provider factory and storing a dummy key so the 'no key' guard
    passes (the dummy is never validated, since make_provider is replaced)."""
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    monkeypatch.setattr(server, "make_provider", lambda s: MockProvider())
    client.post("/api/settings", json={"api_key": "AIza_test_dummy_key_ABCDEF"})
    return MockProvider


@pytest.fixture
def run_with_slow_mock(monkeypatch):
    """Like run_with_mock but each Gemini call sleeps briefly, so a job stays
    'running' long enough to be cancelled deterministically."""
    import time as _time
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    class _Slow(MockProvider):
        def transcribe_text(self, *a, **k):
            _time.sleep(0.2); return super().transcribe_text(*a, **k)

        def detect_lines(self, *a, **k):
            _time.sleep(0.2); return super().detect_lines(*a, **k)

        def transcribe_with_boxes(self, *a, **k):
            _time.sleep(0.2); return super().transcribe_with_boxes(*a, **k)

    monkeypatch.setattr(server, "make_provider", lambda s: _Slow())
    client.post("/api/settings", json={"api_key": "AIza_test_dummy_key_ABCDEF"})
    return _Slow


def test_settings_hides_key_and_roundtrips():
    r = client.get("/api/settings").json()
    assert r["api_key_set"] is False
    assert "api_key" not in r
    assert r["api_key_hint"] == ""
    assert r["api_key_source"] is None

    r2 = client.post("/api/settings", json={"mode": "one_pass"}).json()
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


def test_full_flow(run_with_mock, png_path):
    client.post("/api/settings", json={"mode": "two_pass"})

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


def test_process_requires_key(png_path):
    # Isolated config has no stored key and no ambient env key -> 400.
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


def test_index_explains_how_to_get_an_api_key():
    # Non-technical users (GLAM staff) need an in-app pointer to creating a key,
    # not just an empty field. The collapsed help links to AI Studio, names the
    # create step, and reassures that there's a free tier.
    html = client.get("/").text
    assert "aistudio.google.com" in html
    assert "Create API key" in html
    assert "free" in html.lower()


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
    # Richer diagnostics so the UI can explain *which* piece is missing.
    for key in ("wrapper_present", "binary_found", "cmd_path", "version",
                "error", "install_hint"):
        assert key in body
    assert isinstance(body["wrapper_present"], bool)
    assert isinstance(body["binary_found"], bool)


def test_content_type_round_trips_through_settings_api():
    r = client.post(
        "/api/settings", json={"content_type": "handwriting", "tesseract_language": "eng"}
    ).json()
    assert r["content_type"] == "handwriting"
    assert r["tesseract_language"] == "eng"


def test_refine_word_boxes_round_trips_through_settings_api():
    r = client.post("/api/settings", json={"refine_word_boxes": True}).json()
    assert r["refine_word_boxes"] is True


def test_legacy_mixed_content_type_migrates_to_handwriting_plus_refine():
    # "mixed" was retired; posting it (e.g. from an old client) must migrate to
    # the handwriting flow with word-box refinement on, not persist "mixed".
    r = client.post("/api/settings", json={"content_type": "mixed"}).json()
    assert r["content_type"] == "handwriting"
    assert r["refine_word_boxes"] is True


def test_index_has_content_type_selector_and_tesseract_status():
    html = client.get("/").text
    # The two content-type radios are present; "mixed" was retired in favor of
    # the refine-word-positions toggle.
    assert 'name="content_type"' in html
    for v in ("handwriting", "text"):
        assert f'value="{v}"' in html
    assert 'value="mixed"' not in html
    assert 'id="refine_word_boxes"' in html
    # A visible status block + a place to pick a Tesseract language.
    assert 'id="tesseract-info"' in html
    assert 'id="tesseract_language"' in html


def test_demo_mode_is_gone():
    # The user-facing demo/mock mode was removed entirely.
    html = client.get("/").text
    assert 'id="use_mock"' not in html
    assert "Demo mode" not in html
    # ...and it isn't a setting the API exposes or accepts.
    settings = client.get("/api/settings").json()
    assert "use_mock" not in settings
    client.post("/api/settings", json={"use_mock": True})
    assert "use_mock" not in client.get("/api/settings").json()


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


def _make_access_record(status: int) -> "logging.LogRecord":
    import logging
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname="", lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", "GET", "/x", "1.1", status),
        exc_info=None,
    )


def test_pretty_access_formatter_marks_status_classes():
    from cursbreaker.server import PrettyAccessFormatter
    fmt = PrettyAccessFormatter(use_colors=False)
    assert " ok " in fmt.format(_make_access_record(200))
    assert " ok " in fmt.format(_make_access_record(304))
    assert "warn" in fmt.format(_make_access_record(404))
    assert " err" in fmt.format(_make_access_record(500))
    # The leading "INFO" prefix from the default formatter is gone.
    assert "INFO" not in fmt.format(_make_access_record(200))


def test_pretty_access_formatter_emits_ansi_when_colors_on():
    from cursbreaker.server import PrettyAccessFormatter
    out = PrettyAccessFormatter(use_colors=True).format(_make_access_record(200))
    assert "\x1b[32m" in out and "\x1b[0m" in out  # green + reset

    out = PrettyAccessFormatter(use_colors=True).format(_make_access_record(500))
    assert "\x1b[31m" in out and "\x1b[0m" in out  # red + reset

    # use_colors=False should leave the line free of escape codes.
    out = PrettyAccessFormatter(use_colors=False).format(_make_access_record(200))
    assert "\x1b[" not in out


def test_pretty_access_formatter_falls_back_for_non_access_records():
    import logging
    from cursbreaker.server import PrettyAccessFormatter
    record = logging.LogRecord(
        name="uvicorn.error", level=logging.INFO, pathname="", lineno=0,
        msg="some lifecycle message", args=None, exc_info=None,
    )
    out = PrettyAccessFormatter(use_colors=False).format(record)
    assert "some lifecycle message" in out


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


def test_key_status_no_key_by_default():
    # Fresh isolated config + cleared env -> nothing stored.
    assert client.get("/api/key-status").json()["state"] == "no_key"


def test_key_status_reports_invalid_revoked_key(monkeypatch):
    from cursbreaker import gemini_client

    monkeypatch.setenv("GEMINI_API_KEY", "revoked-key")

    def boom(key):
        e = Exception("400 API key not valid. Please pass a valid API key.")
        e.code = 400
        raise e

    monkeypatch.setattr(gemini_client, "_probe_models", boom)
    r = client.get("/api/key-status").json()
    assert r["state"] == "invalid"
    assert r["message"]


# --- token usage + cost estimate ----------------------------------------- #

def test_job_status_includes_token_fields(run_with_mock, png_path):
    client.post("/api/settings", json={"mode": "two_pass"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])

    assert "tokens" in status
    for k in ("input", "output", "thinking", "total", "calls", "cost"):
        assert k in status["tokens"]
    # MockProvider makes no real call, so nothing is billed.
    assert status["tokens"]["calls"] == 0
    assert status["results"][0]["tokens"]["total"] == 0
    # The internal provider handle must never be serialized to the client.
    assert "_provider" not in status


def test_append_capped_keeps_most_recent():
    from cursbreaker.server import _append_capped
    log = []
    for i in range(10):
        _append_capped(log, f"line {i}", cap=4)
    assert log == ["line 6", "line 7", "line 8", "line 9"]


def test_job_status_exposes_activity_log_and_unit_counters(run_with_mock, pdf_path):
    client.post("/api/settings", json={"mode": "two_pass"})
    with open(pdf_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("doc.pdf", fh, "application/pdf")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    status = _wait_done(started["job_id"])

    assert status["status"] == "done"
    # Page-driven bar units: the 2-page fixture -> 2 of 2.
    assert status["total_units"] == 2
    assert status["done_units"] == 2
    # Verbose activity log is present and captures the real steps.
    log = status["log"]
    assert isinstance(log, list) and log
    joined = "\n".join(log)
    assert "2 page(s) to transcribe" in joined
    assert "Page 1/2" in joined and "Page 2/2" in joined
    assert "Writing outputs" in joined
    assert isinstance(status["current"], str) and status["current"]
    assert "stage" in status
    # The private cancel flag is never serialized to the client.
    assert "_cancel" not in status


def test_cancel_running_job(run_with_slow_mock, pdf_path):
    client.post("/api/settings", json={"mode": "two_pass"})  # 2 calls/page -> slow
    with open(pdf_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("doc.pdf", fh, "application/pdf")}
        ).json()
    file_id = up["files"][0]["id"]
    started = client.post("/api/process", json={"file_ids": [file_id]}).json()
    jid = started["job_id"]
    r = client.post(f"/api/jobs/{jid}/cancel").json()
    assert r["cancelling"] is True
    # The cancellation notice is in the activity log immediately -- before the
    # worker reaches a cancel boundary (it's still mid-call).
    interim = client.get(f"/api/jobs/{jid}").json()
    assert any("Cancellation requested" in line for line in interim["log"])
    status = _wait_done(jid)
    assert status["status"] == "cancelled"
    # ...and the final "Cancelled." line is there once it actually stops.
    assert any(line == "Cancelled." for line in status["log"])
    # The app is still alive and serving after a cancel.
    assert client.get("/").status_code == 200


def test_cancel_unknown_job_is_404():
    assert client.post("/api/jobs/does-not-exist/cancel").status_code == 404


def test_estimate_not_billable_for_printed_only(png_path):
    # Printed-only runs locally (Tesseract), so there's no Gemini token cost.
    client.post("/api/settings", json={"content_type": "text"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]}).json()
    assert r["billable"] is False
    assert "Printed-only" in r["reason"]
    assert r["total_low"] == 0 and r["total_high"] == 0
    assert r["cost_low"] is None and r["cost_high"] is None


def test_estimate_requires_key_when_billable(png_path):
    client.post("/api/settings", json={"content_type": "handwriting"})
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]})
    assert r.status_code == 400  # no key -> can't estimate Gemini cost


def test_estimate_billable_with_fake_provider(monkeypatch, png_path):
    from cursbreaker import server
    from cursbreaker.gemini_client import MockProvider

    class _P(MockProvider):
        def count_input_tokens(self, image_png, mime="image/png"):
            return 1000

    client.post(
        "/api/settings",
        json={
            "content_type": "handwriting",
            "mode": "one_pass",
            "transcription_model": "gemini-3.1-flash-lite",  # $0.25 in / $1.50 out
            "api_key": "AIza_estimate_test_key_WXYZ",
        },
    )
    monkeypatch.setattr(server, "make_provider", lambda s: _P())
    with open(png_path, "rb") as fh:
        up = client.post(
            "/api/upload", files={"files": ("sample.png", fh, "image/png")}
        ).json()
    file_id = up["files"][0]["id"]
    r = client.post("/api/estimate", json={"file_ids": [file_id]}).json()
    assert r["billable"] is True
    assert r["input"] == 1000          # 1 page * 1 call * 1000 input tokens
    # Cost is a range derived from the model's published price; one-pass output
    # range is 1800..5400 tokens/page.
    expected_low = 1000 / 1_000_000 * 0.25 + 1800 / 1_000_000 * 1.50
    expected_high = 1000 / 1_000_000 * 0.25 + 5400 / 1_000_000 * 1.50
    assert r["cost_low"] == pytest.approx(expected_low)
    assert r["cost_high"] == pytest.approx(expected_high)
    assert r["model"] == "gemini-3.1-flash-lite"
    assert r["price_input_per_mtok"] == 0.25


def test_estimate_no_staged_files_is_400():
    r = client.post("/api/estimate", json={"file_ids": ["nope"]})
    assert r.status_code == 400


def test_models_endpoint_returns_priced_catalog():
    body = client.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    # Pro is first (the dropdown's default position + the saved default model).
    assert ids[0] == "gemini-3.1-pro-preview"
    assert "gemini-3.5-flash" in ids
    assert "gemini-3.1-flash-lite" in ids
    assert body["prices_as_of"]            # shown in the UI for transparency
    flash = next(m for m in body["models"] if m["id"] == "gemini-3.5-flash")
    assert flash["input_per_mtok"] == 1.50 and flash["output_per_mtok"] == 9.00
    pro = next(m for m in body["models"] if m["id"] == "gemini-3.1-pro-preview")
    assert pro["tier_threshold"] == 200_000   # tiered pricing is exposed


def test_model_choice_round_trips_through_settings_api():
    r = client.post(
        "/api/settings", json={"transcription_model": "gemini-3.5-flash"}
    ).json()
    assert r["transcription_model"] == "gemini-3.5-flash"


def test_detection_model_follows_transcription_model():
    # The single picker is enforced server-side: posting only the transcription
    # model keeps detection (two-pass) on the same model, so the priced/reported
    # model can't drift from the one detection actually uses.
    r = client.post(
        "/api/settings", json={"transcription_model": "gemini-3.1-flash-lite"}
    ).json()
    assert r["transcription_model"] == "gemini-3.1-flash-lite"
    assert r["detection_model"] == "gemini-3.1-flash-lite"


def test_index_has_cost_controls():
    html = client.get("/").text
    assert 'id="estimate"' in html
    assert 'id="model"' in html             # curated dropdown replaces free text
    assert 'id="token-text"' in html
    # The manual price inputs are gone; pricing is automatic now.
    assert 'id="price_input_per_mtok"' not in html
    assert 'id="price_output_per_mtok"' not in html
