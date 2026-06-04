import pytest

from cursbreaker import gemini_client
from cursbreaker.config import Settings
from cursbreaker.gemini_client import (
    GeminiProvider,
    _is_auth_error,
    _is_model_unavailable,
    check_api_key,
)


class _Err(Exception):
    """Stand-in for a google.genai ClientError carrying a status code."""

    def __init__(self, code, message):
        super().__init__(f"{code} {message}")
        self.code = code
        self.message = message


def test_is_auth_error_flags_bad_keys():
    assert _is_auth_error(_Err(400, "API key not valid. Please pass a valid API key."))
    assert _is_auth_error(_Err(403, "PERMISSION_DENIED"))
    assert _is_auth_error(_Err(401, "unauthorized"))


def test_is_auth_error_ignores_transient_failures():
    # A valid-but-throttled or offline key must never be reported as invalid.
    assert not _is_auth_error(_Err(429, "RESOURCE_EXHAUSTED"))
    assert not _is_auth_error(_Err(503, "service unavailable"))
    assert not _is_auth_error(ConnectionError("network down"))


def test_check_api_key_no_key():
    # isolated_config fixture clears ambient GEMINI_API_KEY/GOOGLE_API_KEY.
    assert check_api_key(Settings()).state == "no_key"


def test_check_api_key_valid(monkeypatch):
    monkeypatch.setattr(gemini_client, "_probe_models", lambda key: None)
    assert check_api_key(Settings(api_key="k")).state == "valid"


def test_check_api_key_invalid(monkeypatch):
    def boom(key):
        raise _Err(400, "API key not valid. Please pass a valid API key.")

    monkeypatch.setattr(gemini_client, "_probe_models", boom)
    st = check_api_key(Settings(api_key="revoked"))
    assert st.state == "invalid"
    assert st.message  # a human-readable explanation is provided


def test_check_api_key_unknown_on_transient(monkeypatch):
    def boom(key):
        raise ConnectionError("network down")

    monkeypatch.setattr(gemini_client, "_probe_models", boom)
    # Never cry wolf: an ambiguous failure is "unknown", not "invalid".
    assert check_api_key(Settings(api_key="k")).state == "unknown"


# --- model-unavailable detection + fallback ------------------------------- #

def test_is_model_unavailable_flags_retired_models():
    assert _is_model_unavailable(_Err(404, "models/gemini-3-pro-preview is no longer available"))
    assert _is_model_unavailable(_Err(404, "NOT_FOUND"))
    # message-only signal (no code attribute)
    assert _is_model_unavailable(Exception("models/foo is not found for API version v1beta"))


def test_is_model_unavailable_ignores_other_errors():
    assert not _is_model_unavailable(_Err(403, "PERMISSION_DENIED"))   # key/access
    assert not _is_model_unavailable(_Err(400, "INVALID_ARGUMENT: bad image"))
    assert not _is_model_unavailable(ConnectionError("network down"))


class _FakeModels:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        return self._behavior(model)


class _FakeClient:
    def __init__(self, behavior):
        self.models = _FakeModels(behavior)


def _provider(model, behavior):
    from types import SimpleNamespace

    prov = GeminiProvider(Settings(api_key="dummy", transcription_model=model))
    prov.client = _FakeClient(behavior)
    return prov, SimpleNamespace


def test_generate_falls_back_when_configured_model_retired():
    from types import SimpleNamespace

    def behavior(model):
        if model == "gemini-3-pro-preview":
            raise _Err(404, "models/gemini-3-pro-preview is no longer available")
        return SimpleNamespace(text="ok", parsed=None)

    prov, _ = _provider("gemini-3-pro-preview", behavior)
    assert prov.transcribe_text(b"img") == "ok"          # job still succeeds
    calls = prov.client.models.calls
    assert calls[0] == "gemini-3-pro-preview"            # tried the configured one
    assert calls[-1] in gemini_client.FALLBACK_MODELS    # then a stable fallback


def test_dead_model_is_remembered_and_skipped_next_call():
    from types import SimpleNamespace

    def behavior(model):
        if model == "gemini-3-pro-preview":
            raise _Err(404, "no longer available")
        return SimpleNamespace(text="ok", parsed=None)

    prov, _ = _provider("gemini-3-pro-preview", behavior)
    prov.transcribe_text(b"img")
    prov.client.models.calls.clear()
    prov.transcribe_text(b"img")  # second call (e.g. another page)
    # The retired model is not re-attempted; we go straight to the fallback.
    assert "gemini-3-pro-preview" not in prov.client.models.calls


def test_generate_raises_clear_error_when_all_models_unavailable():
    def behavior(model):
        raise _Err(404, "NOT_FOUND")

    prov, _ = _provider("gemini-x", behavior)
    with pytest.raises(RuntimeError) as ei:
        prov.transcribe_text(b"img")
    msg = str(ei.value).lower()
    assert "unavailable" in msg and "settings" in msg  # actionable, not a raw 404


def test_generate_does_not_mask_real_errors_with_fallback():
    def behavior(model):
        raise _Err(400, "INVALID_ARGUMENT: bad image")

    prov, _ = _provider("gemini-2.5-pro", behavior)
    with pytest.raises(Exception) as ei:
        prov.transcribe_text(b"img")
    assert "invalid_argument" in str(ei.value).lower()
    # Only the configured model is tried (full + minimal retry) -- never a fallback.
    assert set(prov.client.models.calls) == {"gemini-2.5-pro"}


# --- transient-failure retries (e.g. 503 deadline on big/dense images) ----- #

def test_is_transient_classification():
    assert gemini_client._is_transient(_Err(503, "UNAVAILABLE. Deadline expired before operation could complete."))
    assert gemini_client._is_transient(_Err(429, "RESOURCE_EXHAUSTED"))
    assert gemini_client._is_transient(_Err(500, "INTERNAL"))
    assert gemini_client._is_transient(ConnectionError("connection reset by peer"))
    # NOT transient: real client errors, auth, and model-gone are handled elsewhere.
    assert not gemini_client._is_transient(_Err(400, "INVALID_ARGUMENT: bad image"))
    assert not gemini_client._is_transient(_Err(403, "PERMISSION_DENIED"))
    assert not gemini_client._is_transient(_Err(404, "model is no longer available"))


def test_call_retries_transient_then_succeeds(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(gemini_client.time, "sleep", lambda *_: None)  # no real waiting
    calls = {"n": 0}

    def behavior(model):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Err(503, "UNAVAILABLE. Deadline expired before operation could complete.")
        return SimpleNamespace(text="ok", parsed=None)

    prov, _ = _provider("gemini-2.5-pro", behavior)
    assert prov.transcribe_text(b"img") == "ok"   # succeeds despite two 503s
    assert calls["n"] == 3                          # two retries, then success


def test_transient_exhausted_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(gemini_client.time, "sleep", lambda *_: None)

    def behavior(model):
        raise _Err(503, "UNAVAILABLE. Deadline expired before operation could complete.")

    prov, _ = _provider("gemini-2.5-pro", behavior)
    with pytest.raises(RuntimeError) as ei:
        prov.transcribe_text(b"img")
    msg = str(ei.value).lower()
    assert "timed out" in msg or "unavailable" in msg
    assert "max image dimension" in msg            # actionable, not a raw 503


# --- token usage accounting ---------------------------------------------- #

def test_call_accumulates_token_usage():
    from types import SimpleNamespace

    def behavior(model):
        return SimpleNamespace(
            text="ok",
            parsed=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=300,
                candidates_token_count=40,
                thoughts_token_count=10,
                total_token_count=350,
            ),
        )

    prov, _ = _provider("gemini-2.5-pro", behavior)
    assert prov.usage.calls == 0  # nothing billed before the first call
    prov.transcribe_text(b"img")
    assert prov.usage.input == 300
    assert prov.usage.output == 40
    assert prov.usage.thinking == 10
    assert prov.usage.calls == 1
    # A second page accumulates onto the running total.
    prov.transcribe_text(b"img")
    assert prov.usage.calls == 2
    assert prov.usage.input == 600


def test_failed_retried_attempts_are_not_billed(monkeypatch):
    # Only the successful response carries usage; the two 503s before it do not.
    from types import SimpleNamespace

    monkeypatch.setattr(gemini_client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def behavior(model):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Err(503, "UNAVAILABLE. Deadline expired.")
        return SimpleNamespace(
            text="ok",
            parsed=None,
            usage_metadata={"prompt_token_count": 100, "candidates_token_count": 5},
        )

    prov, _ = _provider("gemini-2.5-pro", behavior)
    prov.transcribe_text(b"img")
    assert prov.usage.calls == 1          # one billed call despite two retries
    assert prov.usage.input == 100


def test_count_input_tokens_uses_count_tokens_endpoint():
    from types import SimpleNamespace

    prov = GeminiProvider(Settings(api_key="dummy", transcription_model="gemini-2.5-pro"))
    seen = {}

    class _Models:
        def count_tokens(self, model, contents):
            seen["model"] = model
            seen["contents"] = contents
            return SimpleNamespace(total_tokens=1234)

    prov.client = SimpleNamespace(models=_Models())
    assert prov.count_input_tokens(b"imgbytes") == 1234
    assert seen["model"] == "gemini-2.5-pro"
    # The transcription prompt + the image are what get measured.
    assert any("paleographer" in str(c).lower() for c in seen["contents"])


def test_count_input_tokens_returns_zero_on_failure():
    from types import SimpleNamespace

    prov = GeminiProvider(Settings(api_key="dummy", transcription_model="gemini-2.5-pro"))

    class _Models:
        def count_tokens(self, model, contents):
            raise RuntimeError("network down")

    prov.client = SimpleNamespace(models=_Models())
    assert prov.count_input_tokens(b"x") == 0  # never blocks the estimate


def test_mock_provider_tracks_zero_usage():
    prov = gemini_client.MockProvider()
    prov.transcribe_text(b"img")
    # MockProvider makes no real call, so nothing is billed.
    assert prov.usage.total == 0
    assert prov.usage.calls == 0
    assert prov.count_input_tokens(b"img") == 0
