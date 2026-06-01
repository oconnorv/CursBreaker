from cursbreaker import gemini_client
from cursbreaker.config import Settings
from cursbreaker.gemini_client import _is_auth_error, check_api_key


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


def test_check_api_key_mock_skips_network():
    assert check_api_key(Settings(use_mock=True)).state == "mock"


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
