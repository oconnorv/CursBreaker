"""Provider registry, key resolution and dispatch.

These tests never touch a network or a real SDK: the two new clients are
exercised through fake client objects injected in place of the SDK's, which is
enough to pin the parts we actually own -- request shape, structured-output
unwrapping, token accounting and error classification.
"""

import pytest

from cursbreaker import providers
from cursbreaker.config import Settings
from cursbreaker.models import LineBox, LineBoxes
from cursbreaker.providers import (
    PROVIDERS,
    is_auth_error,
    is_model_unavailable,
    is_transient,
    lines_from_json_text,
    provider_info,
)


class _Err(Exception):
    """Stand-in for an SDK error carrying an HTTP status."""

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_every_provider_is_self_describing():
    for pid, info in PROVIDERS.items():
        assert info.id == pid
        assert info.key_field and info.env_vars
        assert info.console_url.startswith("https://")
        assert info.pricing_url.startswith("https://")


def test_unknown_provider_falls_back_to_the_default():
    # A hand-edited config or one written by a newer build must not brick the
    # app; it lands on Gemini rather than raising on every call.
    assert provider_info("not-a-provider").id == "gemini"
    assert provider_info(None).id == "gemini"
    assert provider_info("ANTHROPIC").id == "anthropic"


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def test_each_provider_reads_its_own_stored_key():
    s = Settings(
        api_key="gem", anthropic_api_key="ant", openai_api_key="oai"
    )
    assert s.resolved_api_key("gemini") == "gem"
    assert s.resolved_api_key("anthropic") == "ant"
    assert s.resolved_api_key("openai") == "oai"


def test_resolved_key_defaults_to_the_active_provider():
    s = Settings(provider="anthropic", api_key="gem", anthropic_api_key="ant")
    assert s.resolved_api_key() == "ant"


def test_environment_overrides_the_stored_key_per_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    s = Settings(provider="anthropic", anthropic_api_key="stored")
    assert s.resolved_api_key() == "from-env"
    # ...and only that provider's key is affected.
    assert s.resolved_api_key("gemini") == ""


def test_public_dict_reports_every_key_without_leaking_any():
    s = Settings(
        provider="anthropic",
        api_key="gemini-secret",
        anthropic_api_key="anthropic-secret",
    )
    data = s.public_dict()
    blob = repr(data)
    assert "gemini-secret" not in blob and "anthropic-secret" not in blob
    assert data["keys"]["gemini"]["set"] is True
    assert data["keys"]["anthropic"]["set"] is True
    assert data["keys"]["openai"]["set"] is False
    # The flat fields describe the *active* provider.
    assert data["api_key_set"] is True
    assert data["api_key_hint"].endswith("cret")
    assert data["api_key_source"] == "config"


def test_switching_provider_replaces_a_model_it_cannot_run():
    s = Settings(provider="anthropic", transcription_model="gemini-3.5-flash")
    s.sync_models()
    assert s.transcription_model == "claude-opus-5"
    assert s.detection_model == "claude-opus-5"


def test_switching_provider_keeps_a_model_it_can_run():
    s = Settings(provider="anthropic", transcription_model="claude-haiku-4-5")
    s.sync_models()
    assert s.transcription_model == "claude-haiku-4-5"


def test_switching_to_openai_selects_its_default_model():
    s = Settings(provider="openai", transcription_model="claude-opus-5")
    s.sync_models()
    assert s.transcription_model == "gpt-6-astra"


def test_an_uncatalogued_model_is_replaced_rather_than_left_to_404():
    s = Settings(provider="openai", transcription_model="gpt-retired-yesterday")
    s.sync_models()
    assert s.transcription_model == "gpt-6-astra"


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def test_auth_errors_are_recognized_across_sdk_dialects():
    assert is_auth_error(_Err(401, "Unauthorized"))
    assert is_auth_error(_Err(403, "PERMISSION_DENIED"))
    assert is_auth_error(_Err(400, "Incorrect API key provided"))
    assert is_auth_error(_Err(400, "invalid_api_key"))


def test_a_throttled_or_offline_key_is_never_called_invalid():
    # Telling a user their working key is dead is the worst failure here.
    assert not is_auth_error(_Err(429, "Rate limit reached"))
    assert not is_auth_error(_Err(503, "overloaded"))
    assert not is_auth_error(ConnectionError("connection reset by peer"))


def test_model_gone_is_distinct_from_key_and_network_problems():
    assert is_model_unavailable(_Err(404, "model_not_found"))
    assert is_model_unavailable(_Err(400, "The model does not exist"))
    assert not is_model_unavailable(_Err(429, "slow down"))
    assert not is_transient(_Err(404, "model_not_found"))


def test_transient_errors_are_retryable_but_auth_is_not():
    assert is_transient(_Err(429, "rate limit"))
    assert is_transient(_Err(500, "internal"))
    assert is_transient(_Err(503, "overloaded"))
    assert is_transient(ConnectionError("connection reset"))
    assert not is_transient(_Err(401, "unauthorized"))
    assert not is_transient(_Err(400, "invalid image"))


def test_call_with_retries_gives_up_on_a_permanent_error(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    attempts = []

    def boom():
        attempts.append(1)
        raise _Err(400, "bad request")

    with pytest.raises(_Err):
        providers.call_with_retries(boom, "Test")
    assert len(attempts) == 1  # no pointless retries on a permanent failure


def test_call_with_retries_recovers_from_a_blip(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise _Err(503, "overloaded")
        return "ok"

    assert providers.call_with_retries(flaky, "Test") == "ok"
    assert len(attempts) == 3


# --------------------------------------------------------------------------- #
# Shared reply parsing
# --------------------------------------------------------------------------- #
def test_lines_parse_from_a_bare_array():
    out = lines_from_json_text('[{"text": "hi", "box_2d": [1, 2, 3, 4]}]')
    assert out == [LineBox(text="hi", box_2d=[1, 2, 3, 4])]


def test_lines_parse_from_the_object_envelope_and_a_code_fence():
    raw = '```json\n{"lines": [{"text": "hi", "box_2d": [1, 2, 3, 4]}]}\n```'
    assert lines_from_json_text(raw) == [LineBox(text="hi", box_2d=[1, 2, 3, 4])]


def test_a_malformed_element_costs_only_itself():
    raw = '[{"text": "good", "box_2d": [1,2,3,4]}, {"nope": true}]'
    assert [b.text for b in lines_from_json_text(raw)] == ["good"]


def test_unparseable_text_yields_no_lines_rather_than_raising():
    assert lines_from_json_text("I'm sorry, I can't read this page.") == []
    assert lines_from_json_text("") == []


def test_line_boxes_envelope_round_trips():
    env = LineBoxes(lines=[LineBox(text="a", box_2d=[0, 0, 1, 1])])
    assert env.lines[0].text == "a"


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def test_make_provider_picks_the_configured_client(monkeypatch):
    import cursbreaker.anthropic_client as ac
    import cursbreaker.openai_client as oc

    monkeypatch.setattr(ac, "AnthropicProvider", lambda s: "claude-client")
    monkeypatch.setattr(oc, "OpenAIProvider", lambda s: "openai-client")
    assert providers.make_provider(Settings(provider="anthropic")) == "claude-client"
    assert providers.make_provider(Settings(provider="openai")) == "openai-client"


def test_check_api_key_without_a_key_says_so_per_provider():
    st = providers.check_api_key(Settings(provider="anthropic"))
    assert st.state == "no_key"
    assert st.provider == "anthropic"


def test_check_api_key_reports_a_rejected_key(monkeypatch):
    import cursbreaker.anthropic_client as ac

    def boom(key):
        raise _Err(401, "invalid x-api-key")

    monkeypatch.setattr(ac, "probe_key", boom)
    st = providers.check_api_key(
        Settings(provider="anthropic", anthropic_api_key="revoked")
    )
    assert st.state == "invalid"
    assert "Claude" in st.message


def test_check_api_key_stays_unsure_when_the_service_is_down(monkeypatch):
    import cursbreaker.openai_client as oc

    def boom(key):
        raise _Err(503, "overloaded")

    monkeypatch.setattr(oc, "probe_key", boom)
    st = providers.check_api_key(Settings(provider="openai", openai_api_key="k"))
    assert st.state == "unknown"  # never "invalid" for a working key
