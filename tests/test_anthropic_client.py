"""Claude provider: request shape, reply handling and token accounting.

No network and no real SDK -- a fake client is swapped in after construction,
so what's under test is the part CursBreaker owns: what it asks for, what it
does with the answer, and what it records as spent.
"""

import base64

import pytest

from cursbreaker.anthropic_client import AnthropicProvider
from cursbreaker.config import Settings
from cursbreaker.models import LineBox, LineBoxes


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0, reasoning=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_tokens_details = (
            type("D", (), {"reasoning_tokens": reasoning})() if reasoning is not None
            else None
        )


class _Resp:
    def __init__(self, text="", parsed=None, usage=None):
        self.content = [_Block(text)] if text else []
        self.parsed_output = parsed
        self.usage = usage


class _FakeMessages:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"input_tokens": 4242})()


class _FakeClient:
    def __init__(self, resp):
        self.messages = _FakeMessages(resp)
        self.models = type("M", (), {"list": lambda self_: []})()

    def with_options(self, **_kwargs):
        # The real SDK returns a shallow copy carrying overridden request
        # options; for these tests the same object is indistinguishable.
        return self


def _provider(resp, **overrides):
    settings = Settings(
        provider="anthropic",
        anthropic_api_key="test-key",
        transcription_model="claude-opus-5",
        **overrides,
    )
    pytest.importorskip("anthropic")
    prov = AnthropicProvider(settings)
    prov.client = _FakeClient(resp)
    return prov


def test_missing_key_is_a_clear_error_not_a_crash():
    pytest.importorskip("anthropic")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(Settings(provider="anthropic"))


def test_transcribe_text_sends_the_image_and_returns_the_text():
    prov = _provider(_Resp(text="  line one\nline two  "))
    assert prov.transcribe_text(b"\x89PNG-bytes") == "line one\nline two"

    sent = prov.client.messages.calls[-1]
    assert sent["model"] == "claude-opus-5"
    content = sent["messages"][0]["content"]
    image = next(c for c in content if c["type"] == "image")
    assert image["source"]["media_type"] == "image/png"
    # The image must arrive intact -- a mangled encoding would transcribe blank
    # pages at full price.
    assert base64.standard_b64decode(image["source"]["data"]) == b"\x89PNG-bytes"


def test_an_unsupported_mime_falls_back_to_png_rather_than_being_rejected():
    prov = _provider(_Resp(text="x"))
    prov.transcribe_text(b"data", mime="image/tiff")
    content = prov.client.messages.calls[-1]["messages"][0]["content"]
    image = next(c for c in content if c["type"] == "image")
    assert image["source"]["media_type"] == "image/png"


def test_reasoning_effort_defaults_low_for_handwriting():
    # More reasoning measurably hurts handwriting transcription, which is why
    # the Gemini path ships a tiny thinking budget; effort is Claude's dial.
    prov = _provider(_Resp(text="x"))
    prov.transcribe_text(b"d")
    assert prov.client.messages.calls[-1]["output_config"] == {"effort": "low"}


def test_thinking_level_setting_drives_effort():
    prov = _provider(_Resp(text="x"), thinking_level="high")
    prov.transcribe_text(b"d")
    assert prov.client.messages.calls[-1]["output_config"] == {"effort": "high"}


def test_boxes_come_from_the_structured_reply():
    parsed = LineBoxes(lines=[LineBox(text="hello", box_2d=[10, 20, 30, 40])])
    prov = _provider(_Resp(parsed=parsed))
    out = prov.transcribe_with_boxes(b"d")
    assert out == [LineBox(text="hello", box_2d=[10, 20, 30, 40])]
    # The schema is requested, not hoped for.
    assert prov.client.messages.calls[-1]["output_format"] is LineBoxes


def test_boxes_fall_back_to_parsing_raw_json_when_the_schema_is_skipped():
    raw = '```json\n{"lines": [{"text": "hi", "box_2d": [1, 2, 3, 4]}]}\n```'
    prov = _provider(_Resp(text=raw, parsed=None))
    assert prov.detect_lines(b"d") == [LineBox(text="hi", box_2d=[1, 2, 3, 4])]


def test_usage_records_what_the_call_billed():
    prov = _provider(_Resp(text="x", usage=_Usage(input_tokens=1200, output_tokens=300)))
    prov.transcribe_text(b"d")
    assert prov.usage.input == 1200
    assert prov.usage.output == 300
    assert prov.usage.calls == 1


def test_reasoning_tokens_are_split_out_without_inflating_the_total():
    # Claude bills reasoning inside output_tokens. The app shows thinking as its
    # own figure, so the split must preserve the billed total.
    prov = _provider(
        _Resp(text="x", usage=_Usage(input_tokens=100, output_tokens=500, reasoning=200))
    )
    prov.transcribe_text(b"d")
    assert prov.usage.output == 300
    assert prov.usage.thinking == 200
    assert prov.usage.output + prov.usage.thinking == 500
    assert prov.usage.total == 600


def test_a_call_with_no_usage_metadata_still_counts_as_a_call():
    prov = _provider(_Resp(text="x", usage=None))
    prov.transcribe_text(b"d")
    assert prov.usage.calls == 1
    assert prov.usage.total == 0


def test_input_tokens_are_counted_for_free_before_a_run():
    prov = _provider(_Resp(text="x"))
    assert prov.count_input_tokens(b"d") == 4242


def test_counting_failure_degrades_to_zero_instead_of_blocking_an_estimate():
    prov = _provider(_Resp(text="x"))

    def boom(**kwargs):
        raise RuntimeError("service down")

    prov.client.messages.count_tokens = boom
    assert prov.count_input_tokens(b"d") == 0
