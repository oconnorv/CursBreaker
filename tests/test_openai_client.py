"""OpenAI provider: request shape, reply handling and the estimate caveat.

No network and no real SDK -- a fake client is swapped in after construction.
"""

import base64

import pytest

from cursbreaker.config import Settings
from cursbreaker.models import LineBox, LineBoxes
from cursbreaker.openai_client import OpenAIProvider


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
        self.output_text = text
        self.output_parsed = parsed
        self.usage = usage


class _FakeResponses:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


class _FakeClient:
    def __init__(self, resp, models=()):
        self.responses = _FakeResponses(resp)
        self.models = type(
            "M", (), {"list": lambda self_: [type("X", (), {"id": m})() for m in models]}
        )()

    def with_options(self, **_kwargs):
        # The real SDK returns a shallow copy carrying overridden request
        # options; for these tests the same object is indistinguishable.
        return self


def _provider(resp, models=(), **overrides):
    pytest.importorskip("openai")
    settings = Settings(
        provider="openai",
        openai_api_key="test-key",
        transcription_model=overrides.pop("model", "some-vision-model"),
        **overrides,
    )
    prov = OpenAIProvider(settings)
    prov.client = _FakeClient(resp, models)
    return prov


def test_missing_key_is_a_clear_error_not_a_crash():
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider(Settings(provider="openai"))


def test_transcribe_text_sends_a_data_url_image_and_returns_the_text():
    prov = _provider(_Resp(text="  transcribed  "))
    assert prov.transcribe_text(b"\x89PNG-bytes") == "transcribed"

    content = prov.client.responses.calls[-1]["input"][0]["content"]
    image = next(c for c in content if c["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")
    payload = image["image_url"].split(",", 1)[1]
    assert base64.standard_b64decode(payload) == b"\x89PNG-bytes"


def test_no_selected_model_asks_the_user_rather_than_guessing_an_id():
    # Settings normally supplies a default, but a blank must not become a
    # request with an empty model that 404s mid-batch.
    prov = _provider(_Resp(text="x"), model="")
    with pytest.raises(RuntimeError, match="Pick one in Settings"):
        prov.transcribe_text(b"d")


def test_a_rejected_model_id_names_the_alternatives_the_key_can_see():
    # Model names change. "model_not_found" alone leaves the user guessing,
    # so the error lists what their own key actually offers.
    prov = _provider(_Resp(text="x"), models=("gpt-6-astra", "gpt-5.6-luna"))

    class _Gone(Exception):
        code = 404
        message = "model_not_found"

    def boom(**kwargs):
        raise _Gone("404 model_not_found")

    prov.client.responses.create = boom
    with pytest.raises(RuntimeError) as err:
        prov.transcribe_text(b"d")
    assert "gpt-6-astra" in str(err.value)
    assert "gpt-5.6-luna" in str(err.value)


def test_a_rejected_model_still_errors_clearly_when_the_list_is_unavailable():
    prov = _provider(_Resp(text="x"))

    class _Gone(Exception):
        code = 404
        message = "model_not_found"

    def boom(**kwargs):
        raise _Gone("404 model_not_found")

    prov.client.responses.create = boom
    with pytest.raises(RuntimeError, match="Pick another in Settings"):
        prov.transcribe_text(b"d")


def test_boxes_come_from_the_structured_reply():
    parsed = LineBoxes(lines=[LineBox(text="hello", box_2d=[10, 20, 30, 40])])
    prov = _provider(_Resp(parsed=parsed))
    assert prov.transcribe_with_boxes(b"d") == [
        LineBox(text="hello", box_2d=[10, 20, 30, 40])
    ]
    assert prov.client.responses.calls[-1]["text_format"] is LineBoxes


def test_boxes_fall_back_to_parsing_raw_json():
    prov = _provider(_Resp(text='[{"text": "hi", "box_2d": [1,2,3,4]}]', parsed=None))
    assert prov.detect_lines(b"d") == [LineBox(text="hi", box_2d=[1, 2, 3, 4])]


def test_model_list_reads_the_users_own_key_and_drops_non_text_models():
    prov = _provider(
        _Resp(text="x"),
        models=("a-vision-model", "text-embedding-3", "whisper-1", "b-model"),
    )
    assert prov.list_models() == ["a-vision-model", "b-model"]


def test_model_list_is_empty_rather_than_fabricated_when_the_key_fails():
    prov = _provider(_Resp(text="x"))

    class _Boom:
        def list(self):
            raise RuntimeError("401 invalid api key")

    prov.client.models = _Boom()
    assert prov.list_models() == []


def test_input_tokens_are_reported_as_unmeasured_not_as_zero_cost():
    # OpenAI has no free token-counting endpoint. Returning 0 is honest only
    # because the estimate flags the input side as unmeasured; see
    # providers.PROVIDERS["openai"].counts_input_tokens.
    from cursbreaker.providers import provider_info

    prov = _provider(_Resp(text="x"))
    assert prov.count_input_tokens(b"d") == 0
    assert provider_info("openai").counts_input_tokens is False


def test_usage_records_what_the_call_billed():
    prov = _provider(_Resp(text="x", usage=_Usage(input_tokens=900, output_tokens=250)))
    prov.transcribe_text(b"d")
    assert (prov.usage.input, prov.usage.output, prov.usage.calls) == (900, 250, 1)


def test_reasoning_tokens_are_split_out_without_inflating_the_total():
    prov = _provider(
        _Resp(text="x", usage=_Usage(input_tokens=10, output_tokens=400, reasoning=150))
    )
    prov.transcribe_text(b"d")
    assert prov.usage.output == 250
    assert prov.usage.thinking == 150
    assert prov.usage.total == 410
