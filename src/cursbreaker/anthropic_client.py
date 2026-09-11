"""Anthropic Claude transcription provider.

Implements the same ``TranscriptionProvider`` contract as ``GeminiProvider``
(see ``gemini_client``), so the pipeline, hOCR/ALTO export and server can't
tell which service produced a page.

Three Claude specifics shape this file:

* **Structured output wants an object.** Claude's ``output_format`` takes a
  Pydantic model, not a bare list, so the box-producing calls ask for
  ``LineBoxes`` (``{"lines": [...]}``) and unwrap it.
* **Reasoning is controlled by effort, not a token budget.** ``budget_tokens``
  is rejected outright by current Claude models; ``output_config.effort`` is the
  equivalent lever. The app's "thinking level" maps onto it directly, and the
  default stays deliberately low -- extra reasoning was found to *hurt*
  handwriting accuracy, which is the same reason the Gemini path ships a tiny
  thinking budget.
* **Input tokens are countable for free.** ``messages.count_tokens`` prices the
  page image before a run, so the pre-flight estimate is as real as Gemini's.
"""

from __future__ import annotations

import base64

from .config import Settings
from .models import LineBox, LineBoxes, TokenUsage
from .prompts import (
    PROMPT_DETECT,
    PROMPT_LINES_ENVELOPE,
    PROMPT_ONE_PASS,
    PROMPT_TRANSCRIBE,
)
from .providers import (
    METADATA_TIMEOUT_S,
    REQUEST_TIMEOUT_S,
    call_with_retries,
    is_transient,
    lines_from_json_text,
    transient_message,
)

# Shown when the live model list can't be reached. Kept to models this app can
# actually use: current, vision-capable, and priced in ``pricing.CATALOG``.
SUGGESTED_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]

DEFAULT_MODEL = "claude-opus-5"

# Claude reads images as base64 blocks; these are the media types the app's
# rendering step can produce.
_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def probe_key(key: str) -> None:
    """One models-list request to verify a key. Metadata only -- it spends no
    generation tokens. Returns on success, raises on failure. Isolated so tests
    can stub it without the SDK or a network."""
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=0)
    for _ in client.models.list():
        return  # a single item proves the key authenticated
    return       # an empty list still means auth succeeded


class AnthropicProvider:
    def __init__(self, settings: Settings):
        import anthropic  # imported lazily so the app loads without the SDK

        api_key = settings.resolved_api_key("anthropic")
        if not api_key:
            raise RuntimeError(
                "No Anthropic API key set. Add one in Settings or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        self.settings = settings
        self._anthropic = anthropic
        # A generous timeout so one slow (large, dense) image isn't cut off, and
        # no SDK-level retries: ``call_with_retries`` owns retry policy so the
        # two layers can't multiply into a very long stall.
        self.client = anthropic.Anthropic(
            api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0
        )
        self.usage = TokenUsage()

    # -- public API ---------------------------------------------------------

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        resp = self._call(PROMPT_TRANSCRIBE, image_png, mime)
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "\n".join(parts).strip()

    def detect_lines(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._lines(PROMPT_DETECT, image_png, mime)

    def transcribe_with_boxes(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._lines(PROMPT_ONE_PASS, image_png, mime)

    def list_models(self) -> list[str]:
        try:
            client = self.client.with_options(timeout=METADATA_TIMEOUT_S)
            names = [
                m.id for m in client.models.list()
                if "claude" in (getattr(m, "id", "") or "")
            ]
        except Exception:
            return list(SUGGESTED_MODELS)
        return names or list(SUGGESTED_MODELS)

    def count_input_tokens(
        self, image_png: bytes, mime: str = "image/png"
    ) -> int:
        """Input tokens this page (image + transcription prompt) will cost, via
        the free ``count_tokens`` endpoint -- no generation quota is spent, so a
        pre-flight estimate costs nothing. Output tokens can't be known until
        the text exists, so only the (usually dominant) image side is measured.
        Returns 0 on any failure so an estimate never blocks the workflow."""
        try:
            resp = self.client.messages.count_tokens(
                model=self._model(),
                messages=[{
                    "role": "user",
                    "content": self._content(PROMPT_TRANSCRIBE, image_png, mime),
                }],
            )
        except Exception:
            return 0
        return int(getattr(resp, "input_tokens", 0) or 0)

    # -- internals ----------------------------------------------------------

    def _model(self) -> str:
        return self.settings.transcription_model or DEFAULT_MODEL

    def _content(self, prompt: str, image_png: bytes, mime: str) -> list[dict]:
        media_type = mime if mime in _MEDIA_TYPES else "image/png"
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(image_png).decode("ascii"),
                },
            },
            {"type": "text", "text": prompt},
        ]

    def _request_kwargs(self, prompt: str, image_png: bytes, mime: str) -> dict:
        kwargs: dict = {
            "model": self._model(),
            "max_tokens": self.settings.max_output_tokens,
            "messages": [{
                "role": "user",
                "content": self._content(prompt, image_png, mime),
            }],
        }
        # Effort is Claude's reasoning dial. Default low: more reasoning makes
        # handwriting transcription worse, not better (same finding that keeps
        # the Gemini thinking budget at 128).
        effort = (self.settings.thinking_level or "low").lower()
        if effort in ("low", "medium", "high", "xhigh", "max"):
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    def _call(self, prompt: str, image_png: bytes, mime: str):
        """One transcription request, retried on transient failures."""
        kwargs = self._request_kwargs(prompt, image_png, mime)
        try:
            resp = call_with_retries(
                lambda: self.client.messages.create(**kwargs), "Claude"
            )
        except Exception as exc:
            if is_transient(exc):
                raise RuntimeError(transient_message(exc, "Claude")) from exc
            raise
        self._tally(resp)
        return resp

    def _lines(self, prompt: str, image_png: bytes, mime: str) -> list[LineBox]:
        """A box-producing call. Asks for the ``LineBoxes`` schema so the reply
        is machine-checked, and still falls back to parsing raw JSON text -- a
        model that answers with a bare array or a fenced block shouldn't cost
        the user a page."""
        kwargs = self._request_kwargs(prompt + PROMPT_LINES_ENVELOPE, image_png, mime)
        try:
            resp = call_with_retries(
                lambda: self.client.messages.parse(
                    output_format=LineBoxes, **kwargs
                ),
                "Claude",
            )
        except Exception as exc:
            if is_transient(exc):
                raise RuntimeError(transient_message(exc, "Claude")) from exc
            raise
        self._tally(resp)

        parsed = getattr(resp, "parsed_output", None)
        if isinstance(parsed, LineBoxes):
            return list(parsed.lines)
        return lines_from_json_text(_response_text(resp))

    def _tally(self, resp) -> None:
        """Record what this (successful) call billed. Failed attempts produced
        no output and are not counted, so the total tracks real charges.

        Claude reports reasoning inside ``output_tokens``; the app shows what
        thinking cost as its own figure, so the reasoning share is split out
        here rather than added on top -- the two must still sum to what Claude
        billed."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            self.usage.add()
            return
        output = int(getattr(usage, "output_tokens", 0) or 0)
        reasoning = min(_thinking(getattr(usage, "output_tokens_details", None)), output)
        self.usage.add(
            input=getattr(usage, "input_tokens", 0) or 0,
            output=output - reasoning,
            thinking=reasoning,
        )


def _thinking(details) -> int:
    if details is None:
        return 0
    for name in ("reasoning_tokens", "thinking_tokens"):
        value = getattr(details, name, None)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _response_text(resp) -> str:
    parts = [
        b.text for b in getattr(resp, "content", []) or []
        if getattr(b, "type", "") == "text"
    ]
    return "\n".join(parts).strip()
