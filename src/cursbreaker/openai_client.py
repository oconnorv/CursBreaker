"""OpenAI transcription provider.

Implements the same ``TranscriptionProvider`` contract as ``GeminiProvider``
(see ``gemini_client``), so nothing downstream knows which service ran.

One OpenAI specific shapes this file, and it is visible to the user rather
than hidden: **input tokens can't be counted for free.** OpenAI exposes no
count-tokens endpoint, so ``count_input_tokens`` returns 0 and the pre-flight
estimate covers the output side only, presented as a minimum rather than an
expected total. The *actual* cost reported after a run is complete, because
the response's usage does carry real input tokens.

When a model id is rejected, the error names what the key can actually see --
a catalogued id can go stale, and "model not found" alone leaves a user
guessing.

Requests use the Responses API (``responses.parse`` / ``responses.create``),
which is where structured output and image input live in the current SDK.
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
    is_model_unavailable,
    is_transient,
    lines_from_json_text,
    short_error,
    transient_message,
)

# Model ids that exist on a key but can't transcribe a page image. Filtering by
# what a model *isn't* keeps new vision models appearing in the list the day
# they ship, instead of waiting for this file to learn their names.
_NON_TEXT_MARKERS = (
    "embedding", "whisper", "tts", "audio", "realtime", "dall-e", "image",
    "moderation", "transcribe", "search", "codex", "sora", "guard",
)


def probe_key(key: str) -> None:
    """One models-list request to verify a key. Metadata only -- it spends no
    generation tokens. Returns on success, raises on failure. Isolated so tests
    can stub it without the SDK or a network."""
    import openai

    client = openai.OpenAI(api_key=key, timeout=30.0, max_retries=0)
    for _ in client.models.list():
        return  # a single item proves the key authenticated
    return       # an empty list still means auth succeeded


class OpenAIProvider:
    def __init__(self, settings: Settings):
        import openai  # imported lazily so the app loads without the SDK

        api_key = settings.resolved_api_key("openai")
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key set. Add one in Settings or set the "
                "OPENAI_API_KEY environment variable."
            )
        self.settings = settings
        self._openai = openai
        # Generous timeout for large, dense images; retries are owned by
        # ``call_with_retries`` so the two layers can't compound.
        self.client = openai.OpenAI(
            api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0
        )
        self.usage = TokenUsage()

    # -- public API ---------------------------------------------------------

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        resp = self._run(
            lambda: self.client.responses.create(
                **self._request_kwargs(PROMPT_TRANSCRIBE, image_png, mime)
            )
        )
        return (getattr(resp, "output_text", "") or "").strip()

    def detect_lines(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._lines(PROMPT_DETECT, image_png, mime)

    def transcribe_with_boxes(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._lines(PROMPT_ONE_PASS, image_png, mime)

    def list_models(self) -> list[str]:
        """Text-capable models this key can reach. The dropdown is the curated
        priced catalog, so this is used for diagnostics: naming the real
        alternatives when a configured model id turns out to be wrong."""
        try:
            client = self.client.with_options(timeout=METADATA_TIMEOUT_S)
            names = [
                mid for m in client.models.list()
                if (mid := (getattr(m, "id", "") or ""))
                and not any(marker in mid for marker in _NON_TEXT_MARKERS)
            ]
        except Exception:
            return []
        return sorted(set(names))

    def count_input_tokens(
        self, image_png: bytes, mime: str = "image/png"
    ) -> int:
        """Always 0: OpenAI has no free token-counting endpoint, and guessing
        an image's token count from its dimensions would put an invented number
        underneath a dollar figure. The estimate reports this honestly -- as a
        minimum, not a total (see
        ``providers.PROVIDERS['openai'].counts_input_tokens``)."""
        return 0

    # -- internals ----------------------------------------------------------

    def _model(self) -> str:
        model = (self.settings.transcription_model or "").strip()
        if not model:
            raise RuntimeError(
                "No OpenAI model selected. Pick one in Settings -- the list is "
                "read from your own API key."
            )
        return model

    def _content(self, prompt: str, image_png: bytes, mime: str) -> list[dict]:
        data = base64.standard_b64encode(image_png).decode("ascii")
        return [
            {"type": "input_image", "image_url": f"data:{mime};base64,{data}"},
            {"type": "input_text", "text": prompt},
        ]

    def _request_kwargs(self, prompt: str, image_png: bytes, mime: str) -> dict:
        return {
            "model": self._model(),
            "max_output_tokens": self.settings.max_output_tokens,
            "input": [{
                "role": "user",
                "content": self._content(prompt, image_png, mime),
            }],
        }

    def _run(self, call):
        """Issue one request with retries, tally what it billed, and turn an
        exhausted transient failure into advice instead of a raw 503."""
        try:
            resp = call_with_retries(call, "OpenAI")
        except Exception as exc:
            if is_transient(exc):
                raise RuntimeError(transient_message(exc, "OpenAI")) from exc
            if is_model_unavailable(exc):
                raise RuntimeError(self._model_gone_message(exc)) from exc
            raise
        self._tally(resp)
        return resp

    def _model_gone_message(self, exc: Exception) -> str:
        """A rejected model id is recoverable, but only if the user can see
        what to switch to -- model names change, and a key may simply not be
        granted the one we default to."""
        msg = (
            f"OpenAI rejected the model '{self._model()}' "
            f"({short_error(exc)}). It may have been renamed or retired, or "
            "your key may not have access to it. Pick another in Settings."
        )
        available = self.list_models()
        if available:
            shown = ", ".join(available[:12])
            more = "…" if len(available) > 12 else ""
            msg += f" Models your key can see: {shown}{more}"
        return msg

    def _lines(self, prompt: str, image_png: bytes, mime: str) -> list[LineBox]:
        """A box-producing call. Uses the ``LineBoxes`` schema so the reply is
        machine-checked, and still falls back to parsing raw JSON text so a
        stray code fence or bare array doesn't cost the user a page."""
        kwargs = self._request_kwargs(prompt + PROMPT_LINES_ENVELOPE, image_png, mime)
        resp = self._run(
            lambda: self.client.responses.parse(text_format=LineBoxes, **kwargs)
        )
        parsed = getattr(resp, "output_parsed", None)
        if isinstance(parsed, LineBoxes):
            return list(parsed.lines)
        return lines_from_json_text(getattr(resp, "output_text", "") or "")

    def _tally(self, resp) -> None:
        """Record what this (successful) call billed. Reasoning tokens are
        billed inside ``output_tokens``, so the reasoning share is split out
        rather than added on top -- the parts still sum to what OpenAI charged."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            self.usage.add()
            return
        output = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        reasoning = 0
        if details is not None:
            try:
                reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
            except (TypeError, ValueError):
                reasoning = 0
        reasoning = min(reasoning, output)
        self.usage.add(
            input=getattr(usage, "input_tokens", 0) or 0,
            output=output - reasoning,
            thinking=reasoning,
        )
