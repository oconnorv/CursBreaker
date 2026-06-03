"""Transcription providers.

``GeminiProvider`` wraps the official ``google-genai`` SDK. ``MockProvider``
returns deterministic sample output so the entire application (UI, pipeline,
hOCR export) can be exercised without a real API key.

The defaults follow the recipe from Mark Humphries' "Gemini 3 Solves
Handwriting Recognition": temperature 0, high media resolution, and a
deliberately *low* thinking budget (extra reasoning was found to hurt
handwriting accuracy).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .models import LineBox, TokenUsage

# Shown as hints only when the live model list is unavailable; the UI prefers
# the live list from the user's key. We keep these to currently-callable models:
# retired "preview" names (e.g. gemini-3-pro-preview) still show up in the API's
# ListModels output but 404 on use, so suggesting them would mislead.
SUGGESTED_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# Stable, broadly-available models to fall back to when the configured model has
# been retired (preview models get removed without notice). Ordered best-first;
# flash is included because it's reachable on more keys (incl. free tier).
FALLBACK_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]

_READING_ORDER_RULE = (
    "Use natural reading order: if the page has multiple columns, finish each "
    "column from top to bottom before moving on to the next column (left to "
    "right). For a single column, just go top to bottom."
)

PROMPT_TRANSCRIBE = (
    "You are an expert paleographer. Carefully transcribe the handwriting in "
    "this document image. Transcribe every line of the main text, preserving "
    "the original line breaks (one source line per output line). "
    + _READING_ORDER_RULE
    + " Expand nothing, correct nothing, and translate nothing - reproduce the "
    "text exactly as written. Respond with ONLY the transcription text: no "
    "commentary, labels, or code fences."
)

PROMPT_DETECT = (
    "Detect every line of handwritten or printed text in this document image. "
    "Return a JSON array where each element has two fields: 'text' (your "
    "accurate transcription of that single line, exactly as written) and "
    "'box_2d' (the line's bounding box as [ymin, xmin, ymax, xmax], integers "
    "normalized to 0-1000 with the origin at the top-left). "
    + _READING_ORDER_RULE
    + " One element per source line. Do not merge separate lines. Never return "
    "masks, explanations, or code fences."
)

PROMPT_ONE_PASS = (
    "Carefully transcribe the handwriting in this document image, line by line. "
    "Return a JSON array where each element has 'text' (the accurate "
    "transcription of one source line, reproduced exactly as written) and "
    "'box_2d' ([ymin, xmin, ymax, xmax] integers normalized to 0-1000, origin "
    "top-left). "
    + _READING_ORDER_RULE
    + " One element per source line. Never return masks, explanations, or code "
    "fences."
)

class TranscriptionProvider(Protocol):
    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str: ...

    def detect_lines(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]: ...

    def transcribe_with_boxes(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]: ...

    def list_models(self) -> list[str]: ...

    # Running total of tokens this provider has spent (input/output/thinking/
    # calls). The pipeline snapshots deltas per file; the server reads it live.
    usage: TokenUsage

    def count_input_tokens(
        self, image_png: bytes, mime: str = "image/png"
    ) -> int: ...


def make_provider(settings: Settings) -> TranscriptionProvider:
    if settings.use_mock:
        return MockProvider()
    return GeminiProvider(settings)


@dataclass
class KeyStatus:
    """Result of a cheap, generation-free check that an API key still works."""

    state: str          # valid | invalid | unknown | no_key | mock
    message: str = ""


# Substrings that mark a genuine authentication failure (bad/revoked/expired
# key) in a Gemini error, independent of SDK version.
_AUTH_MARKERS = (
    "API_KEY_INVALID", "API KEY NOT VALID", "API KEY EXPIRED",
    "PERMISSION_DENIED", "UNAUTHENTICATED", "UNAUTHORIZED",
    "INVALID AUTHENTICATION",
)


def _is_auth_error(exc: Exception) -> bool:
    """True only when an exception clearly means a bad/revoked key.

    Deliberately conservative: a transient network error, a 5xx, or a 429
    rate-limit must NOT be classified as 'invalid', or we would tell a user
    their good key is dead."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (401, 403):
        return True
    blob = f"{getattr(exc, 'message', '')} {exc}".upper()
    return any(m in blob for m in _AUTH_MARKERS)


# Substrings marking a model that exists in ListModels but can't be called
# (retired/renamed/not granted). Version-independent.
_MODEL_GONE_MARKERS = (
    "NOT_FOUND", "NO LONGER AVAILABLE", "IS NOT FOUND", "NOT SUPPORTED",
    "DOES NOT EXIST", "UNKNOWN MODEL", "NOT FOUND FOR API VERSION",
)


def _is_model_unavailable(exc: Exception) -> bool:
    """True when an error means the *model* is gone (vs. a key/network problem),
    so we can fall back to another model instead of failing the whole job."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 404:
        return True
    blob = f"{getattr(exc, 'message', '')} {exc}".upper()
    return any(m in blob for m in _MODEL_GONE_MARKERS)


# Transient failures worth retrying with backoff: the service was momentarily
# unavailable/overloaded, hit a deadline, or a network blip occurred -- common
# with very large/dense images. Never auth or model-gone (handled separately).
_TRANSIENT_MARKERS = (
    "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED", "INTERNAL", "OVERLOADED",
    "TIMEOUT", "TIMED OUT", "TEMPORARILY", "CONNECTION", "RESET BY PEER",
)
# A dense map scan can legitimately take minutes, and a one-off 503/deadline
# usually clears on a retry.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0          # seconds; doubles each attempt (2, 4, 8)
_REQUEST_TIMEOUT_MS = 300_000    # 5-minute client timeout for slow, large images


def _is_transient(exc: Exception) -> bool:
    """True for retryable service/timeout errors (503 UNAVAILABLE, deadline
    exceeded, 429, 5xx, network blips) -- but never for auth or model-gone."""
    if _is_auth_error(exc) or _is_model_unavailable(exc):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (408, 429, 500, 502, 503, 504):
        return True
    blob = f"{getattr(exc, 'message', '')} {type(exc).__name__} {exc}".upper()
    return any(m in blob for m in _TRANSIENT_MARKERS)


def _short_error(exc: Exception) -> str:
    """A compact one-line version of an SDK error (drops the JSON blob)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None) or ""
    msg = str(getattr(exc, "message", "") or exc).split("{", 1)[0].strip()
    return f"{code} {msg}".strip()[:140] or "error"


def _transient_message(exc: Exception) -> str:
    """Actionable guidance when retries are exhausted on a transient error."""
    return (
        f"Gemini was unavailable or timed out ({_short_error(exc)}). This often "
        "happens with very large or dense images -- a high-resolution map or "
        "scan is a common cause. Try again; if it keeps happening, lower 'Max "
        "image dimension' in Advanced (e.g. 3000-4000) and/or set 'Media "
        "resolution' to medium so each request is lighter. For a printed map you "
        "can also use 'Printed only' mode, which runs locally with no API call."
    )


def _probe_models(key: str) -> None:
    """One ListModels request to verify a key. Free -- it returns metadata only,
    spending no generation tokens or quota. Returns on success; raises on
    failure. Isolated so tests can stub it without the SDK or a network."""
    from google import genai

    client = genai.Client(api_key=key)
    for _ in client.models.list():
        return  # a single item proves the key authenticated
    return       # an empty list still means auth succeeded


def check_api_key(settings: Settings) -> KeyStatus:
    """Verify a stored key is still active *without* spending generation quota.

    Uses the free ListModels endpoint so a revoked/expired/mistyped key is
    caught here -- in Settings -- instead of mid-transcription. A genuine auth
    failure is reported as ``invalid``; anything ambiguous (offline, timeout,
    5xx, rate-limit) is ``unknown`` so a good key is never called dead."""
    if settings.use_mock:
        return KeyStatus("mock", "Demo mode is on -- no real API call is made.")
    key = settings.resolved_api_key()
    if not key:
        return KeyStatus("no_key", "No API key is stored.")
    try:
        _probe_models(key)
        return KeyStatus("valid", "Key verified -- it's active.")
    except Exception as exc:  # noqa: BLE001 -- classify, never propagate
        if _is_auth_error(exc):
            return KeyStatus(
                "invalid",
                "This key was rejected -- it may have been revoked, expired, or "
                "mistyped. Paste a current key.",
            )
        return KeyStatus(
            "unknown",
            "Couldn't verify the key right now (a network or service issue); "
            "it may still be fine.",
        )


class GeminiProvider:
    def __init__(self, settings: Settings):
        from google import genai  # imported lazily so --mock works offline

        api_key = settings.resolved_api_key()
        if not api_key:
            raise RuntimeError(
                "No Gemini API key set. Add one in Settings or set "
                "GEMINI_API_KEY, or enable mock mode."
            )
        self.settings = settings
        self._genai = genai
        # A generous client timeout so one slow (large/dense) image isn't cut off
        # early; tolerate SDK variations in the option's shape.
        try:
            from google.genai import types

            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
            )
        except Exception:
            self.client = genai.Client(api_key=api_key)
        # Models that returned "not found" this session; skip re-trying them and
        # go straight to a fallback so we don't repeat the failed call per page.
        self._dead_models: set[str] = set()
        # Running token total across every billed call (input/output/thinking/
        # calls). Read live by the server for the in-progress token counter.
        self.usage = TokenUsage()

    # -- public API ---------------------------------------------------------

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        resp = self._generate(
            self.settings.transcription_model,
            PROMPT_TRANSCRIBE,
            image_png,
            mime,
        )
        return (getattr(resp, "text", "") or "").strip()

    def detect_lines(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        resp = self._generate(
            self.settings.detection_model,
            PROMPT_DETECT,
            image_png,
            mime,
            schema=list[LineBox],
        )
        return _parse_lineboxes(resp)

    def transcribe_with_boxes(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        resp = self._generate(
            self.settings.transcription_model,
            PROMPT_ONE_PASS,
            image_png,
            mime,
            schema=list[LineBox],
        )
        return _parse_lineboxes(resp)

    def list_models(self) -> list[str]:
        names: list[str] = []
        try:
            for m in self.client.models.list():
                name = (getattr(m, "name", "") or "").replace("models/", "")
                if "gemini" in name:
                    names.append(name)
        except Exception:
            return SUGGESTED_MODELS
        return sorted(set(names)) or SUGGESTED_MODELS

    def count_input_tokens(
        self, image_png: bytes, mime: str = "image/png"
    ) -> int:
        """How many *input* tokens this page (image + transcription prompt) will
        cost, via the SDK's ``count_tokens``. This is metadata-only -- it spends
        no generation quota -- so it powers a free pre-flight estimate. Output
        tokens can't be known until the text is generated, so only the (usually
        dominant) image-input side is measured here. Returns 0 on any failure so
        an estimate never blocks the workflow."""
        from google.genai import types

        part = types.Part.from_bytes(data=image_png, mime_type=mime)
        model = self._model_candidates(self.settings.transcription_model)[0]
        try:
            resp = self.client.models.count_tokens(
                model=model, contents=[PROMPT_TRANSCRIBE, part]
            )
        except Exception:
            return 0
        return int(getattr(resp, "total_tokens", 0) or 0)

    # -- internals ----------------------------------------------------------

    def _generate(self, model, prompt, image_png, mime, schema=None):
        """Call ``model``; if it has been retired (404 / not-found), fall back to
        a stable model so a stale config can't kill the job. Any other failure
        (auth, bad request, network) is surfaced unchanged -- we only switch
        models when the model itself is the problem."""
        last_exc: Exception | None = None
        candidates = self._model_candidates(model)
        for m in candidates:
            try:
                return self._invoke(m, prompt, image_png, mime, schema)
            except Exception as exc:
                if _is_model_unavailable(exc):
                    self._dead_models.add(m)
                    last_exc = exc
                    if m != candidates[-1]:
                        print(
                            f"WARNING: Gemini model '{m}' is unavailable (it may "
                            "have been retired); falling back to another model.",
                            file=sys.stderr,
                        )
                    continue
                if _is_transient(exc):
                    # _invoke already retried with backoff; give actionable
                    # guidance rather than surfacing a raw 503/deadline.
                    raise RuntimeError(_transient_message(exc)) from exc
                raise  # auth / bad request / etc. -- surface unchanged
        raise RuntimeError(
            "The configured Gemini model is unavailable and no fallback worked "
            f"(tried: {', '.join(candidates)}). It may have been retired -- pick "
            f"a current model in Settings. Last error: {last_exc}"
        )

    def _model_candidates(self, model: str) -> list[str]:
        """Requested model first, then stable fallbacks, minus any already known
        unavailable this session."""
        ordered = [model] + [m for m in FALLBACK_MODELS if m != model]
        live = [m for m in ordered if m not in self._dead_models]
        return live or ordered

    def _invoke(self, model, prompt, image_png, mime, schema):
        """One model attempt, with the optional-knob minimal-config retry. A
        not-found error is re-raised at once so ``_generate`` can fall back
        without wasting the minimal retry on a model that's gone."""
        from google.genai import types

        part = types.Part.from_bytes(data=image_png, mime_type=mime)
        contents = [prompt, part]
        try:
            cfg = self._config(types, schema=schema, minimal=False)
            return self._call(model, contents, cfg)
        except Exception as exc:
            if _is_model_unavailable(exc) or _is_transient(exc):
                raise  # model fallback / already-retried transient: no minimal retry
            # Optional knobs (thinking/media resolution) aren't supported by
            # every model; retry once with a minimal config before giving up.
            cfg = self._config(types, schema=schema, minimal=True)
            return self._call(model, contents, cfg)

    def _call(self, model, contents, cfg):
        """``generate_content`` with exponential backoff on transient failures
        (503/UNAVAILABLE, deadline-exceeded, 429, 5xx, network blips). A retried
        success keeps a one-off timeout from killing the whole job."""
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=model, contents=contents, config=cfg
                )
                # Tally what this (successful) call actually billed. Failed
                # attempts before a retry are not counted -- they produced no
                # output -- so this reflects tokens we were charged for.
                self.usage.add_response(getattr(resp, "usage_metadata", None))
                return resp
            except Exception as exc:
                if attempt >= _MAX_RETRIES or not _is_transient(exc):
                    raise
                print(
                    f"WARNING: Gemini call failed ({_short_error(exc)}); retrying "
                    f"in {delay:.0f}s ({attempt + 1}/{_MAX_RETRIES})…",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay *= 2

    def _config(self, types, *, schema, minimal: bool):
        kwargs: dict = {
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        if not minimal:
            tc = self._thinking_config(types)
            if tc is not None:
                kwargs["thinking_config"] = tc
            mr = self._media_resolution(types)
            if mr is not None:
                kwargs["media_resolution"] = mr
        return types.GenerateContentConfig(**kwargs)

    def _thinking_config(self, types):
        # Prefer Gemini 3's thinking_level; fall back to a token budget.
        level = self.settings.thinking_level
        if level:
            try:
                return types.ThinkingConfig(thinking_level=level)
            except Exception:
                pass
        try:
            return types.ThinkingConfig(thinking_budget=self.settings.thinking_budget)
        except Exception:
            return None

    def _media_resolution(self, types):
        mapping = {
            "high": "MEDIA_RESOLUTION_HIGH",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "low": "MEDIA_RESOLUTION_LOW",
        }
        attr = mapping.get(self.settings.media_resolution)
        if not attr:
            return None
        return getattr(types.MediaResolution, attr, None)


def _parse_lineboxes(resp) -> list[LineBox]:
    parsed = getattr(resp, "parsed", None)
    if parsed:
        out: list[LineBox] = []
        for item in parsed:
            if isinstance(item, LineBox):
                out.append(item)
            elif isinstance(item, dict):
                out.append(LineBox(**item))
        if out:
            return out
    # Fallback: parse the raw text as JSON.
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        return []
    text = _strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [LineBox(**d) for d in data if isinstance(d, dict)]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


class MockProvider:
    """Deterministic sample output for keyless testing and demos."""

    _LINES = [
        "This is a CursBreaker mock transcription.",
        "It runs without a Gemini API key.",
        "Each detected line carries a bounding box.",
        "Add your own key in Settings for real results.",
    ]

    def __init__(self):
        # Demo mode makes no real API call, so nothing is ever billed; the
        # counter stays at zero, which is the honest figure to show.
        self.usage = TokenUsage()

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        return "\n".join(self._LINES)

    def detect_lines(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._boxes()

    def transcribe_with_boxes(
        self, image_png: bytes, mime: str = "image/png"
    ) -> list[LineBox]:
        return self._boxes()

    def list_models(self) -> list[str]:
        return ["mock-model", *SUGGESTED_MODELS]

    def count_input_tokens(
        self, image_png: bytes, mime: str = "image/png"
    ) -> int:
        return 0

    def _boxes(self) -> list[LineBox]:
        out: list[LineBox] = []
        top = 120
        step = 180
        height = 110
        for i, text in enumerate(self._LINES):
            ymin = top + i * step
            out.append(
                LineBox(text=text, box_2d=[ymin, 80, ymin + height, 920])
            )
        return out
