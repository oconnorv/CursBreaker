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
from typing import Protocol

from .config import Settings
from .models import LineBox

# Shown in the UI as hints when a live model list is unavailable. The UI prefers
# the live list from the user's key; model names change frequently.
SUGGESTED_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

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


def make_provider(settings: Settings) -> TranscriptionProvider:
    if settings.use_mock:
        return MockProvider()
    return GeminiProvider(settings)


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
        self.client = genai.Client(api_key=api_key)

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

    # -- internals ----------------------------------------------------------

    def _generate(self, model, prompt, image_png, mime, schema=None):
        from google.genai import types

        part = types.Part.from_bytes(data=image_png, mime_type=mime)
        contents = [prompt, part]
        try:
            cfg = self._config(types, schema=schema, minimal=False)
            return self.client.models.generate_content(
                model=model, contents=contents, config=cfg
            )
        except Exception:
            # Optional knobs (thinking/media resolution) aren't supported by
            # every model; retry once with a minimal config before giving up.
            cfg = self._config(types, schema=schema, minimal=True)
            return self.client.models.generate_content(
                model=model, contents=contents, config=cfg
            )

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
