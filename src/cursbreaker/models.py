"""Data models shared across the pipeline.

Two families of models live here:

* ``LineBox`` is the *wire* schema handed to Gemini as a ``response_schema``.
  It must contain no default values: the Gemini structured-output API rejects
  JSON schemas that carry ``default`` keywords.
* The remaining models describe results in real pixel space and are used
  internally by the pipeline, hOCR builder and web server.
"""

from __future__ import annotations

from pydantic import BaseModel


class LineBox(BaseModel):
    """A single text line as returned by Gemini.

    ``box_2d`` is ``[ymin, xmin, ymax, xmax]`` normalized to a 0-1000 grid with
    the origin at the top-left corner, which is the format Gemini emits for
    spatial/detection tasks. Do not add field defaults here (see module docs).
    """

    text: str
    box_2d: list[int]


class PlacedLine(BaseModel):
    """A line text placed onto a normalized box, with provenance.

    Used internally between alignment and the pipeline. ``is_interpolated`` is
    ``True`` for boxes the aligner had to estimate (no matching detected box),
    so the hOCR builder can mark them with a lower word confidence.
    """

    text: str
    box_2d: list[int]
    is_interpolated: bool = False


class PixelBox(BaseModel):
    """An axis-aligned box in real image pixels (origin top-left)."""

    x0: int
    y0: int
    x1: int
    y1: int

    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    def height(self) -> int:
        return max(0, self.y1 - self.y0)


class OcrWord(BaseModel):
    """A word with a real (engine-provided) bounding box and confidence."""

    text: str
    box: PixelBox
    confidence: int = 95


class TranscribedLine(BaseModel):
    """One transcribed line placed on the page."""

    text: str
    box: PixelBox
    # Word-level confidence written into hOCR (x_wconf). A lower value flags
    # lines whose bounding box was interpolated rather than detected.
    confidence: int = 95
    # When set, real engine-provided per-word data (text + boxes + per-word
    # confidence). When None, the hOCR builder falls back to proportional
    # word-box synthesis from ``box``. Tesseract populates this; Gemini does not.
    words: list[OcrWord] | None = None


class PageResult(BaseModel):
    """Everything needed to emit ``.txt`` and ``.hocr`` for one page image."""

    image_name: str
    width: int
    height: int
    lines: list[TranscribedLine]
    plain_text: str


class TokenUsage(BaseModel):
    """Gemini token counts accumulated over one or more API calls.

    Google bills two kinds of tokens, at different per-million rates:

    * **input** -- ``prompt_token_count``; dominated by the page *image*, whose
      token count grows with media resolution (the high/medium/low tiling).
    * **output** -- ``candidates_token_count`` (the transcription text) plus
      ``thoughts_token_count`` (reasoning), both billed at the output rate. We
      keep ``thinking`` as a *separate* field (it is part of the billed output)
      so the cost of reasoning is visible -- handwriting accuracy is best with a
      small thinking budget, so users tuning it can see what it costs.

    ``calls`` is the number of billed ``generate_content`` responses, so a
    two-pass page (transcribe + locate) shows 2. Adding two usages sums every
    field, which is how per-file totals roll up into a per-job total."""

    input: int = 0
    output: int = 0
    thinking: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        """Every billed token: input + visible output + thinking."""
        return self.input + self.output + self.thinking

    def add_response(self, usage_metadata) -> None:
        """Accumulate one response's ``usage_metadata`` (an SDK object or a
        dict). A successful call with no metadata still counts as one call."""
        self.calls += 1
        if usage_metadata is None:
            return

        def _get(name: str) -> int:
            if isinstance(usage_metadata, dict):
                val = usage_metadata.get(name)
            else:
                val = getattr(usage_metadata, name, 0)
            try:
                return int(val or 0)
            except (TypeError, ValueError):
                return 0

        prompt = _get("prompt_token_count")
        thoughts = _get("thoughts_token_count")
        candidates = _get("candidates_token_count")
        # Some SDK/model combinations omit candidates but report a total; derive
        # the visible output from it so the numbers still add up.
        if not candidates:
            total = _get("total_token_count")
            if total:
                candidates = max(0, total - prompt - thoughts)
        self.input += prompt
        self.thinking += thoughts
        self.output += candidates

    def cost(self, price_input_per_mtok: float, price_output_per_mtok: float) -> float:
        """Rough dollar cost from per-million-token prices. Thinking is billed at
        the output rate, so it joins ``output`` here."""
        return (
            self.input / 1_000_000 * price_input_per_mtok
            + (self.output + self.thinking) / 1_000_000 * price_output_per_mtok
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            thinking=self.thinking + other.thinking,
            calls=self.calls + other.calls,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        """Field-wise difference, clamped at zero -- used to snapshot the usage a
        single file added to a long-lived provider."""
        return TokenUsage(
            input=max(0, self.input - other.input),
            output=max(0, self.output - other.output),
            thinking=max(0, self.thinking - other.thinking),
            calls=max(0, self.calls - other.calls),
        )
