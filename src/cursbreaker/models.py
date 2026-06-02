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
