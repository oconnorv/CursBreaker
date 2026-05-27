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


class TranscribedLine(BaseModel):
    """One transcribed line placed on the page."""

    text: str
    box: PixelBox


class PageResult(BaseModel):
    """Everything needed to emit ``.txt`` and ``.hocr`` for one page image."""

    image_name: str
    width: int
    height: int
    lines: list[TranscribedLine]
    plain_text: str
