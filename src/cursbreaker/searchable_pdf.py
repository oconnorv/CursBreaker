"""Build a searchable PDF: page image + invisible OCR text overlay.

Each PDF page shows the rendered page image and carries an *invisible*
text layer placed at the bounding-box positions. Users see the original
handwriting, and any PDF viewer can select/search the transcribed text
on top of it.

The text uses PyMuPDF's ``render_mode=3``, which writes the characters
into the PDF content stream without filling or stroking them: invisible
on screen, but harvested by every PDF text extractor and search index.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from .hocr import split_line_into_words
from .models import PageResult

_FONT_NAME = "cb-uni"
_FONT_PATH = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"


def _register_overlay_font(page) -> str:
    """Make the bundled Unicode font available on this page, falling back to
    PyMuPDF's built-in Helvetica if the file is somehow missing."""
    if _FONT_PATH.is_file():
        try:
            page.insert_font(fontname=_FONT_NAME, fontfile=str(_FONT_PATH))
            return _FONT_NAME
        except Exception:
            pass
    return "helv"


def build_searchable_pdf(
    pages: list[PageResult],
    image_paths: list[Path],
) -> bytes:
    if len(pages) != len(image_paths):
        raise ValueError("pages and image_paths must have the same length")

    doc = fitz.open()
    try:
        for page, image_path in zip(pages, image_paths):
            pdf_page = doc.new_page(width=page.width, height=page.height)
            pdf_page.insert_image(
                fitz.Rect(0, 0, page.width, page.height),
                filename=str(image_path),
            )
            font = _register_overlay_font(pdf_page)
            for line in page.lines:
                # Prefer real per-word data (e.g. Tesseract) when present, so
                # cursor selection and search lands on the actual word boxes;
                # otherwise fall back to splitting the line box proportionally
                # (the only option for Gemini-sourced lines).
                if line.words:
                    word_items = [(w.text, w.box) for w in line.words]
                else:
                    word_items = list(split_line_into_words(line.text, line.box))
                for word, wbox in word_items:
                    if wbox.x1 <= wbox.x0 or wbox.y1 <= wbox.y0:
                        continue
                    fontsize = max(1.0, (wbox.y1 - wbox.y0) * 0.7)
                    # PyMuPDF's high-level API uses a top-left origin, so the
                    # baseline sits at the box's bottom edge.
                    # Trailing space helps the PDF text extractor see a word
                    # break between tightly packed word boxes (otherwise "the
                    # cat" can extract as "thecat").
                    pdf_page.insert_text(
                        (wbox.x0, wbox.y1),
                        word + " ",
                        fontsize=fontsize,
                        fontname=font,
                        render_mode=3,
                    )
        return doc.tobytes()
    finally:
        doc.close()
