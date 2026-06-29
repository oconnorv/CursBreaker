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

import io
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


def _overlay_text(pdf_page, page_result: PageResult) -> None:
    """Lay the invisible (``render_mode=3``) OCR text onto an open PDF page.

    Box coordinates are in the OCR image's pixel space (the upright render the
    model saw). We scale them to the page's displayed size, then map the point
    from displayed -> unrotated coordinates via the page's derotation matrix and
    pass the page rotation to ``insert_text`` -- so the text lands correctly even
    on a page with a ``/Rotate`` (verified for 0/90/180/270)."""
    rect = pdf_page.rect  # displayed size, with any rotation already applied
    if not page_result.width or not page_result.height:
        return
    sx = rect.width / page_result.width
    sy = rect.height / page_result.height
    rot = pdf_page.rotation
    derot = pdf_page.derotation_matrix
    font = _register_overlay_font(pdf_page)
    for line in page_result.lines:
        # Prefer real per-word data (e.g. Tesseract) when present, so cursor
        # selection and search land on the actual word boxes; otherwise split the
        # line box proportionally (the only option for Gemini-sourced lines).
        if line.words:
            word_items = [(w.text, w.box) for w in line.words]
        else:
            word_items = list(split_line_into_words(line.text, line.box))
        for word, wbox in word_items:
            if wbox.x1 <= wbox.x0 or wbox.y1 <= wbox.y0:
                continue
            fontsize = max(1.0, (wbox.y1 - wbox.y0) * sy * 0.7)
            # Baseline at the box's bottom-left (top-left origin), in displayed
            # space, then mapped into the page's unrotated coordinate system.
            point = fitz.Point(wbox.x0 * sx, wbox.y1 * sy) * derot
            # Trailing space helps the PDF text extractor see a word break
            # between tightly packed boxes (otherwise "the cat" -> "thecat").
            pdf_page.insert_text(
                point, word + " ",
                fontsize=fontsize, fontname=font, render_mode=3, rotate=rot,
            )


def write_searchable_pdf_over_source(
    src_pdf_path: str | Path, pages: list[PageResult], out_path: str | Path
) -> None:
    """Searchable PDF for a *PDF* input: overlay the invisible OCR text onto a
    copy of the original PDF, preserving its images/vectors, page sizes and
    orientation exactly -- no re-rasterization, so the user keeps their original
    image quality. ``pages[i]`` corresponds to source page ``i``."""
    doc = fitz.open(str(src_pdf_path))
    try:
        if len(pages) > doc.page_count:
            raise ValueError("more OCR pages than the source PDF has")
        for i, page_result in enumerate(pages):
            _overlay_text(doc[i], page_result)
        # garbage=3 drops unused objects; deflate compresses our added text
        # streams. Already-compressed page images (JPEG/Flate) are left untouched.
        doc.save(str(out_path), garbage=3, deflate=True)
    finally:
        doc.close()


def write_searchable_pdf_from_images(
    images, pages: list[PageResult], out_path: str | Path
) -> None:
    """Searchable PDF for *image* inputs: one page per full-resolution source
    image (the user's original pixels, oriented upright -- not the downscaled or
    enhanced OCR render), with the invisible OCR text overlaid. ``images`` are PIL
    images; saved losslessly so no quality is lost."""
    if len(images) != len(pages):
        raise ValueError("images and pages must have the same length")
    doc = fitz.open()
    try:
        for img, page_result in zip(images, pages):
            w, h = img.size
            page = doc.new_page(width=w, height=h)
            buf = io.BytesIO()
            img.save(buf, format="PNG")  # lossless: keeps the original pixels
            page.insert_image(fitz.Rect(0, 0, w, h), stream=buf.getvalue())
            _overlay_text(page, page_result)
        doc.save(str(out_path), deflate=True)
    finally:
        doc.close()
