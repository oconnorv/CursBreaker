"""Local Tesseract OCR for printed text.

Tesseract is a C++ system binary (apt/brew/Windows installer); ``pytesseract``
is just a thin wrapper. So both the binary AND the Python package can be absent
in a CursBreaker install — when they are, the rest of the app still works in
handwriting-only mode and any path that needs Tesseract raises a clear error.

Why integrate Tesseract at all when Gemini already transcribes handwriting?
For *printed* text it is:

* Local and free (no API cost, no network).
* Near-state-of-the-art accuracy on clean printed scans.
* Returns **real per-word boxes and per-word confidences**, which is strictly
  better than the proportional word-box synthesis we do for Gemini lines.

This module exposes a small, side-effect-free surface used by the pipeline:

* :func:`is_available` — quick environmental check for the UI status badge.
* :func:`available_languages` — what language packs the local install supports.
* :func:`transcribe_region` — OCR a (possibly cropped) PIL image and return
  ``TranscribedLine`` objects whose ``words`` field carries real per-word data,
  with coordinates offset back to page space when caller passes ``offset``.
* :func:`transcribe_page` — convenience wrapper for the whole-page case.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from .models import OcrWord, PixelBox, TranscribedLine


# pytesseract level codes (see the docs). We only care about word rows.
_WORD_LEVEL = 5


def is_available() -> bool:
    """Return True iff ``pytesseract`` is importable AND the binary responds."""
    try:
        import pytesseract  # noqa: F401  (probing the import)

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def available_languages() -> list[str]:
    """Return the language codes the local install supports (e.g. ``["eng"]``)."""
    try:
        import pytesseract

        langs = pytesseract.get_languages(config="")
    except Exception:
        return []
    return sorted(set(langs))


def transcribe_page(
    image: Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
) -> list[TranscribedLine]:
    """OCR the whole image; coordinates are already in page space."""
    return transcribe_region(image, lang=lang, psm=psm, offset=(0, 0))


def transcribe_region(
    image: Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
    offset: tuple[int, int] = (0, 0),
) -> list[TranscribedLine]:
    """OCR ``image`` (a possibly-cropped region) and return per-line results.

    ``offset`` is added to every returned coordinate so the caller can pass a
    crop of the page and still receive boxes in the original page's coordinate
    space — that is what the mixed-content pipeline does for printed lines.
    """
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    return _data_to_lines(data, offset)


def _data_to_lines(
    data: dict, offset: tuple[int, int]
) -> list[TranscribedLine]:
    """Group word-level rows by (block, par, line) and build TranscribedLine."""
    ox, oy = offset
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    n = len(data["text"])
    for i in range(n):
        if data["level"][i] != _WORD_LEVEL:
            continue
        text = data["text"][i].strip()
        # pytesseract returns confidence as a string in some versions and -1
        # for non-text rows; coerce and drop blanks.
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            conf = -1
        if not text or conf < 0:
            continue
        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": conf,
                "left": int(data["left"][i]) + ox,
                "top": int(data["top"][i]) + oy,
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )

    lines: list[TranscribedLine] = []
    # Preserve detection order: sort by (block, par, line) so reading order
    # follows Tesseract's own page-layout analysis.
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        # Within a line, sort words left-to-right so the line text reads right.
        rows.sort(key=lambda r: r["left"])
        words = [
            OcrWord(
                text=r["text"],
                box=PixelBox(
                    x0=r["left"],
                    y0=r["top"],
                    x1=r["left"] + r["width"],
                    y1=r["top"] + r["height"],
                ),
                confidence=r["conf"],
            )
            for r in rows
        ]
        if not words:
            continue
        x0 = min(w.box.x0 for w in words)
        y0 = min(w.box.y0 for w in words)
        x1 = max(w.box.x1 for w in words)
        y1 = max(w.box.y1 for w in words)
        line_text = " ".join(w.text for w in words)
        avg_conf = round(sum(w.confidence for w in words) / len(words))
        lines.append(
            TranscribedLine(
                text=line_text,
                box=PixelBox(x0=x0, y0=y0, x1=x1, y1=y1),
                confidence=avg_conf,
                words=words,
            )
        )
    return lines


def require_available() -> None:
    """Raise a clear, actionable error if Tesseract is not usable here."""
    if is_available():
        return
    raise RuntimeError(
        "Tesseract is not installed or not on PATH. Install the Tesseract OCR "
        "engine (Linux: 'sudo apt install tesseract-ocr'; macOS: "
        "'brew install tesseract'; Windows: the UB-Mannheim installer) and "
        "ensure the 'tesseract' command works, then restart CursBreaker."
    )
