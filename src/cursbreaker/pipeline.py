"""Batch orchestration: input files -> .txt + .hocr (+ page PNGs).

Outputs are written to a dedicated directory (the web server uses a per-job temp
dir) so the user's source files are never touched or overwritten. Each page is
re-rendered to PNG at the dimensions the bounding boxes refer to, so the
``.hocr`` and its image always pair correctly and travel together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .align import align_lines, align_words
from .config import Settings
from .gemini_client import TranscriptionProvider
from .hocr import build_hocr, normalized_to_pixel
from .images import load_pages
from .models import OcrWord, PageResult, PixelBox, PlacedLine, TranscribedLine
from .searchable_pdf import build_searchable_pdf
from . import tesseract_client

ProgressCb = Callable[[int, int, str], None]


@dataclass
class FileResult:
    source_name: str
    n_pages: int = 0
    n_lines: int = 0
    txt_name: str | None = None
    hocr_name: str | None = None
    pdf_name: str | None = None
    image_names: list[str] = field(default_factory=list)
    error: str | None = None


def process_page(loaded, provider: TranscriptionProvider, settings: Settings) -> PageResult:
    content = (settings.content_type or "handwriting").lower()
    if content == "text":
        return _process_page_text_only(loaded, settings)
    if content == "mixed":
        # Retired alias: "Gemini text + Tesseract" is now handwriting with word-
        # box refinement. Defensive, in case an un-migrated value reaches here.
        settings = settings.model_copy(
            update={"content_type": "handwriting", "refine_word_boxes": True}
        )
    return _process_page_handwriting(loaded, provider, settings)


def _process_page_handwriting(
    loaded, provider: TranscriptionProvider, settings: Settings
) -> PageResult:
    """Gemini transcription (one-pass, or two-pass with line alignment). The
    Gemini text is always authoritative; when ``refine_word_boxes`` is set and
    Tesseract is available, real per-word boxes are layered on afterwards."""
    png = loaded.to_png_bytes()
    if settings.mode == "one_pass":
        items = provider.transcribe_with_boxes(png)
        plain_text = "\n".join(i.text for i in items)
        placed: list[PlacedLine] = [
            PlacedLine(text=i.text, box_2d=list(i.box_2d), is_interpolated=False)
            for i in items
        ]
    else:
        text = provider.transcribe_text(png)
        detected = provider.detect_lines(png)
        placed = align_lines(text.splitlines(), detected)
        plain_text = text

    w, h = loaded.sent_width, loaded.sent_height
    lines: list[TranscribedLine] = []
    for p in placed:
        conf = (
            settings.interpolated_confidence if p.is_interpolated
            else settings.word_confidence
        )
        lines.append(
            TranscribedLine(
                text=p.text,
                box=normalized_to_pixel(p.box_2d, w, h),
                confidence=conf,
            )
        )
    # Optional, purely additive: refine word *positions* with Tesseract. The
    # Gemini transcription above is already final; this only attaches real
    # per-word boxes where Tesseract agrees, and can never change the text.
    if settings.refine_word_boxes and tesseract_client.is_available(settings):
        _refine_word_boxes(loaded.image, lines, settings)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _process_page_text_only(loaded, settings: Settings) -> PageResult:
    """Tesseract-only flow: no Gemini call, real per-word boxes throughout."""
    tesseract_client.require_available(settings)
    w, h = loaded.sent_width, loaded.sent_height
    lines = tesseract_client.transcribe_page(
        loaded.image, lang=settings.tesseract_language
    )
    plain_text = "\n".join(l.text for l in lines)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _refine_word_boxes(
    page_image, lines: list[TranscribedLine], settings: Settings
) -> None:
    """In place: attach real Tesseract word boxes to each line where the engine's
    text agrees with Gemini's. Line text is never touched; any per-line failure
    is swallowed so that line simply keeps its proportional word boxes."""
    for line in lines:
        try:
            ocr_words = _tesseract_words_for_line(page_image, line.box, settings)
        except Exception:
            # Tesseract choking on one region must not lose that line's text.
            continue
        if not ocr_words:
            continue  # leave words=None -> hOCR splits the line box proportionally
        line.words = align_words(
            line.text,
            ocr_words,
            line.box,
            matched_conf=settings.word_confidence,
            fallback_conf=settings.interpolated_confidence,
        )


def _tesseract_words_for_line(
    page_image, line_box: PixelBox, settings: Settings
) -> list[OcrWord]:
    """OCR a small crop around ``line_box`` and return Tesseract's per-word boxes
    in page coordinates (PSM 7, since Gemini already isolated one line)."""
    # Pad a few pixels so we don't clip characters at the edges of Gemini's box.
    pad = 4
    x0 = max(0, line_box.x0 - pad)
    y0 = max(0, line_box.y0 - pad)
    x1 = min(page_image.width, line_box.x1 + pad)
    y1 = min(page_image.height, line_box.y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    crop = page_image.crop((x0, y0, x1, y1))
    words: list[OcrWord] = []
    for tl in tesseract_client.transcribe_region(
        crop, lang=settings.tesseract_language, psm=7, offset=(x0, y0)
    ):
        words.extend(tl.words or [])
    return words


def process_file(
    path: str | Path,
    provider: TranscriptionProvider,
    settings: Settings,
    out_dir: Path,
) -> FileResult:
    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_pages = load_pages(
        path,
        preprocess=settings.preprocess,
        max_dimension=settings.max_dimension,
        pdf_dpi=settings.pdf_dpi,
    )

    page_results: list[PageResult] = []
    image_names: list[str] = []
    for loaded in loaded_pages:
        page_results.append(process_page(loaded, provider, settings))
        png_name = f"{loaded.output_stem}.png"
        loaded.image.save(out_dir / png_name)
        image_names.append(png_name)

    stem = path.stem
    txt_name = f"{stem}.txt"
    hocr_name = f"{stem}.hocr"
    pdf_name = f"{stem}.pdf"

    txt = "\n\n".join(p.plain_text.strip() for p in page_results)
    (out_dir / txt_name).write_text(txt, "utf-8")

    content = settings.content_type or "handwriting"
    if content == "text":
        mode_label = "text"
    else:
        mode_label = f"{content}/{settings.mode}"
        if settings.refine_word_boxes:
            mode_label += "+wordboxes"
    hocr_bytes = build_hocr(
        page_results,
        ocr_system=f"CursBreaker ({mode_label})",
        language=settings.language,
    )
    (out_dir / hocr_name).write_bytes(hocr_bytes)

    pdf_bytes = build_searchable_pdf(
        page_results, [out_dir / n for n in image_names]
    )
    (out_dir / pdf_name).write_bytes(pdf_bytes)

    return FileResult(
        source_name=path.name,
        n_pages=len(page_results),
        n_lines=sum(len(p.lines) for p in page_results),
        txt_name=txt_name,
        hocr_name=hocr_name,
        pdf_name=pdf_name,
        image_names=image_names,
    )


def process_batch(
    paths: list[str | Path],
    provider: TranscriptionProvider,
    settings: Settings,
    out_dir: Path,
    progress_cb: ProgressCb | None = None,
) -> list[FileResult]:
    results: list[FileResult] = []
    total = len(paths)
    for idx, path in enumerate(paths):
        name = Path(path).name
        if progress_cb:
            progress_cb(idx, total, name)
        try:
            results.append(process_file(path, provider, settings, out_dir))
        except Exception as exc:  # one bad file must not kill the batch
            results.append(FileResult(source_name=name, error=str(exc)))
        if progress_cb:
            progress_cb(idx + 1, total, name)
    return results
