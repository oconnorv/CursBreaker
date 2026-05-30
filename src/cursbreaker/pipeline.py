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

from .align import align_lines
from .config import Settings
from .gemini_client import TranscriptionProvider
from .hocr import build_hocr, normalized_to_pixel
from .images import load_pages
from .models import LabeledLineBox, PageResult, PlacedLine, TranscribedLine
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
        return _process_page_mixed(loaded, provider, settings)
    return _process_page_handwriting(loaded, provider, settings)


def _process_page_handwriting(
    loaded, provider: TranscriptionProvider, settings: Settings
) -> PageResult:
    """Original Gemini-only flow: one-pass or two-pass with alignment."""
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
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _process_page_text_only(loaded, settings: Settings) -> PageResult:
    """Tesseract-only flow: no Gemini call, real per-word boxes throughout."""
    tesseract_client.require_available()
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


def _process_page_mixed(
    loaded, provider: TranscriptionProvider, settings: Settings
) -> PageResult:
    """Gemini labels each line printed/handwritten; printed lines go through
    Tesseract on the cropped region (real per-word boxes), handwritten lines
    use Gemini's own transcription from the labeled detection. Outputs are
    merged in reading order. Gemini never sees Tesseract's text."""
    tesseract_client.require_available()
    png = loaded.to_png_bytes()
    labeled = provider.detect_lines_labeled(png)

    w, h = loaded.sent_width, loaded.sent_height
    lines: list[TranscribedLine] = []
    for item in labeled:
        # Normalized -> pixel for the line bounds first; we may crop the page
        # image at those coordinates to feed Tesseract.
        line_box = normalized_to_pixel(item.box_2d, w, h)
        if item.kind == "printed":
            lines.extend(_tesseract_line_from_crop(loaded.image, line_box, settings))
        else:
            lines.append(
                TranscribedLine(
                    text=item.text,
                    box=line_box,
                    confidence=settings.word_confidence,
                )
            )

    plain_text = "\n".join(l.text for l in lines)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _tesseract_line_from_crop(
    page_image, line_box, settings: Settings
) -> list[TranscribedLine]:
    """Crop ``line_box`` from the page, OCR it with Tesseract, and return
    TranscribedLine objects whose coordinates are back in page space.

    Tesseract may detect multiple physical lines inside a single Gemini-labeled
    region (uncommon but possible); we return them all so nothing is lost.
    """
    # Pad a few pixels so we don't clip characters at the edges of Gemini's
    # detected box; Tesseract benefits from a small surrounding margin.
    pad = 4
    x0 = max(0, line_box.x0 - pad)
    y0 = max(0, line_box.y0 - pad)
    x1 = min(page_image.width, line_box.x1 + pad)
    y1 = min(page_image.height, line_box.y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    crop = page_image.crop((x0, y0, x1, y1))
    try:
        # PSM 7 = "single line"; appropriate when Gemini has already isolated
        # one line for us.
        return tesseract_client.transcribe_region(
            crop, lang=settings.tesseract_language, psm=7, offset=(x0, y0)
        )
    except Exception:
        # If Tesseract chokes on this region, keep going with no output for
        # this line rather than killing the whole page.
        return []


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
    mode_label = (
        content if content in ("text", "mixed") else f"{content}/{settings.mode}"
    )
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
