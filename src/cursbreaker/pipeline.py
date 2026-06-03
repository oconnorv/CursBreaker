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
from .images import count_content_pages, load_pages
from .models import OcrWord, PageResult, PixelBox, PlacedLine, TokenUsage, TranscribedLine
from .pricing import PRICES_AS_OF, cost_for, effective_rates, pricing_for
from .searchable_pdf import build_searchable_pdf
from . import tesseract_client

ProgressCb = Callable[[int, int, str], None]

# Pre-flight estimate only: a transparent, clearly-labelled assumption for how
# many tokens one Gemini call returns (transcription text plus a little
# thinking). Output length is genuinely unknowable until the page is read, so
# this is surfaced to the user rather than hidden -- the live counter shows the
# real number during the run.
_EST_OUTPUT_TOKENS_PER_CALL = 800


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
    # Tokens this file billed to Gemini (zero for Printed-only mode).
    token_usage: TokenUsage = field(default_factory=TokenUsage)


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

    # Snapshot the provider's running token total so we can report exactly what
    # *this* file added (the provider is shared across a batch).
    usage_before = _provider_usage(provider)

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
        token_usage=_provider_usage(provider) - usage_before,
    )


def _provider_usage(provider: TranscriptionProvider) -> TokenUsage:
    """A copy of a provider's running token total, or an empty one if the
    provider doesn't track usage (keeps callers defensive)."""
    usage = getattr(provider, "usage", None)
    if isinstance(usage, TokenUsage):
        return usage.model_copy()
    return TokenUsage()


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


def estimate_usage(
    paths: list[str | Path],
    provider: TranscriptionProvider,
    settings: Settings,
) -> dict:
    """Pre-flight token/cost estimate for a set of files -- before any
    transcription runs, so the user can decide whether to proceed.

    *Input* tokens (dominated by the page image) are measured with the
    provider's free ``count_tokens`` on the first page of each file and scaled by
    the page count -- fast, and exact enough that the user can sanity-check the
    bill. *Output* tokens can't be known until the text is generated, so they use
    a single, surfaced per-call assumption. Two-pass sends the image twice, so it
    counts two calls per page; Printed-only runs make no Gemini call and estimate
    to zero. The returned dict is clearly labelled as an estimate by the UI,
    which also shows the per-million prices used."""
    content = (settings.content_type or "handwriting").lower()
    if content == "text":
        calls_per_page = 0
    else:
        calls_per_page = 1 if settings.mode == "one_pass" else 2

    input_tokens = 0
    total_pages = 0
    for path in paths:
        path = Path(path)
        try:
            n_pages = count_content_pages(path)
        except Exception:
            n_pages = 1
        total_pages += n_pages
        if calls_per_page == 0:
            continue
        # Render only the first page (cheap even for a big PDF) and scale its
        # measured input tokens across the file's pages and per-page calls.
        try:
            first = load_pages(
                path,
                preprocess=settings.preprocess,
                max_dimension=settings.max_dimension,
                pdf_dpi=settings.pdf_dpi,
                max_pages=1,
            )[0]
            per_page_input = provider.count_input_tokens(first.to_png_bytes())
        except Exception:
            per_page_input = 0
        input_tokens += per_page_input * n_pages * calls_per_page

    calls = total_pages * calls_per_page
    output_tokens = calls * _EST_OUTPUT_TOKENS_PER_CALL
    usage = TokenUsage(input=input_tokens, output=output_tokens, calls=calls)

    # Price automatically from the selected model's published rate (no manual
    # entry). No catalog entry, or no calls -> tokens only, no dollar figure.
    pricing = pricing_for(settings.transcription_model)
    if pricing and calls:
        in_rate, out_rate = effective_rates(pricing, usage)
        cost = cost_for(pricing, usage)
    else:
        in_rate = out_rate = 0.0
        cost = None
    return {
        "files": len(paths),
        "pages": total_pages,
        "calls": calls,
        "calls_per_page": calls_per_page,
        "input": usage.input,
        "output": usage.output,  # assumed; see assumed_output_tokens_per_call
        "total": usage.total,
        "assumed_output_tokens_per_call": _EST_OUTPUT_TOKENS_PER_CALL,
        "cost": cost,
        "model": settings.transcription_model,
        "model_label": pricing.label if pricing else settings.transcription_model,
        "price_input_per_mtok": in_rate,
        "price_output_per_mtok": out_rate,
        "prices_as_of": PRICES_AS_OF,
    }
