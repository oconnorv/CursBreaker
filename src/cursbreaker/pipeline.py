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
from .images import count_content_pages, iter_pages, load_pages
from .models import OcrWord, PageResult, PixelBox, PlacedLine, TokenUsage, TranscribedLine
from .pricing import PRICES_AS_OF, cost_for, effective_rates, pricing_for
from .searchable_pdf import build_searchable_pdf
from . import tesseract_client

@dataclass(frozen=True)
class ProgressEvent:
    """One step of work, reported live as a batch runs.

    ``units_done``/``units_total`` are *pages* across the whole batch (the
    progress bar is page-driven). ``stage`` is a machine tag
    ("load" | "page" | "page_done" | "write" | "file_done" | "error" |
    "batch_done") and ``message`` is the human line shown in the activity log."""

    message: str
    stage: str
    file_index: int
    file_total: int
    file_name: str
    units_done: int
    units_total: int


# Top-level reporter the server supplies; receives one ProgressEvent per step.
Reporter = Callable[["ProgressEvent"], None]
# The per-file line emitter threaded into process_file/process_page: a callable
# ``report(message, *, stage="page")`` built by process_batch (which injects the
# file context and the running page counter). Defaults to a no-op everywhere so
# direct callers (e.g. tests) need not pass one.
StepReporter = Callable[..., None]


def _noop(*_args, **_kwargs) -> None:
    pass


# Cooperative-cancellation predicate, checked at page/file boundaries (a
# synchronous Gemini call can't be interrupted mid-flight).
CancelCheck = Callable[[], bool]


class JobCancelled(Exception):
    """Raised inside ``process_file`` when ``should_cancel()`` turns true, so the
    partially-processed file is abandoned cleanly (not recorded as an error)."""

# Pre-flight output-token estimate, per page. Output length is genuinely
# unknowable until a page is read, so these are deliberately ballpark figures,
# surfaced to the user and clearly labelled an estimate (the live counter shows
# the real number). Calibrated to dense archival handwriting on Gemini 3.x:
#   * the transcribe call returns plain text;
#   * the structured call returns that text AGAIN plus per-line bounding-box
#     coordinates + JSON, so it's larger.
# One-pass makes one structured call; two-pass makes a text call + a structured
# call (it effectively generates the page's text twice), which is why two-pass
# output is much more than 2x a single flat per-call number -- the bug this
# replaces (a flat 800/call) understated it by ~2.5x.
_EST_TEXT_OUTPUT_PER_PAGE = 1600        # plain transcription (the transcribe call)
_EST_STRUCTURED_OUTPUT_PER_PAGE = 2400  # text + box-coordinate JSON


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


def process_page(
    loaded,
    provider: TranscriptionProvider,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
    should_cancel: CancelCheck | None = None,
) -> PageResult:
    report = report or _noop
    # A cancel that landed while this page was rendering is caught here, before
    # the expensive transcription call is made.
    if should_cancel and should_cancel():
        raise JobCancelled()
    content = (settings.content_type or "handwriting").lower()
    if content == "text":
        return _process_page_text_only(
            loaded, settings, report=report, page_no=page_no, page_total=page_total
        )
    if content == "mixed":
        # Retired alias: "Gemini text + Tesseract" is now handwriting with word-
        # box refinement. Defensive, in case an un-migrated value reaches here.
        settings = settings.model_copy(
            update={"content_type": "handwriting", "refine_word_boxes": True}
        )
    return _process_page_handwriting(
        loaded, provider, settings,
        report=report, page_no=page_no, page_total=page_total,
        should_cancel=should_cancel,
    )


def _process_page_handwriting(
    loaded,
    provider: TranscriptionProvider,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
    should_cancel: CancelCheck | None = None,
) -> PageResult:
    """Gemini transcription (one-pass, or two-pass with line alignment). The
    Gemini text is always authoritative; when ``refine_word_boxes`` is set and
    Tesseract is available, real per-word boxes are layered on afterwards."""
    report = report or _noop
    pg = f"Page {page_no}/{page_total}"
    png = loaded.to_png_bytes()
    if settings.mode == "one_pass":
        report(f"{pg} · transcribing + locating (Gemini)…", stage="page")
        items = provider.transcribe_with_boxes(png)
        plain_text = "\n".join(i.text for i in items)
        placed: list[PlacedLine] = [
            PlacedLine(text=i.text, box_2d=list(i.box_2d), is_interpolated=False)
            for i in items
        ]
    else:
        report(f"{pg} · transcribing (Gemini)…", stage="page")
        text = provider.transcribe_text(png)
        # Cancelled during the first pass? Stop before the second (billed) call
        # so the user isn't charged for a request they no longer want.
        if should_cancel and should_cancel():
            raise JobCancelled()
        report(f"{pg} · locating lines (Gemini)…", stage="page")
        detected = provider.detect_lines(png)
        placed = align_lines(text.splitlines(), detected)
        plain_text = text

    # Skip the (local but per-line) Tesseract refinement if cancelled by now.
    if should_cancel and should_cancel():
        raise JobCancelled()
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
        report(f"{pg} · refining word positions (Tesseract)…", stage="page")
        _refine_word_boxes(loaded.image, lines, settings)
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


def _process_page_text_only(
    loaded,
    settings: Settings,
    *,
    report: StepReporter | None = None,
    page_no: int = 0,
    page_total: int = 0,
) -> PageResult:
    """Tesseract-only flow: no Gemini call, real per-word boxes throughout."""
    report = report or _noop
    report(f"Page {page_no}/{page_total} · reading text (Tesseract)…", stage="page")
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
    *,
    report: StepReporter | None = None,
    should_cancel: CancelCheck | None = None,
) -> FileResult:
    report = report or _noop
    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the provider's running token total so we can report exactly what
    # *this* file added (the provider is shared across a batch).
    usage_before = _provider_usage(provider)

    report("Loading…", stage="load")
    # Page count from cheap metadata (no rasterization), so a 48-page PDF doesn't
    # block here for a minute before the first cancel check -- pages are rendered
    # lazily, one per loop, with a cancel check before each.
    total_pages = count_content_pages(path)
    report(f"{total_pages} page(s) to transcribe", stage="load")

    page_results: list[PageResult] = []
    image_names: list[str] = []
    pages = iter_pages(
        path,
        preprocess=settings.preprocess,
        max_dimension=settings.max_dimension,
        pdf_dpi=settings.pdf_dpi,
    )
    i = 0
    try:
        while True:
            # Cooperative cancellation, checked BEFORE rendering the next page
            # (the render happens inside next()). A second check inside
            # process_page catches a cancel that lands mid-render, before the
            # expensive transcription. Either abandons this file's partial work.
            if should_cancel and should_cancel():
                raise JobCancelled()
            try:
                loaded = next(pages)
            except StopIteration:
                break
            i += 1
            page_results.append(
                process_page(
                    loaded, provider, settings,
                    report=report, page_no=i, page_total=total_pages,
                    should_cancel=should_cancel,
                )
            )
            png_name = f"{loaded.output_stem}.png"
            loaded.image.save(out_dir / png_name)
            image_names.append(png_name)
            # A page counts as "done" (advancing the bar) once it's transcribed
            # and its PNG is saved; process_batch increments the global page
            # counter on this stage.
            report(f"Page {i}/{total_pages} done", stage="page_done")
    finally:
        # Close the lazy renderer (releases the open PDF) if we bailed early.
        pages.close()

    report("Writing outputs (text, hOCR, searchable PDF)…", stage="write")
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

    n_pages = len(page_results)
    n_lines = sum(len(p.lines) for p in page_results)
    report(f"Done — {n_pages} page(s), {n_lines} line(s)", stage="file_done")
    return FileResult(
        source_name=path.name,
        n_pages=n_pages,
        n_lines=n_lines,
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
    report: Reporter | None = None,
    units_total: int = 0,
    should_cancel: CancelCheck | None = None,
) -> list[FileResult]:
    """Transcribe each file, emitting a ``ProgressEvent`` per step to ``report``.

    ``units_total`` is the total page count across the batch (the bar is
    page-driven); the running "pages done" counter advances on each ``page_done``
    step. Messages are prefixed with the filename only when the batch has more
    than one file. ``should_cancel`` (checked between files and pages) stops the
    batch cooperatively; files already finished keep their outputs."""
    results: list[FileResult] = []
    total = len(paths)
    state = {"done": 0}  # pages completed across the whole batch
    cancelled = False

    for idx, path in enumerate(paths):
        name = Path(path).name
        prefix = f"{name} · " if total > 1 else ""

        # A per-file line emitter that injects file context + the running page
        # counter. ``page_done`` advances the global counter before the event is
        # built, so ``units_done`` reflects the just-finished page.
        def report_line(message, *, stage="page", _idx=idx, _name=name, _prefix=prefix):
            if stage == "page_done":
                state["done"] += 1
            if report:
                report(ProgressEvent(
                    message=_prefix + message, stage=stage,
                    file_index=_idx, file_total=total, file_name=_name,
                    units_done=state["done"], units_total=units_total,
                ))

        if should_cancel and should_cancel():  # stop before starting a new file
            cancelled = True
            break
        try:
            results.append(
                process_file(
                    path, provider, settings, out_dir,
                    report=report_line, should_cancel=should_cancel,
                )
            )
        except JobCancelled:  # cancelled mid-file: drop the partial file, stop
            cancelled = True
            break
        except Exception as exc:  # one bad file must not kill the batch
            results.append(FileResult(source_name=name, error=str(exc)))
            report_line(f"Failed: {exc}", stage="error")

    if report and cancelled:
        report(ProgressEvent(
            message="Cancelled.", stage="cancelled",
            file_index=0, file_total=total,
            file_name=Path(paths[0]).name if paths else "",
            units_done=state["done"], units_total=units_total,
        ))
    elif report and total > 1:
        report(ProgressEvent(
            message="All files complete.", stage="batch_done",
            file_index=max(0, total - 1), file_total=total,
            file_name=Path(paths[-1]).name if paths else "",
            units_done=state["done"], units_total=units_total,
        ))
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
    # Output scales per page, by mode: one-pass = one structured call; two-pass =
    # a plain-text call plus a structured (boxes) call.
    if calls_per_page == 0:
        output_per_page = 0
    elif settings.mode == "one_pass":
        output_per_page = _EST_STRUCTURED_OUTPUT_PER_PAGE
    else:
        output_per_page = _EST_TEXT_OUTPUT_PER_PAGE + _EST_STRUCTURED_OUTPUT_PER_PAGE
    output_tokens = total_pages * output_per_page
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
        "output": usage.output,  # assumed; see assumed_output_tokens_per_page
        "total": usage.total,
        "assumed_output_tokens_per_page": output_per_page,
        "cost": cost,
        "model": settings.transcription_model,
        "model_label": pricing.label if pricing else settings.transcription_model,
        "price_input_per_mtok": in_rate,
        "price_output_per_mtok": out_rate,
        "prices_as_of": PRICES_AS_OF,
    }
