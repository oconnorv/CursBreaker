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
from .models import PageResult, TranscribedLine

ProgressCb = Callable[[int, int, str], None]


@dataclass
class FileResult:
    source_name: str
    n_pages: int = 0
    n_lines: int = 0
    txt_name: str | None = None
    hocr_name: str | None = None
    image_names: list[str] = field(default_factory=list)
    error: str | None = None


def process_page(loaded, provider: TranscriptionProvider, settings: Settings) -> PageResult:
    png = loaded.to_png_bytes()
    if settings.mode == "one_pass":
        items = provider.transcribe_with_boxes(png)
        plain_text = "\n".join(i.text for i in items)
        wire_lines = items
    else:
        text = provider.transcribe_text(png)
        detected = provider.detect_lines(png)
        wire_lines = align_lines(text.splitlines(), detected)
        plain_text = text

    w, h = loaded.sent_width, loaded.sent_height
    lines = [
        TranscribedLine(text=item.text, box=normalized_to_pixel(item.box_2d, w, h))
        for item in wire_lines
    ]
    return PageResult(
        image_name=f"{loaded.output_stem}.png",
        width=w,
        height=h,
        lines=lines,
        plain_text=plain_text,
    )


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

    txt = "\n\n".join(p.plain_text.strip() for p in page_results)
    (out_dir / txt_name).write_text(txt, "utf-8")

    hocr_bytes = build_hocr(
        page_results,
        ocr_system=f"CurseBreaker ({settings.mode})",
        confidence=settings.word_confidence,
    )
    (out_dir / hocr_name).write_bytes(hocr_bytes)

    return FileResult(
        source_name=path.name,
        n_pages=len(page_results),
        n_lines=sum(len(p.lines) for p in page_results),
        txt_name=txt_name,
        hocr_name=hocr_name,
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
