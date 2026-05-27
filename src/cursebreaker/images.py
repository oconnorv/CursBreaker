"""Load and normalize input images.

Supports TIFF, JPEG, PNG and GIF (raster, including multi-frame) plus PDF
(rasterized per page with PyMuPDF). Every page is returned as an RGB PIL image
alongside the bytes we will actually send to Gemini, so bounding boxes can be
mapped back to the correct dimensions.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

RASTER_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = RASTER_EXT | PDF_EXT


@dataclass
class LoadedPage:
    """A single page ready for transcription.

    ``image`` is the (possibly preprocessed/resized) RGB image that the
    bounding boxes will refer to. ``orig_width``/``orig_height`` are the
    dimensions of the user's source page; if we resized before sending to the
    API, the pipeline scales boxes back to these so the hOCR pairs with the
    original file.
    """

    image: Image.Image
    orig_width: int
    orig_height: int
    source_path: Path
    page_index: int  # 0 for single images; page number within a PDF/multiframe
    output_stem: str  # base name for the .txt/.hocr/.png outputs

    @property
    def sent_width(self) -> int:
        return self.image.width

    @property
    def sent_height(self) -> int:
        return self.image.height

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


def _preprocess(img: Image.Image, *, enabled: bool, max_dimension: int) -> Image.Image:
    """Conservative preprocessing. Aggressive filters can hurt handwriting, so
    we only normalize orientation/color and apply gentle enhancement."""
    img = ImageOps.exif_transpose(img)  # honor camera/scanner rotation
    if img.mode != "RGB":
        img = img.convert("RGB")
    if enabled:
        img = ImageEnhance.Brightness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.05)
        img = img.filter(ImageFilter.MedianFilter(size=3))  # mild denoise
    if max_dimension and max(img.size) > max_dimension:
        scale = max_dimension / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _raster_frames(path: Path) -> list[Image.Image]:
    """Return every frame of a raster file (1 for normal images, N for
    multi-frame TIFF/animated GIF)."""
    frames: list[Image.Image] = []
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            frames.append(im.copy())
    return frames


def load_pages(
    path: str | Path,
    *,
    preprocess: bool = True,
    max_dimension: int = 0,
    pdf_dpi: int = 300,
) -> list[LoadedPage]:
    """Load one input file into a list of pages."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type: {ext}")

    pages: list[LoadedPage] = []
    if ext in PDF_EXT:
        raw_frames = _rasterize_pdf(path, dpi=pdf_dpi)
    else:
        raw_frames = _raster_frames(path)

    multi = len(raw_frames) > 1
    for i, frame in enumerate(raw_frames):
        orig_w, orig_h = frame.size
        processed = _preprocess(
            frame, enabled=preprocess, max_dimension=max_dimension
        )
        stem = f"{path.stem}_page_{i + 1:04d}" if multi else path.stem
        pages.append(
            LoadedPage(
                image=processed,
                orig_width=orig_w,
                orig_height=orig_h,
                source_path=path,
                page_index=i,
                output_stem=stem,
            )
        )
    return pages


def _rasterize_pdf(path: Path, *, dpi: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            frames.append(img)
    return frames
