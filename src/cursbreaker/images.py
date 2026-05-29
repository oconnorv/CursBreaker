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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

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


def _is_thumbnail_frame(im: Image.Image) -> bool:
    """Return True for a TIFF frame flagged as a reduced-resolution thumbnail
    or transparency mask via the ``NewSubfileType`` tag (254).

    Many scanners embed a thumbnail as a second internal frame, which Pillow's
    ``n_frames`` would otherwise count as a page.
    """
    try:
        nst = im.tag_v2.get(254, 0)
    except (AttributeError, KeyError, TypeError):
        return False
    # Bit 0 (= 1): reduced-resolution version of another image (thumbnail).
    # Bit 2 (= 4): transparency mask for another image in this file.
    return bool(nst & 0b101)


def count_content_pages(path: str | Path) -> int:
    """Return the number of *content* pages, skipping embedded thumbnails."""
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in PDF_EXT:
            with fitz.open(path) as doc:
                return doc.page_count
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            if n <= 1:
                return n
            keep = 0
            for i in range(n):
                im.seek(i)
                if not _is_thumbnail_frame(im):
                    keep += 1
            # If every frame is somehow flagged, fall back to the raw count
            # rather than reporting zero pages.
            return keep or n
    except Exception:
        return 1


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


def _raster_frames(path: Path, *, dpi: int) -> list[Image.Image]:
    """Return every content frame of a raster file (1 for normal images, N for
    multi-frame TIFF/animated GIF). Embedded thumbnails / reduced-resolution
    preview frames (TIFF ``NewSubfileType`` flag) are skipped.

    Tries Pillow first. Falls back to PyMuPDF (already a dependency) for
    files Pillow can't decode -- most commonly TIFFs whose compression
    isn't supported by Pillow's bundled libtiff (old fax/JBIG/JPEG-in-
    TIFF variants and similar).
    """
    try:
        frames: list[Image.Image] = []
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            for i in range(n):
                im.seek(i)
                if n > 1 and _is_thumbnail_frame(im):
                    continue
                frames.append(im.copy())
        if not frames:
            # Every frame was flagged; keep the first so we never return
            # nothing for an otherwise readable file.
            with Image.open(path) as im:
                im.seek(0)
                frames = [im.copy()]
        return frames
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return _fitz_frames(path, zoom=dpi / 72.0)


def _fitz_frames(path: Path, *, zoom: float) -> list[Image.Image]:
    frames: list[Image.Image] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            frames.append(
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            )
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
        raw_frames = _raster_frames(path, dpi=pdf_dpi)

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
    return _fitz_frames(path, zoom=dpi / 72.0)
