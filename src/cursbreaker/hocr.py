"""Build valid hOCR 1.2 from transcribed lines.

hOCR is an HTML microformat: an ``ocr_page`` contains lines (``ocr_line``) which
contain words (``ocrx_word``). Each element carries a ``title`` attribute whose
``bbox x0 y0 x1 y1`` gives its pixel rectangle (origin top-left). Pairing this
file with the page image makes the text searchable and findable on the page.

Gemini reliably returns *line* boxes; per-word boxes across a dense page are
not reliable. We therefore synthesize per-word boxes by splitting each line box
horizontally in proportion to word length, which keeps words individually
searchable without depending on unreliable per-word detection.
"""

from __future__ import annotations

from lxml import etree

from .models import PageResult, PixelBox, TranscribedLine

XHTML_NS = "http://www.w3.org/1999/xhtml"
XML_NS = "http://www.w3.org/XML/1998/namespace"
DOCTYPE = (
    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
    '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
)
CAPABILITIES = "ocr_page ocr_carea ocr_par ocr_line ocrx_word"


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalized_to_pixel(box_2d: list[int], width: int, height: int) -> PixelBox:
    """Convert Gemini's ``[ymin, xmin, ymax, xmax]`` (0-1000) to pixels."""
    ymin, xmin, ymax, xmax = box_2d[0], box_2d[1], box_2d[2], box_2d[3]
    x0 = round(xmin / 1000.0 * width)
    x1 = round(xmax / 1000.0 * width)
    y0 = round(ymin / 1000.0 * height)
    y1 = round(ymax / 1000.0 * height)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return PixelBox(
        x0=_clamp(x0, 0, width),
        y0=_clamp(y0, 0, height),
        x1=_clamp(x1, 0, width),
        y1=_clamp(y1, 0, height),
    )


def split_line_into_words(text: str, box: PixelBox) -> list[tuple[str, PixelBox]]:
    """Split a line box into per-word boxes proportional to word length."""
    words = text.split()
    if not words:
        return []
    joined = " ".join(words)
    total = len(joined)
    span = box.x1 - box.x0
    out: list[tuple[str, PixelBox]] = []
    cursor = 0
    for word in words:
        start, end = cursor, cursor + len(word)
        if total > 0:
            wx0 = box.x0 + round(start / total * span)
            wx1 = box.x0 + round(end / total * span)
        else:
            wx0, wx1 = box.x0, box.x1
        if wx1 <= wx0:
            wx1 = wx0 + 1
        out.append((word, PixelBox(x0=wx0, y0=box.y0, x1=wx1, y1=box.y1)))
        cursor = end + 1  # skip the separating space
    boxes = sanitize_word_boxes([b for _, b in out], box)
    return [(word, b) for (word, _), b in zip(out, boxes)]


def sanitize_word_boxes(
    boxes: list[PixelBox], line_box: PixelBox, *, min_width: int = 2
) -> list[PixelBox]:
    """Clean per-word boxes in reading order: clip any overlap between adjacent
    boxes at their shared midpoint, then grow degenerate (sub-``min_width``)
    boxes into the surrounding gap. Length- and order-preserving. Synthesized
    boxes carry inter-word gaps to grow into, so a word whose proportional slice
    rounded to nothing becomes a real box instead of a 1px sliver."""
    if not boxes:
        return list(boxes)
    xs = [[b.x0, b.y0, b.x1, b.y1] for b in boxes]
    lo = min([line_box.x0, *(b[0] for b in xs)])   # widen to any legit overhang
    hi = max([line_box.x1, *(b[2] for b in xs)])
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if b[0] < a[2]:                            # clip the overlap at the midpoint
            mid = (b[0] + a[2]) // 2
            a[2] = b[0] = mid
    for i, box in enumerate(xs):
        if box[2] - box[0] >= min_width:
            continue
        right = xs[i + 1][0] if i + 1 < len(xs) else hi
        left = xs[i - 1][2] if i - 1 >= 0 else lo
        box[2] = min(max(box[2], box[0] + min_width), right)
        if box[2] - box[0] < min_width:
            box[0] = max(min(box[0], box[2] - min_width), left)
        if box[2] <= box[0]:                       # truly no room -> at least 1px
            box[2] = box[0] + 1
    return [PixelBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for b in xs]


def word_boxes_sane(
    boxes: list[PixelBox], line_box: PixelBox, *, min_width: int = 2, max_overlap: float = 0.5
) -> bool:
    """True when a line's per-word boxes are usable: none degenerate, and no pair
    overlapping more than ``max_overlap`` of the smaller box. Shaky word detection
    (e.g. Tesseract on a hard hand) can return nested or zero-width boxes; when it
    does, the builders fall back to the clean proportional split instead."""
    if not boxes:
        return True
    if any(b.width() < min_width or b.height() < min_width for b in boxes):
        return False
    for a, b in zip(boxes, boxes[1:]):
        ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
        smaller = min(a.width() * a.height(), b.width() * b.height())
        if smaller and (ix * iy) / smaller > max_overlap:
            return False
    return True


def word_boxes_for_line(line: TranscribedLine) -> list[tuple[str, PixelBox, int]]:
    """Per-word ``(text, box, confidence)`` for a line. Uses real engine boxes
    when the line carries usable ones; otherwise -- or when those boxes fail the
    sanity check -- falls back to the sanitized proportional split, so a line
    never emits word geometry worse than the synthesized boxes."""
    if line.words and word_boxes_sane([w.box for w in line.words], line.box):
        return [(w.text, w.box, w.confidence) for w in line.words]
    return [
        (text, wbox, line.confidence)
        for text, wbox in split_line_into_words(line.text, line.box)
    ]


def _bbox(b: PixelBox) -> str:
    return f"bbox {b.x0} {b.y0} {b.x1} {b.y1}"


def _union(lines: list[TranscribedLine], width: int, height: int) -> PixelBox:
    if not lines:
        return PixelBox(x0=0, y0=0, x1=width, y1=height)
    x0 = min(l.box.x0 for l in lines)
    y0 = min(l.box.y0 for l in lines)
    x1 = max(l.box.x1 for l in lines)
    y1 = max(l.box.y1 for l in lines)
    return PixelBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _q(tag: str) -> str:
    return f"{{{XHTML_NS}}}{tag}"


def _sub(parent, tag: str, **attrib) -> etree._Element:
    el = etree.SubElement(parent, _q(tag))
    for k, v in attrib.items():
        el.set(k.replace("_", "-") if k == "http_equiv" else k, v)
    return el


def build_hocr(
    pages: list[PageResult],
    *,
    ocr_system: str = "CursBreaker",
    language: str = "en",
) -> bytes:
    html = etree.Element(_q("html"), nsmap={None: XHTML_NS})
    html.set(f"{{{XML_NS}}}lang", language)

    head = _sub(html, "head")
    title = _sub(head, "title")
    title.text = "CursBreaker hOCR"
    _sub(head, "meta", http_equiv="Content-Type", content="text/html; charset=utf-8")
    _sub(head, "meta", name="ocr-system", content=ocr_system)
    _sub(head, "meta", name="ocr-capabilities", content=CAPABILITIES)

    body = _sub(html, "body")
    for pi, page in enumerate(pages, start=1):
        _build_page(body, page, pi, language=language)

    return etree.tostring(
        etree.ElementTree(html),
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        doctype=DOCTYPE,
    )


def _build_page(body, page: PageResult, pi: int, *, language: str) -> None:
    page_div = _sub(
        body,
        "div",
        **{
            "class": "ocr_page",
            "id": f"page_{pi}",
            "title": f'image "{page.image_name}"; bbox 0 0 {page.width} {page.height}; ppageno {pi - 1}',
        },
    )
    union = _union(page.lines, page.width, page.height)
    carea = _sub(
        page_div,
        "div",
        **{"class": "ocr_carea", "id": f"block_{pi}_1", "title": _bbox(union)},
    )
    par = _sub(
        carea,
        "p",
        **{
            "class": "ocr_par",
            "id": f"par_{pi}_1",
            "title": f"{_bbox(union)}; lang {language}",
        },
    )
    for li, line in enumerate(page.lines, start=1):
        # Approximate baseline ~20 % above the line's bottom edge (typical
        # descender depth). hOCR baseline offset is measured upward from the
        # bbox bottom and written as a negative number.
        descender = max(1, line.box.height() // 5)
        line_span = _sub(
            par,
            "span",
            **{
                "class": "ocr_line",
                "id": f"line_{pi}_{li}",
                "title": f"{_bbox(line.box)}; baseline 0 -{descender}; lang {language}",
            },
        )
        # Real engine boxes when usable; otherwise (or when those boxes fail the
        # sanity check) the sanitized proportional split.
        word_iter = word_boxes_for_line(line)
        for wi, (word_text, wbox, wconf) in enumerate(word_iter, start=1):
            w = _sub(
                line_span,
                "span",
                **{
                    "class": "ocrx_word",
                    "id": f"word_{pi}_{li}_{wi}",
                    "title": f"{_bbox(wbox)}; x_wconf {wconf}",
                },
            )
            w.text = word_text
