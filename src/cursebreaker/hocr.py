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
    return out


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
    ocr_system: str = "CurseBreaker",
    language: str = "en",
) -> bytes:
    html = etree.Element(_q("html"), nsmap={None: XHTML_NS})
    html.set(f"{{{XML_NS}}}lang", language)

    head = _sub(html, "head")
    title = _sub(head, "title")
    title.text = "CurseBreaker hOCR"
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
        for wi, (word, wbox) in enumerate(
            split_line_into_words(line.text, line.box), start=1
        ):
            w = _sub(
                line_span,
                "span",
                **{
                    "class": "ocrx_word",
                    "id": f"word_{pi}_{li}_{wi}",
                    "title": f"{_bbox(wbox)}; x_wconf {line.confidence}",
                },
            )
            w.text = word
