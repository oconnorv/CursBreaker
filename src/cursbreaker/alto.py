"""Build valid ALTO XML (schema v4.2) from transcribed lines.

ALTO (Analyzed Layout and Text Object, a Library of Congress standard --
http://www.loc.gov/standards/alto/) is the preservation/DAMS counterpart to
hOCR. A ``Layout`` holds one ``Page`` per page image; each ``Page`` carries a
``PrintSpace`` containing ``TextBlock`` -> ``TextLine`` -> ``String`` elements.
Every block records pixel geometry (``HPOS``/``VPOS``/``WIDTH``/``HEIGHT``,
origin top-left) and each ``String`` an optional word confidence ``WC`` in
``[0, 1]``. Pairing this file with the page image makes the text searchable and
locatable, and lets ALTO/METS-based systems (e.g. Veridian and many library
repositories) ingest CursBreaker output directly -- complementing the hOCR we
already emit for the IIIF/Islandora world.

We emit one ALTO document per source file with one ``<Page>`` per page,
mirroring the hOCR builder. Word boxes reuse real engine data when a line
carries it (Tesseract); otherwise they are synthesized by splitting the line
box in proportion to word length (the same approach, and the same helper, as
hOCR), since Gemini returns reliable *line* boxes but not per-word boxes.
"""

from __future__ import annotations

from lxml import etree

from .hocr import split_line_into_words
from .models import PageResult, PixelBox, TranscribedLine

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_VERSION = "4.2"
SCHEMA_LOCATION = (
    "http://www.loc.gov/standards/alto/ns-v4# "
    "http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
)


def _q(tag: str) -> str:
    return f"{{{ALTO_NS}}}{tag}"


def _sub(parent, tag: str, **attrib) -> etree._Element:
    el = etree.SubElement(parent, _q(tag))
    for k, v in attrib.items():
        el.set(k, str(v))
    return el


def _geom(b: PixelBox) -> dict[str, str]:
    """ALTO geometry attributes for a pixel box (origin top-left)."""
    return {
        "HPOS": str(b.x0),
        "VPOS": str(b.y0),
        "WIDTH": str(b.width()),
        "HEIGHT": str(b.height()),
    }


def _union(lines: list[TranscribedLine], width: int, height: int) -> PixelBox:
    if not lines:
        return PixelBox(x0=0, y0=0, x1=width, y1=height)
    return PixelBox(
        x0=min(l.box.x0 for l in lines),
        y0=min(l.box.y0 for l in lines),
        x1=max(l.box.x1 for l in lines),
        y1=max(l.box.y1 for l in lines),
    )


def build_alto(
    pages: list[PageResult],
    *,
    ocr_system: str = "CursBreaker",
    language: str = "en",
) -> bytes:
    alto = etree.Element(_q("alto"), nsmap={None: ALTO_NS, "xsi": XSI_NS})
    alto.set("SCHEMAVERSION", SCHEMA_VERSION)
    alto.set(f"{{{XSI_NS}}}schemaLocation", SCHEMA_LOCATION)

    # Description: order matters (MeasurementUnit, sourceImageInformation,
    # OCRProcessing), as the ALTO XSD models it as a sequence.
    desc = _sub(alto, "Description")
    _sub(desc, "MeasurementUnit").text = "pixel"
    if pages:
        # ALTO carries a single source image at the Description level; for the
        # common one-image-per-file case this is exact, and for multi-page docs
        # the per-page ordering lives on Page/@PHYSICAL_IMG_NR below.
        src = _sub(desc, "sourceImageInformation")
        _sub(src, "fileName").text = pages[0].image_name
    ocrp = _sub(desc, "OCRProcessing", ID="OCR_1")
    step = _sub(ocrp, "ocrProcessingStep")
    _sub(step, "processingStepSettings").text = f"language:{language}"
    software = _sub(step, "processingSoftware")
    _sub(software, "softwareName").text = ocr_system

    layout = _sub(alto, "Layout")
    for pi, page in enumerate(pages, start=1):
        _build_page(layout, page, pi)

    return etree.tostring(
        etree.ElementTree(alto),
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )


def _build_page(layout, page: PageResult, pi: int) -> None:
    page_el = _sub(
        layout,
        "Page",
        ID=f"page_{pi}",
        PHYSICAL_IMG_NR=pi,
        WIDTH=page.width,
        HEIGHT=page.height,
    )
    print_space = _sub(
        page_el,
        "PrintSpace",
        HPOS=0,
        VPOS=0,
        WIDTH=page.width,
        HEIGHT=page.height,
    )
    if not page.lines:
        return
    # One TextBlock per page spanning the union of its lines (mirrors hOCR's
    # single ocr_carea); the geometry stays meaningful for block-level consumers.
    block = _sub(
        print_space,
        "TextBlock",
        ID=f"block_{pi}_1",
        **_geom(_union(page.lines, page.width, page.height)),
    )
    for li, line in enumerate(page.lines, start=1):
        line_el = _sub(block, "TextLine", ID=f"line_{pi}_{li}", **_geom(line.box))
        # Prefer real per-word data (e.g. Tesseract) when the line carries it;
        # otherwise fall back to proportionally splitting the line box, the only
        # option for Gemini-sourced lines.
        if line.words:
            word_iter = [(w.text, w.box, w.confidence) for w in line.words]
        else:
            word_iter = [
                (text, wbox, line.confidence)
                for text, wbox in split_line_into_words(line.text, line.box)
            ]
        for wi, (word_text, wbox, wconf) in enumerate(word_iter, start=1):
            if wi > 1:
                # ALTO marks inter-word whitespace with an explicit <SP/>.
                _sub(line_el, "SP")
            conf = max(0, min(100, wconf)) / 100.0
            _sub(
                line_el,
                "String",
                ID=f"word_{pi}_{li}_{wi}",
                **_geom(wbox),
                CONTENT=word_text,
                WC=f"{conf:.2f}",
            )
