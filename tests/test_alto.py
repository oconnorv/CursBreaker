from lxml import etree

from cursbreaker.alto import ALTO_NS, build_alto
from cursbreaker.models import OcrWord, PageResult, PixelBox, TranscribedLine

NS = {"a": ALTO_NS}


def _page(lines, *, image_name="sample.png", width=800, height=600):
    return PageResult(
        image_name=image_name, width=width, height=height, lines=lines, plain_text="x"
    )


def test_build_alto_structure_and_geometry():
    lines = [
        TranscribedLine(text="hello world", box=PixelBox(x0=10, y0=10, x1=300, y1=40)),
        TranscribedLine(text="second line", box=PixelBox(x0=10, y0=50, x1=320, y1=80)),
    ]
    root = etree.fromstring(build_alto([_page(lines)]))

    # Root element, namespace and declared schema version.
    assert root.tag == f"{{{ALTO_NS}}}alto"
    assert root.get("SCHEMAVERSION") == "4.2"

    # Description: pixel measurement unit, source image, and the OCR software.
    assert root.xpath("//a:Description/a:MeasurementUnit/text()", namespaces=NS) == ["pixel"]
    assert root.xpath("//a:sourceImageInformation/a:fileName/text()", namespaces=NS) == ["sample.png"]
    assert root.xpath("//a:processingSoftware/a:softwareName/text()", namespaces=NS) == ["CursBreaker"]

    # One Page carrying the real pixel dimensions, with a PrintSpace.
    pages = root.xpath("//a:Layout/a:Page", namespaces=NS)
    assert len(pages) == 1
    assert (pages[0].get("WIDTH"), pages[0].get("HEIGHT")) == ("800", "600")
    assert pages[0].get("PHYSICAL_IMG_NR") == "1"
    assert root.xpath("//a:PrintSpace", namespaces=NS)

    # Two TextLines; the first carries its line box as HPOS/VPOS/WIDTH/HEIGHT.
    text_lines = root.xpath("//a:TextLine", namespaces=NS)
    assert len(text_lines) == 2
    assert (text_lines[0].get("HPOS"), text_lines[0].get("VPOS")) == ("10", "10")
    assert (text_lines[0].get("WIDTH"), text_lines[0].get("HEIGHT")) == ("290", "30")

    # Words become String elements with CONTENT and a confidence in [0, 1].
    words = text_lines[0].xpath("./a:String", namespaces=NS)
    assert [w.get("CONTENT") for w in words] == ["hello", "world"]
    assert all(0.0 <= float(w.get("WC")) <= 1.0 for w in words)


def test_build_alto_separates_words_with_sp():
    line = TranscribedLine(text="alpha beta gamma", box=PixelBox(x0=0, y0=0, x1=300, y1=20))
    root = etree.fromstring(build_alto([_page([line])]))
    text_line = root.xpath("//a:TextLine", namespaces=NS)[0]
    # n words -> n Strings and n-1 explicit <SP/> separators.
    assert len(text_line.xpath("./a:String", namespaces=NS)) == 3
    assert len(text_line.xpath("./a:SP", namespaces=NS)) == 2


def test_build_alto_uses_real_word_boxes_and_confidence():
    # A Tesseract-style line whose `words` carry per-word data: the builder must
    # use those exact boxes and confidences, NOT the proportional split.
    words = [
        OcrWord(text="hello",  box=PixelBox(x0=10,  y0=10, x1=100, y1=40), confidence=92),
        OcrWord(text="world!", box=PixelBox(x0=200, y0=10, x1=350, y1=40), confidence=70),
    ]
    line = TranscribedLine(
        text="hello world!",
        box=PixelBox(x0=10, y0=10, x1=350, y1=40),
        confidence=95,
        words=words,
    )
    root = etree.fromstring(build_alto([_page([line])]))
    strings = root.xpath("//a:String", namespaces=NS)
    assert [s.get("CONTENT") for s in strings] == ["hello", "world!"]
    # Real geometry shows through verbatim (HPOS=x0, WIDTH=x1-x0).
    assert (strings[0].get("HPOS"), strings[0].get("WIDTH")) == ("10", "90")
    assert (strings[1].get("HPOS"), strings[1].get("WIDTH")) == ("200", "150")
    # WC is the per-word confidence as a 0-1 float, not the line-level fallback.
    assert strings[0].get("WC") == "0.92"
    assert strings[1].get("WC") == "0.70"


def test_build_alto_falls_back_to_synthesized_words():
    # A Gemini-style line with no per-word data: builder still emits Strings via
    # the proportional split, inheriting the line-level confidence.
    line = TranscribedLine(
        text="alpha beta", box=PixelBox(x0=0, y0=0, x1=200, y1=20), confidence=80
    )
    root = etree.fromstring(build_alto([_page([line])]))
    strings = root.xpath("//a:String", namespaces=NS)
    assert [s.get("CONTENT") for s in strings] == ["alpha", "beta"]
    assert all(s.get("WC") == "0.80" for s in strings)  # 80 -> 0.80


def test_build_alto_escapes_special_characters():
    line = TranscribedLine(text="a < b & c", box=PixelBox(x0=0, y0=0, x1=100, y1=20))
    out = build_alto([_page([line])])
    # Must stay well-formed and round-trip the literal characters in CONTENT.
    root = etree.fromstring(out)
    contents = [s.get("CONTENT") for s in root.xpath("//a:String", namespaces=NS)]
    assert contents == ["a", "<", "b", "&", "c"]


def test_build_alto_multiple_pages_numbered():
    out = build_alto([_page([], image_name="p1.png"), _page([], image_name="p2.png")])
    root = etree.fromstring(out)
    pages = root.xpath("//a:Layout/a:Page", namespaces=NS)
    assert len(pages) == 2
    assert [p.get("PHYSICAL_IMG_NR") for p in pages] == ["1", "2"]


def test_build_alto_empty_page_has_printspace_without_textblock():
    root = etree.fromstring(build_alto([_page([])]))
    assert root.xpath("//a:PrintSpace", namespaces=NS)
    assert not root.xpath("//a:TextBlock", namespaces=NS)
    assert not root.xpath("//a:TextLine", namespaces=NS)


def test_build_alto_records_language_and_ocr_system():
    line = TranscribedLine(text="bonjour", box=PixelBox(x0=0, y0=0, x1=80, y1=20))
    out = build_alto(
        [_page([line])], ocr_system="CursBreaker (handwriting/two_pass)", language="fr"
    )
    root = etree.fromstring(out)
    assert root.xpath("//a:softwareName/text()", namespaces=NS) == [
        "CursBreaker (handwriting/two_pass)"
    ]
    assert "fr" in root.xpath("//a:processingStepSettings/text()", namespaces=NS)[0]
