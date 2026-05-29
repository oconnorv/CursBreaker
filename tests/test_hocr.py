from lxml import etree

from cursebreaker.hocr import (
    XHTML_NS,
    build_hocr,
    normalized_to_pixel,
    split_line_into_words,
)
from cursebreaker.models import PageResult, PixelBox, TranscribedLine

NS = {"x": XHTML_NS}


def test_normalized_to_pixel_full_box():
    box = normalized_to_pixel([0, 0, 1000, 1000], 800, 600)
    assert (box.x0, box.y0, box.x1, box.y1) == (0, 0, 800, 600)


def test_normalized_to_pixel_midpoint_and_order():
    box = normalized_to_pixel([100, 200, 300, 400], 1000, 1000)
    # [ymin, xmin, ymax, xmax] -> x0=200, y0=100, x1=400, y1=300
    assert (box.x0, box.y0, box.x1, box.y1) == (200, 100, 400, 300)


def test_normalized_to_pixel_swaps_inverted():
    box = normalized_to_pixel([300, 400, 100, 200], 1000, 1000)
    assert box.x0 <= box.x1 and box.y0 <= box.y1


def test_word_split_is_monotonic_and_bounded():
    box = PixelBox(x0=100, y0=50, x1=900, y1=90)
    words = split_line_into_words("alpha bb ccccc", box)
    assert [w for w, _ in words] == ["alpha", "bb", "ccccc"]
    prev = box.x0
    for _, wb in words:
        assert wb.x0 >= box.x0 and wb.x1 <= box.x1
        assert wb.x1 > wb.x0
        assert wb.x0 >= prev - 1  # left-to-right
        prev = wb.x1
        assert (wb.y0, wb.y1) == (box.y0, box.y1)


def test_single_word_spans_line():
    box = PixelBox(x0=10, y0=0, x1=110, y1=20)
    (word, wb), = split_line_into_words("solo", box)
    assert word == "solo"
    assert (wb.x0, wb.x1) == (10, 110)


def _page(lines):
    return PageResult(
        image_name="sample.png", width=800, height=600, lines=lines, plain_text="x"
    )


def test_build_hocr_structure_and_bbox():
    lines = [
        TranscribedLine(text="hello world", box=PixelBox(x0=10, y0=10, x1=300, y1=40)),
        TranscribedLine(text="second line", box=PixelBox(x0=10, y0=50, x1=320, y1=80)),
    ]
    out = build_hocr([_page(lines)])
    root = etree.fromstring(out)

    assert root.xpath("//x:meta[@name='ocr-system']/@content", namespaces=NS)
    caps = root.xpath("//x:meta[@name='ocr-capabilities']/@content", namespaces=NS)[0]
    assert "ocrx_word" in caps and "ocr_line" in caps

    page = root.xpath("//x:div[@class='ocr_page']", namespaces=NS)[0]
    assert 'image "sample.png"' in page.get("title")
    assert "bbox 0 0 800 600" in page.get("title")

    line_spans = root.xpath("//x:span[@class='ocr_line']", namespaces=NS)
    assert len(line_spans) == 2
    assert "bbox 10 10 300 40" in line_spans[0].get("title")

    words = line_spans[0].xpath("./x:span[@class='ocrx_word']", namespaces=NS)
    assert [w.text for w in words] == ["hello", "world"]
    assert "x_wconf" in words[0].get("title")


def test_build_hocr_escapes_special_characters():
    lines = [TranscribedLine(text="a < b & c", box=PixelBox(x0=0, y0=0, x1=100, y1=20))]
    out = build_hocr([_page(lines)])
    # Must remain well-formed and round-trip the literal characters.
    root = etree.fromstring(out)
    words = root.xpath("//x:span[@class='ocrx_word']/text()", namespaces=NS)
    assert words == ["a", "<", "b", "&", "c"]


def test_build_hocr_multiple_pages():
    out = build_hocr([_page([]), _page([])])
    root = etree.fromstring(out)
    assert len(root.xpath("//x:div[@class='ocr_page']", namespaces=NS)) == 2


def test_build_hocr_emits_language_and_baseline_and_per_line_confidence():
    lines = [
        TranscribedLine(
            text="detected line",
            box=PixelBox(x0=10, y0=10, x1=300, y1=110),  # height 100
            confidence=95,
        ),
        TranscribedLine(
            text="guessed line",
            box=PixelBox(x0=10, y0=130, x1=300, y1=230),
            confidence=60,
        ),
    ]
    out = build_hocr([_page(lines)], language="fr")
    root = etree.fromstring(out)

    # html-level language picks up the setting
    xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
    assert xml_lang == "fr"

    # ocr_par carries the language too
    par_title = root.xpath("//x:p[@class='ocr_par']/@title", namespaces=NS)[0]
    assert "lang fr" in par_title

    # Each line title contains a real baseline offset (not 0 0) and the lang.
    line_titles = root.xpath("//x:span[@class='ocr_line']/@title", namespaces=NS)
    assert all("lang fr" in t for t in line_titles)
    # 100 / 5 = 20 so the first line gets "baseline 0 -20".
    assert "baseline 0 -20" in line_titles[0]
    assert "baseline 0 0" not in line_titles[0]

    # Word confidences come from each line, not a shared default.
    line_spans = root.xpath("//x:span[@class='ocr_line']", namespaces=NS)
    confs_line1 = [
        w.get("title") for w in line_spans[0].xpath(
            "./x:span[@class='ocrx_word']", namespaces=NS
        )
    ]
    confs_line2 = [
        w.get("title") for w in line_spans[1].xpath(
            "./x:span[@class='ocrx_word']", namespaces=NS
        )
    ]
    assert all("x_wconf 95" in t for t in confs_line1)
    assert all("x_wconf 60" in t for t in confs_line2)
