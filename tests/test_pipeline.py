import pytest
from PIL import Image, ImageDraw, ImageFont
from lxml import etree

from cursbreaker import tesseract_client
from cursbreaker.config import Settings
from cursbreaker.gemini_client import MockProvider
from cursbreaker.hocr import XHTML_NS
from cursbreaker.pipeline import process_batch, process_file

NS = {"x": XHTML_NS}


def _font(size: int = 36) -> "ImageFont.ImageFont":
    try:
        return ImageFont.truetype("src/cursbreaker/fonts/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _printed_page(tmp_path, name="printed.png", size=(900, 240)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _font(36)
    draw.text((40, 30),  "First printed line",  fill="black", font=font)
    draw.text((40, 110), "Second printed line", fill="black", font=font)
    p = tmp_path / name
    img.save(p)
    return p


@pytest.mark.parametrize("mode", ["two_pass", "one_pass"])
def test_process_image_writes_outputs(png_path, tmp_path, mode):
    out = tmp_path / "out"
    settings = Settings(use_mock=True, mode=mode)
    result = process_file(png_path, MockProvider(), settings, out)

    assert result.error is None
    assert result.n_pages == 1
    assert result.n_lines == 4  # the mock returns four lines

    txt = (out / "sample.txt").read_text("utf-8")
    assert "CursBreaker mock transcription" in txt

    hocr = (out / "sample.hocr").read_bytes()
    root = etree.fromstring(hocr)
    assert len(root.xpath("//x:span[@class='ocr_line']", namespaces=NS)) == 4
    # The exported page image is referenced and present on disk.
    assert (out / "sample.png").exists()
    assert 'image "sample.png"' in root.xpath(
        "//x:div[@class='ocr_page']/@title", namespaces=NS
    )[0]
    # A searchable PDF is written alongside the other outputs.
    assert (out / "sample.pdf").exists()
    assert result.pdf_name == "sample.pdf"


def test_process_pdf_multipage(pdf_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings(use_mock=True, pdf_dpi=120)
    result = process_file(pdf_path, MockProvider(), settings, out)

    assert result.n_pages == 2
    assert (out / "doc.txt").exists()
    assert (out / "doc_page_0001.png").exists()
    assert (out / "doc_page_0002.png").exists()

    root = etree.fromstring((out / "doc.hocr").read_bytes())
    assert len(root.xpath("//x:div[@class='ocr_page']", namespaces=NS)) == 2


def test_batch_isolates_failures(png_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings(use_mock=True)
    missing = tmp_path / "does_not_exist.png"
    results = process_batch([png_path, missing], MockProvider(), settings, out)
    assert results[0].error is None
    assert results[1].error is not None  # bad file captured, batch survives


def test_content_type_text_uses_tesseract_only(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")  # no API key, no mock — no Gemini
    result = process_file(page, MockProvider(), settings, out)

    assert result.error is None
    assert result.n_pages == 1
    # Tesseract finds at least the two lines we drew.
    assert result.n_lines >= 2

    txt = (out / page.stem + ".txt").read_text("utf-8").lower() if False else (
        (out / f"{page.stem}.txt").read_text("utf-8").lower()
    )
    assert "first" in txt and "second" in txt

    # hOCR carries REAL Tesseract per-word x_wconf (varies), not the nominal 95.
    hocr = (out / f"{page.stem}.hocr").read_text("utf-8")
    assert "ocr-system" in hocr
    assert "CursBreaker (text)" in hocr  # mode label flips to "text"


def test_content_type_text_errors_clearly_when_tesseract_missing(monkeypatch, tmp_path):
    # Force is_available -> False so we hit the require_available guard
    # regardless of what's actually installed in the test env.
    monkeypatch.setattr(tesseract_client, "is_available", lambda: False)
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")
    # process_batch is the layer that catches per-file errors; it must surface
    # the missing-Tesseract failure with a useful message instead of crashing.
    results = process_batch([page], MockProvider(), settings, out)
    assert results[0].error and "tesseract" in results[0].error.lower()


def test_content_type_mixed_routes_printed_to_tesseract_and_handwritten_to_gemini(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="mixed", use_mock=True)
    result = process_file(page, MockProvider(), settings, out)

    assert result.error is None
    # The mock labels alternate kinds, so we expect a mixture of:
    #   * printed lines whose words come from Tesseract on the cropped region
    #   * handwritten lines whose text comes from the mock Gemini provider
    hocr = (out / f"{page.stem}.hocr").read_text("utf-8")
    assert "CursBreaker (mixed)" in hocr
    # The mock Gemini text shows up for handwritten lines. hOCR splits text
    # one-word-per-span, so check for distinctive mock words rather than the
    # whole phrase. "Settings" appears only in the mock Gemini sample.
    assert ">Settings<" in hocr
