import pytest
from lxml import etree

from cursbreaker.config import Settings
from cursbreaker.gemini_client import MockProvider
from cursbreaker.hocr import XHTML_NS
from cursbreaker.pipeline import process_batch, process_file

NS = {"x": XHTML_NS}


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
