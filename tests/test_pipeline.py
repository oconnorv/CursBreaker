import pytest
from PIL import Image, ImageDraw, ImageFont
from lxml import etree

from cursbreaker import tesseract_client
from cursbreaker.config import Settings
from cursbreaker.gemini_client import MockProvider
from cursbreaker.hocr import XHTML_NS
from cursbreaker.models import LineBox
from cursbreaker.pipeline import (
    estimate_usage,
    process_batch,
    process_file,
    process_page,
)

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
    settings = Settings(mode=mode)
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
    settings = Settings(pdf_dpi=120)
    result = process_file(pdf_path, MockProvider(), settings, out)

    assert result.n_pages == 2
    assert (out / "doc.txt").exists()
    assert (out / "doc_page_0001.png").exists()
    assert (out / "doc_page_0002.png").exists()

    root = etree.fromstring((out / "doc.hocr").read_bytes())
    assert len(root.xpath("//x:div[@class='ocr_page']", namespaces=NS)) == 2


def test_batch_isolates_failures(png_path, tmp_path):
    out = tmp_path / "out"
    settings = Settings()
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
    # Force status -> not installed so we hit the require_available guard
    # regardless of what's actually installed in the test env.
    monkeypatch.setattr(
        tesseract_client,
        "status",
        lambda settings=None: tesseract_client.TesseractStatus(
            wrapper_present=True,
            binary_found=False,
            error="Tesseract OCR engine not found. Install it ...",
        ),
    )
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(content_type="text")
    # process_batch is the layer that catches per-file errors; it must surface
    # the missing-Tesseract failure with a useful message instead of crashing.
    results = process_batch([page], MockProvider(), settings, out)
    assert results[0].error and "tesseract" in results[0].error.lower()


class _MatchingProvider(MockProvider):
    """A mock whose Gemini transcription matches the printed test page, so the
    refine step has something Tesseract can agree with."""

    _TEXT = "First printed line\nSecond printed line"

    def transcribe_text(self, image_png: bytes, mime: str = "image/png") -> str:
        return self._TEXT

    def detect_lines(self, image_png: bytes, mime: str = "image/png"):
        return [
            LineBox(text="First printed line", box_2d=[110, 40, 300, 960]),
            LineBox(text="Second printed line", box_2d=[440, 40, 640, 960]),
        ]


def _load_one(page, **overrides):
    from cursbreaker.images import load_pages

    return load_pages(
        page,
        preprocess=overrides.get("preprocess", True),
        max_dimension=overrides.get("max_dimension", 0),
        pdf_dpi=overrides.get("pdf_dpi", 300),
    )[0]


def test_refine_word_boxes_adopts_real_tesseract_boxes_keeping_gemini_text(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    page = _printed_page(tmp_path)
    settings = Settings(
        content_type="handwriting", mode="two_pass", refine_word_boxes=True
    )
    loaded = _load_one(page)
    result = process_page(loaded, _MatchingProvider(), settings)

    line = next(l for l in result.lines if l.text == "First printed line")
    # Refinement attached real per-word data; the *text* is still Gemini's words.
    assert line.words is not None
    assert [w.text for w in line.words] == ["First", "printed", "line"]
    # Every word matched a Tesseract word, so all carry the matched confidence
    # (synthesized fallback boxes would use interpolated_confidence instead).
    assert all(w.confidence == settings.word_confidence for w in line.words)
    # Real boxes advance left-to-right and sit within the page.
    xs = [w.box.x0 for w in line.words]
    assert xs == sorted(xs)
    assert all(0 <= w.box.x0 < w.box.x1 <= loaded.sent_width for w in line.words)


def test_refine_off_leaves_word_boxes_unsynthesized(tmp_path):
    # With the toggle off, the handwriting flow is unchanged: no per-word data,
    # so hOCR falls back to proportional splitting (words=None).
    page = _printed_page(tmp_path)
    settings = Settings(content_type="handwriting", refine_word_boxes=False)
    loaded = _load_one(page)
    result = process_page(loaded, _MatchingProvider(), settings)
    assert all(l.words is None for l in result.lines)


def test_refine_never_changes_transcription_text(tmp_path):
    # The whole point: the plain-text transcription must be byte-for-byte the
    # Gemini output regardless of what Tesseract reads.
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    page = _printed_page(tmp_path)
    loaded = _load_one(page)
    on = process_page(
        loaded,
        _MatchingProvider(),
        Settings(content_type="handwriting", refine_word_boxes=True),
    )
    off = process_page(
        loaded,
        _MatchingProvider(),
        Settings(content_type="handwriting", refine_word_boxes=False),
    )
    assert on.plain_text == off.plain_text == _MatchingProvider._TEXT


def test_refine_word_boxes_label_in_hocr(tmp_path):
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    out = tmp_path / "out"
    page = _printed_page(tmp_path)
    settings = Settings(
        content_type="handwriting", mode="two_pass", refine_word_boxes=True
    )
    process_file(page, _MatchingProvider(), settings, out)
    hocr = (out / f"{page.stem}.hocr").read_text("utf-8")
    assert "handwriting/two_pass+wordboxes" in hocr


# --- token usage + cost estimate ----------------------------------------- #

class _BillingProvider(MockProvider):
    """A mock that 'bills' a fixed amount per Gemini call so token deltas are
    non-zero, and reports a fixed input-token count for estimates."""

    def transcribe_text(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 20})
        return super().transcribe_text(image_png, mime)

    def detect_lines(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 5})
        return super().detect_lines(image_png, mime)

    def transcribe_with_boxes(self, image_png, mime="image/png"):
        self.usage.add_response({"prompt_token_count": 100, "candidates_token_count": 25})
        return super().transcribe_with_boxes(image_png, mime)

    def count_input_tokens(self, image_png, mime="image/png"):
        return 500


def test_process_file_records_per_file_token_usage(png_path, tmp_path):
    out = tmp_path / "out"
    prov = _BillingProvider()
    settings = Settings(content_type="handwriting", mode="two_pass")
    result = process_file(png_path, prov, settings, out)
    # Two-pass on one page = transcribe + detect = 2 calls.
    assert result.token_usage.calls == 2
    assert result.token_usage.input == 200
    assert result.token_usage.output == 25


def test_per_file_usage_is_a_delta_not_the_running_total(png_path, tmp_path):
    out = tmp_path / "out"
    prov = _BillingProvider()  # shared across both files
    settings = Settings(content_type="handwriting", mode="one_pass")
    first = process_file(png_path, prov, settings, out)
    second = process_file(png_path, prov, settings, out)
    # Each file reports only its own one call, even though the provider's
    # running total has grown to two.
    assert first.token_usage.calls == 1
    assert second.token_usage.calls == 1
    assert prov.usage.calls == 2


def test_estimate_text_mode_is_free(png_path):
    d = estimate_usage([png_path], _BillingProvider(), Settings(content_type="text"))
    assert d["calls"] == 0
    assert d["input"] == 0
    assert d["output"] == 0
    assert d["cost"] is None


def test_estimate_counts_two_calls_per_page_in_two_pass(png_path):
    settings = Settings(content_type="handwriting", mode="two_pass")
    d = estimate_usage([png_path], _BillingProvider(), settings)
    assert d["pages"] == 1
    assert d["calls"] == 2                 # transcribe + detect
    assert d["input"] == 500 * 1 * 2       # per-page input * pages * calls


def test_estimate_scales_input_by_page_count(pdf_path):
    settings = Settings(content_type="handwriting", mode="one_pass", pdf_dpi=120)
    d = estimate_usage([pdf_path], _BillingProvider(), settings)
    assert d["pages"] == 2
    assert d["calls"] == 2                  # 2 pages * 1 call
    assert d["input"] == 500 * 2 * 1        # first-page count scaled by 2 pages


def test_estimate_prices_automatically_from_selected_model(png_path):
    # Flat-priced model: cost comes straight from the catalog, no manual entry.
    settings = Settings(
        content_type="handwriting",
        mode="one_pass",
        transcription_model="gemini-3.5-flash",  # $1.50 in / $9.00 out
    )
    d = estimate_usage([png_path], _BillingProvider(), settings)
    # input = 500 tokens; assumed output = 800 tokens (one call).
    expected = 500 / 1_000_000 * 1.50 + 800 / 1_000_000 * 9.00
    assert d["cost"] == pytest.approx(expected)
    # The model + the rates it used are echoed back for the UI to display.
    assert d["model"] == "gemini-3.5-flash"
    assert d["price_input_per_mtok"] == 1.50
    assert d["price_output_per_mtok"] == 9.00
    assert d["prices_as_of"]
    assert d["assumed_output_tokens_per_call"] > 0


def test_estimate_no_cost_for_uncatalogued_model(png_path):
    settings = Settings(
        content_type="handwriting", mode="one_pass", transcription_model="some-old-model"
    )
    d = estimate_usage([png_path], _BillingProvider(), settings)
    assert d["cost"] is None            # unknown price -> tokens only, no dollars
    assert d["input"] == 500
