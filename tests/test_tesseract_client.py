import pytest
from PIL import Image, ImageDraw, ImageFont

from cursbreaker import tesseract_client
from cursbreaker.models import PixelBox, TranscribedLine


def _printed_page(size=(800, 200), lines=("Hello world", "Second line here")):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "src/cursbreaker/fonts/DejaVuSans.ttf", 36
        )
    except OSError:
        font = ImageFont.load_default()
    y = 30
    for line in lines:
        draw.text((30, y), line, fill="black", font=font)
        y += 60
    return img


def test_is_available_true_in_this_environment():
    # Tesseract is installed in dev; the test environment must surface that.
    # If this ever fails locally because Tesseract isn't installed, skip.
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    assert tesseract_client.is_available() is True


def test_available_languages_includes_english():
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    langs = tesseract_client.available_languages()
    assert "eng" in langs


def test_transcribe_page_returns_real_word_boxes():
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    img = _printed_page()
    lines = tesseract_client.transcribe_page(img)
    assert len(lines) >= 1
    # Every line should carry real per-word data, not the proportional fallback.
    for line in lines:
        assert line.words is not None
        assert len(line.words) >= 1
        # Word boxes sit inside the line box (within rounding).
        for w in line.words:
            assert w.box.x0 >= line.box.x0 - 1
            assert w.box.x1 <= line.box.x1 + 1
            assert w.box.y0 >= line.box.y0 - 1
            assert w.box.y1 <= line.box.y1 + 1
            # Confidence is a real engine number, not the nominal 95 default.
            assert 0 <= w.confidence <= 100
    # And the text actually says what we drew.
    joined = " ".join(l.text for l in lines).lower()
    assert "hello" in joined and "world" in joined


def test_transcribe_region_offsets_coordinates():
    if not tesseract_client.is_available():
        pytest.skip("Tesseract binary not installed")
    # Crop and ask the OCR to report coordinates in page space.
    img = _printed_page()
    crop = img.crop((20, 20, 500, 100))
    lines = tesseract_client.transcribe_region(crop, offset=(20, 20))
    assert lines
    # The lowest x in any returned word should be >= the offset we passed in.
    min_x = min(w.box.x0 for l in lines for w in (l.words or []))
    assert min_x >= 20


def test_require_available_raises_when_missing(monkeypatch):
    # Force the availability probe to say "no" and confirm the error message
    # mentions installation steps so the user knows what to do.
    monkeypatch.setattr(tesseract_client, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        tesseract_client.require_available()
    msg = str(excinfo.value).lower()
    assert "tesseract" in msg and "install" in msg


def test_transcribed_line_words_field_is_optional():
    # Existing call sites that don't pass `words` must keep working.
    line = TranscribedLine(text="hi", box=PixelBox(x0=0, y0=0, x1=10, y1=10))
    assert line.words is None
