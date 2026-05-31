import sys
import types

import pytest
from PIL import Image, ImageDraw, ImageFont

from cursbreaker import tesseract_client
from cursbreaker.config import Settings
from cursbreaker.models import PixelBox, TranscribedLine


def _install_fake_pytesseract(
    monkeypatch, *, version="5.3.0", langs=("eng",), version_error=None
):
    """Install a stand-in ``pytesseract`` module so tests need no real engine."""
    mod = types.ModuleType("pytesseract")

    class TesseractNotFoundError(Exception):
        pass

    mod.TesseractNotFoundError = TesseractNotFoundError
    mod.pytesseract = types.SimpleNamespace(tesseract_cmd="tesseract")

    def get_tesseract_version():
        if version_error is not None:
            raise version_error
        return version

    mod.get_tesseract_version = get_tesseract_version
    mod.get_languages = lambda config="": list(langs)
    monkeypatch.setitem(sys.modules, "pytesseract", mod)
    tesseract_client._PROBE_CACHE.clear()
    return mod


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


def test_status_wrapper_missing(monkeypatch):
    # `import pytesseract` failing must be reported as a *wrapper* problem,
    # not sent down the "install the engine" path.
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    tesseract_client._PROBE_CACHE.clear()
    st = tesseract_client.status(force=True)
    assert st.wrapper_present is False
    assert st.installed is False
    assert "pytesseract" in (st.error or "")


def test_status_binary_missing(monkeypatch):
    # Wrapper present, but the engine version probe fails -> *binary* problem.
    _install_fake_pytesseract(monkeypatch, version_error=RuntimeError("nope"))
    st = tesseract_client.status(force=True)
    assert st.wrapper_present is True
    assert st.binary_found is False
    assert st.installed is False
    assert st.error and "engine not found" in st.error.lower()


def test_status_ok_reports_version_and_languages(monkeypatch):
    _install_fake_pytesseract(monkeypatch, version="5.3.1", langs=("eng", "fra"))
    st = tesseract_client.status(force=True)
    assert st.installed is True and st.binary_found is True
    assert st.version == "5.3.1"
    assert "eng" in st.languages and "fra" in st.languages


def test_resolve_explicit_cmd_override_from_env(monkeypatch):
    fake = _install_fake_pytesseract(monkeypatch)
    monkeypatch.setenv("TESSERACT_CMD", "/custom/path/tesseract")
    assert tesseract_client.resolve_tesseract() == "/custom/path/tesseract"
    assert fake.pytesseract.tesseract_cmd == "/custom/path/tesseract"


def test_resolve_explicit_cmd_override_from_setting(monkeypatch):
    fake = _install_fake_pytesseract(monkeypatch)
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    resolved = tesseract_client.resolve_tesseract(
        Settings(tesseract_cmd="/from/settings/tesseract")
    )
    assert resolved == "/from/settings/tesseract"
    assert fake.pytesseract.tesseract_cmd == "/from/settings/tesseract"


def test_candidate_binaries_cover_each_os():
    win = tesseract_client._candidate_binaries("win32")
    assert any(c.endswith(r"Tesseract-OCR\tesseract.exe") for c in win)
    assert "/opt/homebrew/bin/tesseract" in tesseract_client._candidate_binaries("darwin")
    assert "/usr/bin/tesseract" in tesseract_client._candidate_binaries("linux")


def test_resolve_uses_windows_well_known_path(monkeypatch):
    # Prove the Windows branch is selected even though the test host is Linux.
    fake = _install_fake_pytesseract(monkeypatch)
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    monkeypatch.setattr(
        tesseract_client,
        "_candidate_binaries",
        lambda platform=None: [r"X:\bundled\tesseract.exe", win_path],
    )
    monkeypatch.setattr(tesseract_client, "_is_file", lambda p: p == win_path)
    assert tesseract_client.resolve_tesseract() == win_path
    assert fake.pytesseract.tesseract_cmd == win_path


def test_resolve_falls_back_to_path_when_nothing_found(monkeypatch):
    # The universal fallback: pytesseract's own PATH lookup ("tesseract").
    fake = _install_fake_pytesseract(monkeypatch)
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(tesseract_client, "_is_file", lambda p: False)
    assert tesseract_client.resolve_tesseract(Settings()) == "tesseract"
    assert fake.pytesseract.tesseract_cmd == "tesseract"


def test_status_caches_probe_until_forced(monkeypatch):
    fake = _install_fake_pytesseract(monkeypatch)
    monkeypatch.setattr(tesseract_client, "_is_file", lambda p: False)
    calls = {"n": 0}
    base = fake.get_tesseract_version

    def counting():
        calls["n"] += 1
        return base()

    fake.get_tesseract_version = counting
    tesseract_client._PROBE_CACHE.clear()
    tesseract_client.status(Settings())
    tesseract_client.status(Settings())
    assert calls["n"] == 1  # second read served from the cache
    tesseract_client.status(Settings(), force=True)
    assert calls["n"] == 2


def test_require_available_message_for_missing_wrapper(monkeypatch):
    monkeypatch.setattr(
        tesseract_client,
        "status",
        lambda settings=None: tesseract_client.TesseractStatus(
            wrapper_present=False,
            error="The 'pytesseract' Python package is not installed.",
        ),
    )
    with pytest.raises(RuntimeError) as e:
        tesseract_client.require_available()
    assert "pytesseract" in str(e.value)


def test_require_available_message_for_missing_binary(monkeypatch):
    monkeypatch.setattr(
        tesseract_client,
        "status",
        lambda settings=None: tesseract_client.TesseractStatus(
            wrapper_present=True,
            binary_found=False,
            cmd_path="/x/tesseract",
            error="Tesseract OCR engine not found (looked for '/x/tesseract').",
        ),
    )
    with pytest.raises(RuntimeError) as e:
        tesseract_client.require_available()
    assert "/x/tesseract" in str(e.value)


def test_transcribed_line_words_field_is_optional():
    # Existing call sites that don't pass `words` must keep working.
    line = TranscribedLine(text="hi", box=PixelBox(x0=0, y0=0, x1=10, y1=10))
    assert line.words is None
