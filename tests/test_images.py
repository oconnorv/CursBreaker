from PIL import Image, UnidentifiedImageError

from cursebreaker.images import is_supported, load_pages


def test_supported_extensions():
    assert is_supported("a.tif") and is_supported("b.JPEG") and is_supported("c.pdf")
    assert not is_supported("d.docx")


def test_load_single_image_keeps_dimensions(png_path):
    pages = load_pages(png_path)
    assert len(pages) == 1
    page = pages[0]
    assert (page.orig_width, page.orig_height) == (800, 600)
    # No resize by default => the image sent to the API matches the original.
    assert (page.sent_width, page.sent_height) == (800, 600)
    assert page.output_stem == "sample"
    assert page.to_png_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_resize_scales_sent_image_only(png_path):
    pages = load_pages(png_path, max_dimension=400)
    page = pages[0]
    assert max(page.sent_width, page.sent_height) == 400
    assert (page.orig_width, page.orig_height) == (800, 600)


def test_load_normal_tiff(tmp_path):
    p = tmp_path / "scan.tif"
    Image.new("RGB", (640, 480), "white").save(p)
    pages = load_pages(p)
    assert len(pages) == 1
    assert (pages[0].orig_width, pages[0].orig_height) == (640, 480)


def test_tiff_falls_back_to_fitz_when_pillow_cannot_decode(tmp_path, monkeypatch):
    # Save a real TIFF that fitz can open; force the Pillow path to fail.
    p = tmp_path / "tricky.tif"
    Image.new("RGB", (400, 300), "white").save(p)

    real_open = Image.open

    def reject_tiff(path, *args, **kwargs):
        if str(path).lower().endswith((".tif", ".tiff")):
            raise UnidentifiedImageError("simulated Pillow decoder failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("cursebreaker.images.Image.open", reject_tiff)

    pages = load_pages(p, pdf_dpi=72)  # zoom=1.0 in the fitz fallback
    assert len(pages) == 1
    # Falling back through fitz should still give us a usable page image.
    assert pages[0].sent_width > 0 and pages[0].sent_height > 0


def test_pdf_rasterizes_each_page(pdf_path):
    pages = load_pages(pdf_path, pdf_dpi=150)
    assert len(pages) == 2
    # 612pt * 150/72 ~= 1275 px wide.
    assert abs(pages[0].sent_width - 1275) <= 2
    assert pages[0].output_stem == "doc_page_0001"
    assert pages[1].output_stem == "doc_page_0002"
