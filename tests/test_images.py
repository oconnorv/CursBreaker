from PIL import Image, TiffImagePlugin, UnidentifiedImageError

from cursbreaker.images import count_content_pages, is_supported, load_pages


def _tiff_with_thumbnail(path):
    """A two-frame TIFF: a 'real' page plus a thumbnail flagged via the
    NewSubfileType tag (bit 0 = reduced-resolution version of another image)."""
    main = Image.new("RGB", (800, 600), "white")
    thumb = Image.new("RGB", (100, 75), "gray")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[254] = 1
    thumb.encoderinfo = {"tiffinfo": ifd}
    main.save(path, format="TIFF", save_all=True, append_images=[thumb])
    return path


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

    monkeypatch.setattr("cursbreaker.images.Image.open", reject_tiff)

    pages = load_pages(p, pdf_dpi=72)  # zoom=1.0 in the fitz fallback
    assert len(pages) == 1
    # Falling back through fitz should still give us a usable page image.
    assert pages[0].sent_width > 0 and pages[0].sent_height > 0


def test_count_content_pages_skips_tiff_thumbnail(tmp_path):
    p = _tiff_with_thumbnail(tmp_path / "scan_with_thumb.tif")
    # Pillow reports 2 frames; only one is real content.
    with Image.open(p) as im:
        assert getattr(im, "n_frames", 1) == 2
    assert count_content_pages(p) == 1


def test_load_pages_skips_tiff_thumbnail_frame(tmp_path):
    p = _tiff_with_thumbnail(tmp_path / "scan.tif")
    pages = load_pages(p)
    assert len(pages) == 1
    # The content frame is the larger one (800x600), not the 100x75 thumbnail.
    assert (pages[0].orig_width, pages[0].orig_height) == (800, 600)


def test_pdf_rasterizes_each_page(pdf_path):
    pages = load_pages(pdf_path, pdf_dpi=150)
    assert len(pages) == 2
    # 612pt * 150/72 ~= 1275 px wide.
    assert abs(pages[0].sent_width - 1275) <= 2
    assert pages[0].output_stem == "doc_page_0001"
    assert pages[1].output_stem == "doc_page_0002"


def test_iter_pages_renders_pdf_lazily(pdf_path, monkeypatch):
    # The whole point of the lazy loader: rendering one page does NOT rasterize
    # the rest of the document (so a 48-page PDF can be cancelled promptly and
    # only one page sits in memory at a time).
    from cursbreaker import images

    calls = {"n": 0}
    real = images.Image.frombytes

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(images.Image, "frombytes", counting)
    it = images.iter_pages(pdf_path, pdf_dpi=72)
    next(it)                       # render only the first page
    assert calls["n"] == 1         # page 2 of the 2-page PDF is NOT rendered yet
    next(it)
    assert calls["n"] == 2         # advancing renders the next page
    it.close()
