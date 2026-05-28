import fitz
from PIL import Image

from cursebreaker.models import PageResult, PixelBox, TranscribedLine
from cursebreaker.searchable_pdf import build_searchable_pdf


def _png(tmp_path, name, size=(400, 200)):
    p = tmp_path / name
    Image.new("RGB", size, "white").save(p)
    return p


def test_pdf_contains_invisible_text_and_image(tmp_path):
    img = _png(tmp_path, "page.png")
    page = PageResult(
        image_name="page.png",
        width=400,
        height=200,
        lines=[
            TranscribedLine(
                text="hello world", box=PixelBox(x0=10, y0=20, x1=300, y1=60)
            ),
            TranscribedLine(
                text="second line here",
                box=PixelBox(x0=10, y0=80, x1=380, y1=120),
            ),
        ],
        plain_text="hello world\nsecond line here",
    )

    pdf_bytes = build_searchable_pdf([page], [img])
    out = tmp_path / "out.pdf"
    out.write_bytes(pdf_bytes)

    with fitz.open(out) as doc:
        assert doc.page_count == 1
        text = doc[0].get_text()
        for word in ("hello", "world", "second", "line", "here"):
            assert word in text, f"missing {word!r} in extracted text"
        # The page image is embedded.
        assert len(doc[0].get_images()) >= 1


def test_pdf_multipage_matches_input(tmp_path):
    pages, images = [], []
    for i in range(3):
        p = _png(tmp_path, f"p{i}.png")
        images.append(p)
        pages.append(
            PageResult(
                image_name=p.name,
                width=400,
                height=200,
                lines=[
                    TranscribedLine(
                        text=f"page {i + 1}",
                        box=PixelBox(x0=10, y0=10, x1=200, y1=50),
                    )
                ],
                plain_text=f"page {i + 1}",
            )
        )
    pdf_bytes = build_searchable_pdf(pages, images)
    out = tmp_path / "multi.pdf"
    out.write_bytes(pdf_bytes)
    with fitz.open(out) as doc:
        assert doc.page_count == 3
        for i in range(3):
            assert f"page" in doc[i].get_text()


def test_length_mismatch_raises(tmp_path):
    img = _png(tmp_path, "x.png")
    page = PageResult(
        image_name="x.png", width=400, height=200, lines=[], plain_text=""
    )
    try:
        build_searchable_pdf([page, page], [img])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched lengths")
