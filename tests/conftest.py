import fitz
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def png_path(tmp_path):
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "hello world", fill="black")
    p = tmp_path / "sample.png"
    img.save(p)
    return p


@pytest.fixture
def pdf_path(tmp_path):
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1} text")
    p = tmp_path / "doc.pdf"
    doc.save(p)
    doc.close()
    return p
