from cursebreaker.align import align_lines, sort_for_reading_order
from cursebreaker.models import LineBox


def _box(text, ymin):
    return LineBox(text=text, box_2d=[ymin, 50, ymin + 30, 950])


def test_sort_for_reading_order_is_column_major():
    # Boxes returned in row-major order (L, R, L, R, L, R)
    boxes = [
        LineBox(text="L1", box_2d=[100, 50, 130, 400]),
        LineBox(text="R1", box_2d=[100, 550, 130, 900]),
        LineBox(text="L2", box_2d=[150, 50, 180, 400]),
        LineBox(text="R2", box_2d=[150, 550, 180, 900]),
        LineBox(text="L3", box_2d=[200, 50, 230, 400]),
        LineBox(text="R3", box_2d=[200, 550, 230, 900]),
    ]
    assert [b.text for b in sort_for_reading_order(boxes)] == [
        "L1", "L2", "L3", "R1", "R2", "R3",
    ]


def test_sort_for_reading_order_single_column_passes_through():
    boxes = [
        LineBox(text="A", box_2d=[100, 50, 130, 900]),
        LineBox(text="B", box_2d=[150, 50, 180, 900]),
        LineBox(text="C", box_2d=[200, 50, 230, 900]),
    ]
    assert [b.text for b in sort_for_reading_order(boxes)] == ["A", "B", "C"]


def test_sort_for_reading_order_handles_three_columns():
    # Detection returned them in row-major order across three columns.
    boxes = [
        LineBox(text="C1-1", box_2d=[100, 20, 130, 300]),
        LineBox(text="C2-1", box_2d=[100, 350, 130, 650]),
        LineBox(text="C3-1", box_2d=[100, 700, 130, 980]),
        LineBox(text="C1-2", box_2d=[200, 20, 230, 300]),
        LineBox(text="C2-2", box_2d=[200, 350, 230, 650]),
        LineBox(text="C3-2", box_2d=[200, 700, 230, 980]),
    ]
    assert [b.text for b in sort_for_reading_order(boxes)] == [
        "C1-1", "C1-2", "C2-1", "C2-2", "C3-1", "C3-2",
    ]


def test_alignment_unscrambles_multi_column_pages():
    # This reproduces the index-page bug. The transcription pass returned
    # column-major text (all left, then all right); the detection pass returned
    # boxes interleaved by row. A naive zip would put right-column text onto
    # left-column boxes and vice versa — the new column-major sort fixes it.
    transcription = ["Lα", "Lβ", "Lγ", "Rα", "Rβ", "Rγ"]
    detected = [
        LineBox(text="left α",  box_2d=[100, 50, 130, 400]),
        LineBox(text="right α", box_2d=[100, 550, 130, 900]),
        LineBox(text="left β",  box_2d=[150, 50, 180, 400]),
        LineBox(text="right β", box_2d=[150, 550, 180, 900]),
        LineBox(text="left γ",  box_2d=[200, 50, 230, 400]),
        LineBox(text="right γ", box_2d=[200, 550, 230, 900]),
    ]
    result = align_lines(transcription, detected)
    for line in result:
        if line.text.startswith("L"):
            assert line.box_2d[1] < 500, f"{line.text!r} ended up at {line.box_2d}"
        else:
            assert line.box_2d[1] >= 500, f"{line.text!r} ended up at {line.box_2d}"


def test_equal_counts_zip_with_accurate_text():
    transcription = ["The quick brown fox", "jumps over the dog"]
    detected = [_box("the qiuck bruwn fox", 100), _box("jumps ovr the dog", 200)]
    result = align_lines(transcription, detected)
    assert [r.text for r in result] == transcription
    # Boxes come from the detection pass.
    assert result[0].box_2d == detected[0].box_2d
    assert result[1].box_2d == detected[1].box_2d


def test_more_transcription_lines_than_boxes_keeps_all_text():
    transcription = ["line one", "line two", "line three"]
    detected = [_box("line one", 100), _box("line three", 300)]
    result = align_lines(transcription, detected)
    assert [r.text for r in result] == transcription
    # Every line ends up with some box.
    assert all(len(r.box_2d) == 4 for r in result)


def test_no_detected_boxes_synthesizes_layout():
    transcription = ["alpha", "beta", "gamma"]
    result = align_lines(transcription, [])
    assert [r.text for r in result] == transcription
    # Synthetic boxes are stacked top to bottom.
    ys = [r.box_2d[0] for r in result]
    assert ys == sorted(ys)


def test_no_transcription_returns_detected():
    detected = [_box("only detected", 100)]
    result = align_lines([], detected)
    assert [r.text for r in result] == ["only detected"]
