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


def test_dedupes_paired_wide_and_narrow_detections_for_one_line():
    # Mirrors the right-column issue on the user's index page: Gemini returned
    # both a wide and a narrow box for the same physical line at nearly the
    # same y. The narrow sliver should be dropped so it doesn't steal a slot
    # from a later transcription line.
    boxes = [
        LineBox(text="MOSES",  box_2d=[66, 565, 86, 765]),  # wide
        LineBox(text="x",      box_2d=[72, 508, 86, 540]),  # narrow, same y
        LineBox(text="MOORE",  box_2d=[91, 562, 110, 887]),  # wide
        LineBox(text="x",      box_2d=[95, 505, 108, 549]),  # narrow, same y
    ]
    out = sort_for_reading_order(boxes)
    assert [b.text for b in out] == ["MOSES", "MOORE"]


def test_keeps_tightly_spaced_real_lines():
    # Realistic line spacing where the gap between lines is smaller than the
    # line height itself — but the per-line y overlap stays low enough that
    # nothing is mistakenly deduped.
    boxes = [
        LineBox(text="A", box_2d=[100, 100, 120, 700]),
        LineBox(text="B", box_2d=[125, 100, 145, 700]),
        LineBox(text="C", box_2d=[150, 100, 170, 700]),
    ]
    out = sort_for_reading_order(boxes)
    assert [b.text for b in out] == ["A", "B", "C"]


def test_clamps_anomalously_tall_box_to_column_median_height():
    # 3 normal-height lines + 1 line where the detected box swallowed extra
    # whitespace and ended up 2.5x as tall. The tall one should be clamped to
    # the column's median height so it doesn't overlap the next line.
    boxes = [
        LineBox(text="a",    box_2d=[100, 50, 120, 800]),  # h=20
        LineBox(text="b",    box_2d=[130, 50, 150, 800]),  # h=20
        LineBox(text="c",    box_2d=[160, 50, 180, 800]),  # h=20
        LineBox(text="tall", box_2d=[190, 50, 240, 800]),  # h=50
    ]
    out = sort_for_reading_order(boxes)
    tall = next(b for b in out if b.text == "tall")
    assert tall.box_2d[2] - tall.box_2d[0] == 20  # clamped to median


def test_does_not_clamp_modestly_taller_boxes():
    # Within 1.3 * median, no clamp — variation in handwriting line height is
    # normal and not something to flatten.
    boxes = [
        LineBox(text="a", box_2d=[100, 50, 120, 800]),  # h=20
        LineBox(text="b", box_2d=[130, 50, 152, 800]),  # h=22
        LineBox(text="c", box_2d=[170, 50, 195, 800]),  # h=25 (1.13 * median)
    ]
    out = sort_for_reading_order(boxes)
    assert sorted(b.box_2d[2] - b.box_2d[0] for b in out) == [20, 22, 25]


def test_normalizes_a_lone_narrow_box_to_column_width():
    # Most of the column is around x 100-700; one box only covers x 100-150.
    # That partial detection should be expanded toward the column's typical
    # right edge so word-box synthesis lands on real text positions.
    boxes = [
        LineBox(text="a", box_2d=[100, 100, 120, 700]),
        LineBox(text="b", box_2d=[140, 100, 160, 700]),
        LineBox(text="c", box_2d=[180, 100, 200, 700]),
        LineBox(text="d", box_2d=[220, 100, 240, 150]),  # narrow partial
    ]
    out = sort_for_reading_order(boxes)
    narrow_now = next(b for b in out if b.text == "d")
    assert narrow_now.box_2d[3] >= 600


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


def test_extras_extend_column_at_typical_line_cadence():
    # 9 transcription lines, only the first 5 detected (matches the right-
    # column situation on the index page). The 4 unmatched extras should
    # advance at the typical line spacing observed in the detected boxes
    # rather than stacking tight on top of the last match.
    detected = [
        LineBox(text="row 1", box_2d=[66,  565,  86, 765]),
        LineBox(text="row 2", box_2d=[91,  565, 110, 765]),
        LineBox(text="row 3", box_2d=[114, 565, 134, 765]),
        LineBox(text="row 4", box_2d=[134, 565, 154, 765]),
        LineBox(text="row 5", box_2d=[155, 565, 175, 765]),
    ]
    transcription = [f"line {i}" for i in range(1, 10)]
    result = align_lines(transcription, detected)
    assert len(result) == 9
    # First 5 land on detected positions
    for i in range(5):
        assert result[i].box_2d == detected[i].box_2d
    # The remaining 4 should advance at roughly the same ~22 cadence per step.
    ymins = [r.box_2d[0] for r in result[4:]]
    gaps = [b - a for a, b in zip(ymins, ymins[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2]
    assert 18 <= median_gap <= 30, f"extras did not extend at column cadence: {gaps}"
