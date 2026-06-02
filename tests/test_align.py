from cursbreaker.align import align_lines, align_words, sort_for_reading_order
from cursbreaker.models import LineBox, OcrWord, PixelBox


def _box(text, ymin):
    return LineBox(text=text, box_2d=[ymin, 50, ymin + 30, 950])


def _word(text, x0, x1, y0=10, y1=40):
    return OcrWord(text=text, box=PixelBox(x0=x0, y0=y0, x1=x1, y1=y1))


_LINE = PixelBox(x0=0, y0=10, x1=300, y1=40)


def test_align_words_adopts_boxes_where_text_matches():
    # Every gold word matches a Tesseract word -> all adopt the real boxes and
    # the matched confidence, and the text stays the gold text.
    ocr = [_word("Cause", 10, 60), _word("of", 70, 95), _word("Death", 100, 180)]
    out = align_words(
        "Cause of Death", ocr, _LINE, matched_conf=95, fallback_conf=60
    )
    assert [w.text for w in out] == ["Cause", "of", "Death"]
    assert [(w.box.x0, w.box.x1) for w in out] == [(10, 60), (70, 95), (100, 180)]
    assert all(w.confidence == 95 for w in out)


def test_align_words_keeps_gemini_word_when_tesseract_misreads():
    # Tesseract misreads the last word; we keep Gemini's spelling, do NOT adopt
    # the mismatched box (synthesize instead), and flag it with fallback conf.
    ocr = [_word("Cause", 10, 60), _word("of", 70, 95), _word("Prummonia", 100, 250)]
    out = align_words(
        "Cause of Pneumonia", ocr, _LINE, matched_conf=95, fallback_conf=60
    )
    assert [w.text for w in out] == ["Cause", "of", "Pneumonia"]
    assert out[0].confidence == 95 and out[1].confidence == 95
    assert out[2].confidence == 60  # synthesized, not Tesseract's box
    # The synthesized word sits after its matched neighbour and within the line.
    assert out[2].box.x0 >= out[1].box.x1
    assert out[2].box.x1 <= _LINE.x1


def test_align_words_preserves_every_gold_word_and_order():
    # Counts differ and an extra gold word has no match: nothing is dropped.
    ocr = [_word("the", 10, 40), _word("quick", 50, 120)]
    out = align_words(
        "the quick brown fox", ocr, _LINE, matched_conf=95, fallback_conf=60
    )
    assert [w.text for w in out] == ["the", "quick", "brown", "fox"]


def test_align_words_ignores_extra_tesseract_words():
    # Tesseract hallucinates a trailing word; it must not appear in the output.
    ocr = [_word("hello", 10, 90), _word("world", 100, 190), _word("XX", 200, 230)]
    out = align_words("hello world", ocr, _LINE, matched_conf=95, fallback_conf=60)
    assert [w.text for w in out] == ["hello", "world"]


def test_align_words_no_ocr_synthesizes_all_within_line():
    out = align_words("alpha beta", [], _LINE, matched_conf=95, fallback_conf=60)
    assert [w.text for w in out] == ["alpha", "beta"]
    assert all(w.confidence == 60 for w in out)
    assert all(_LINE.x0 <= w.box.x0 < w.box.x1 <= _LINE.x1 for w in out)


def test_align_words_matches_despite_punctuation_and_case():
    ocr = [_word("DEATH,", 10, 90)]
    out = align_words("Death", ocr, _LINE, matched_conf=95, fallback_conf=60)
    assert out[0].text == "Death"  # gold spelling/case kept
    assert out[0].confidence == 95  # but matched on normalized form
    assert (out[0].box.x0, out[0].box.x1) == (10, 90)


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
    # Matched lines should not be flagged as interpolated.
    assert all(not r.is_interpolated for r in result)


def test_interpolated_lines_carry_provenance_flag():
    # 4 transcription lines but only 2 detected. The trailing 2 are
    # interpolated and should be flagged so the hOCR builder can lower their
    # word confidence.
    transcription = ["a", "b", "c", "d"]
    detected = [_box("a", 100), _box("b", 200)]
    result = align_lines(transcription, detected)
    assert [r.text for r in result] == transcription
    assert [r.is_interpolated for r in result] == [False, False, True, True]


def test_synthetic_layout_marks_everything_interpolated():
    # No detected boxes at all -> the synthesized stack is entirely guessed.
    result = align_lines(["x", "y", "z"], [])
    assert [r.text for r in result] == ["x", "y", "z"]
    assert all(r.is_interpolated for r in result)


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
