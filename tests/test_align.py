from cursebreaker.align import align_lines
from cursebreaker.models import LineBox


def _box(text, ymin):
    return LineBox(text=text, box_2d=[ymin, 50, ymin + 30, 950])


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
