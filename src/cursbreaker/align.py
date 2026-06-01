"""Align two-pass output.

Pass 1 gives the most accurate transcription (a list of text lines, in reading
order). Pass 2 gives line bounding boxes (also in reading order), each with a
rough transcription of its own. We want the *accurate* text placed on the
*detected* boxes.

Both lists are in natural reading order, so we align them order-preservingly
with ``difflib`` over normalized line text. Transcription lines that don't match
a box are attached to a geometrically interpolated box so no text is dropped.

Before alignment we re-sort the detected boxes into column-major reading order.
Gemini's transcription pass naturally walks each column top-to-bottom and then
moves right; the detection pass sometimes returns boxes interleaved by physical
row instead, which scrambles the alignment on multi-column pages (e.g. indexes,
ledgers, news pages). Re-sorting by column-major reading order keeps the two
sequences in the same order so the matcher actually finds them.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import LineBox, OcrWord, PixelBox, PlacedLine

# Gap between consecutive sorted left-edge coordinates that we treat as a
# column boundary. Coordinates are normalized 0-1000, so 100 ≈ 10 % of the
# page width — large enough to ignore within-column wobble but small enough
# to catch a real column gutter.
_COLUMN_GAP_THRESHOLD = 100
# Two boxes whose vertical overlap exceeds this fraction of the shorter box's
# height are treated as the same physical line. Gemini sometimes returns one
# wide and one narrow detection for the same line on dense pages, and the
# narrow one would otherwise burn a slot in the alignment.
_DUPLICATE_Y_OVERLAP = 0.5
# Boxes narrower than this fraction of the column's median width are treated
# as partial detections and expanded to the column's typical x-range.
_NARROW_WIDTH_RATIO = 0.5
# Boxes taller than this multiple of the column's median height are clamped
# back to the median height. Gemini occasionally returns a box that spans
# extra whitespace below the line, which inflates the synthesized word boxes
# and can visibly overlap the next line on the overlay.
_TALL_HEIGHT_RATIO = 1.3


def _identify_columns(detected: list[LineBox]) -> list[list[LineBox]]:
    """Group boxes into columns via gap-based clustering on the left edge."""
    if not detected:
        return []
    by_xmin = sorted(detected, key=lambda b: b.box_2d[1])
    columns: list[list[LineBox]] = [[by_xmin[0]]]
    last_xmin = by_xmin[0].box_2d[1]
    for b in by_xmin[1:]:
        xmin = b.box_2d[1]
        if xmin - last_xmin > _COLUMN_GAP_THRESHOLD:
            columns.append([])
        columns[-1].append(b)
        last_xmin = xmin
    return columns


def _dedupe_overlapping_y(column: list[LineBox]) -> list[LineBox]:
    """Remove duplicate detections of the same physical line within a column.

    Sometimes Gemini returns two boxes for one line — a wide one covering the
    whole line and a narrow sliver covering a fragment — at nearly the same y.
    If both reach alignment, every text line after them shifts by a position.
    Keep the larger-area box; drop the rest."""
    if len(column) <= 1:
        return list(column)
    by_area = sorted(
        column,
        key=lambda b: -((b.box_2d[3] - b.box_2d[1]) * (b.box_2d[2] - b.box_2d[0])),
    )
    kept: list[LineBox] = []
    for b in by_area:
        y0, y1 = b.box_2d[0], b.box_2d[2]
        h = y1 - y0
        if h <= 0:
            continue
        duplicate = False
        for k in kept:
            ky0, ky1 = k.box_2d[0], k.box_2d[2]
            overlap = min(y1, ky1) - max(y0, ky0)
            if overlap > 0:
                shortest = min(h, ky1 - ky0)
                if shortest > 0 and overlap / shortest > _DUPLICATE_Y_OVERLAP:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(b)
    return kept


def _normalize_column_x(column: list[LineBox]) -> list[LineBox]:
    """Expand boxes that are much narrower than the column's typical width to
    span the column's median x-range — Gemini's pass occasionally returns a
    partial-line detection covering only one word."""
    if len(column) < 3:
        return list(column)
    widths = sorted(b.box_2d[3] - b.box_2d[1] for b in column)
    median_width = widths[len(widths) // 2]
    if median_width <= 0:
        return list(column)
    xmins = sorted(b.box_2d[1] for b in column)
    xmaxs = sorted(b.box_2d[3] for b in column)
    col_xmin = xmins[len(xmins) // 2]
    col_xmax = xmaxs[len(xmaxs) // 2]
    if col_xmax <= col_xmin:
        return list(column)
    out: list[LineBox] = []
    for b in column:
        w = b.box_2d[3] - b.box_2d[1]
        if w < median_width * _NARROW_WIDTH_RATIO:
            out.append(
                LineBox(
                    text=b.text,
                    box_2d=[b.box_2d[0], col_xmin, b.box_2d[2], col_xmax],
                )
            )
        else:
            out.append(b)
    return out


def _normalize_column_heights(column: list[LineBox]) -> list[LineBox]:
    """Clamp anomalously tall line boxes back to the column's median height."""
    if len(column) < 3:
        return list(column)
    heights = sorted(b.box_2d[2] - b.box_2d[0] for b in column)
    median_height = heights[len(heights) // 2]
    if median_height <= 0:
        return list(column)
    threshold = median_height * _TALL_HEIGHT_RATIO
    out: list[LineBox] = []
    for b in column:
        y0, x0, y1, x1 = b.box_2d
        if (y1 - y0) > threshold:
            out.append(
                LineBox(text=b.text, box_2d=[y0, x0, y0 + median_height, x1])
            )
        else:
            out.append(b)
    return out


def sort_for_reading_order(detected: list[LineBox]) -> list[LineBox]:
    """Return ``detected`` re-ordered column-major.

    Detection sometimes yields paired wide+narrow boxes for the same physical
    line, partial slivers covering one word, or boxes that swallow extra
    whitespace below the line. We deduplicate by y-overlap, snap surviving
    narrow boxes to the column's median x-range, clamp anomalously tall boxes
    to the median height, then sort each column top-to-bottom and concatenate
    left-to-right.
    """
    if len(detected) <= 1:
        return list(detected)
    result: list[LineBox] = []
    for column in _identify_columns(detected):
        column = _dedupe_overlapping_y(column)
        column = _normalize_column_x(column)
        column = _normalize_column_heights(column)
        column.sort(key=lambda b: b.box_2d[0])  # ymin: top to bottom
        result.extend(column)
    return result


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


_WORD_EDGE = re.compile(r"^\W+|\W+$", re.UNICODE)


def _norm_word(s: str) -> str:
    """Normalize a single token for matching: drop surrounding punctuation and
    lowercase, so 'Death,' and 'death' compare equal."""
    return _WORD_EDGE.sub("", s).lower()


def align_words(
    line_text: str,
    ocr_words: list[OcrWord],
    line_box: PixelBox,
    *,
    matched_conf: int,
    fallback_conf: int,
) -> list[OcrWord]:
    """Place each word of ``line_text`` on a real OCR box where the two agree.

    ``line_text`` is the authoritative (Gemini) transcription of one line;
    ``ocr_words`` are Tesseract's per-word detections for that line, in page
    pixels. Every word of ``line_text`` is returned, in order, with its text
    unchanged -- the OCR engine never edits the transcription. A real OCR box is
    adopted only for words whose text *matches* an OCR word (so a misread can't
    drag a word to the wrong place); the rest get a box interpolated between
    their matched neighbours, or split proportionally across ``line_box`` when
    there are none. ``matched_conf`` is written for adopted boxes,
    ``fallback_conf`` for synthesized ones.
    """
    gold = line_text.split()
    if not gold:
        return []
    assigned: list[OcrWord | None] = [None] * len(gold)
    if ocr_words:
        norm_g = [_norm_word(w) for w in gold]
        norm_o = [_norm_word(w.text) for w in ocr_words]
        matcher = SequenceMatcher(a=norm_g, b=norm_o, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                # Only adopt geometry where the text agrees; replace/delete/
                # insert all fall through to interpolation below.
                continue
            for k in range(i2 - i1):
                assigned[i1 + k] = OcrWord(
                    text=gold[i1 + k],
                    box=ocr_words[j1 + k].box,
                    confidence=matched_conf,
                )
    return _fill_word_gaps(gold, assigned, line_box, fallback_conf)


def _fill_word_gaps(
    gold: list[str],
    assigned: list[OcrWord | None],
    line_box: PixelBox,
    fallback_conf: int,
) -> list[OcrWord]:
    """Give every still-unmatched word a synthesized box in the horizontal gap
    between its nearest matched neighbours (or the line edges), distributed by
    word length. Vertical extent is borrowed from a neighbour, else the line."""
    n = len(gold)
    out: list[OcrWord] = [None] * n  # type: ignore[list-item]
    i = 0
    while i < n:
        if assigned[i] is not None:
            out[i] = assigned[i]  # type: ignore[assignment]
            i += 1
            continue
        j = i
        while j < n and assigned[j] is None:
            j += 1
        left = assigned[i - 1].box if i > 0 else None
        right = assigned[j].box if j < n else None
        x_left = left.x1 if left else line_box.x0
        x_right = right.x0 if right else line_box.x1
        if x_right <= x_left:
            x_right = x_left + 1
        ref = left or right
        y0 = ref.y0 if ref else line_box.y0
        y1 = ref.y1 if ref else line_box.y1
        run = gold[i:j]
        total = max(1, sum(len(w) for w in run) + max(0, len(run) - 1))
        span = x_right - x_left
        cursor = 0
        for k, word in enumerate(run):
            start, end = cursor, cursor + len(word)
            wx0 = x_left + round(start / total * span)
            wx1 = x_left + round(end / total * span)
            if wx1 <= wx0:
                wx1 = wx0 + 1
            out[i + k] = OcrWord(
                text=word,
                box=PixelBox(x0=wx0, y0=y0, x1=wx1, y1=y1),
                confidence=fallback_conf,
            )
            cursor = end + 1
        i = j
    return out


def align_lines(
    transcription_lines: list[str], detected: list[LineBox]
) -> list[PlacedLine]:
    """Return ``PlacedLine`` items carrying the accurate transcription text.

    The returned boxes are still in normalized 0-1000 space (taken from the
    detected boxes); only the ``text`` is replaced with the authoritative
    transcription. ``is_interpolated`` is ``True`` for lines whose box had to
    be estimated (no matching detection) so downstream consumers can flag
    them with a reduced word confidence.
    """
    transcription_lines = [t for t in transcription_lines if t.strip()]
    detected = sort_for_reading_order(detected)

    if not detected:
        # No spatial information at all: stack lines into evenly spaced,
        # full-width boxes so the page is still usable/searchable.
        return _synthetic_layout(transcription_lines)
    if not transcription_lines:
        return [
            PlacedLine(text=d.text, box_2d=list(d.box_2d), is_interpolated=False)
            for d in detected
        ]

    norm_t = [_norm(t) for t in transcription_lines]
    norm_d = [_norm(d.text) for d in detected]

    matcher = SequenceMatcher(a=norm_t, b=norm_d, autojunk=False)
    result: list[PlacedLine] = []
    used_detected = [False] * len(detected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                box = detected[j1 + k]
                used_detected[j1 + k] = True
                result.append(
                    PlacedLine(
                        text=transcription_lines[i1 + k],
                        box_2d=list(box.box_2d),
                        is_interpolated=False,
                    )
                )
        elif tag == "replace":
            # Pair the transcription block onto the detected block positionally;
            # any extras beyond the detected block fall through to interpolation
            # rather than piling up on the last detected box.
            t_block = list(range(i1, i2))
            d_block = list(range(j1, j2))
            for idx, ti in enumerate(t_block):
                if idx < len(d_block):
                    dj = d_block[idx]
                    used_detected[dj] = True
                    result.append(
                        PlacedLine(
                            text=transcription_lines[ti],
                            box_2d=list(detected[dj].box_2d),
                            is_interpolated=False,
                        )
                    )
                else:
                    result.append(
                        PlacedLine(
                            text=transcription_lines[ti],
                            box_2d=_interp_box(result, detected),
                            is_interpolated=True,
                        )
                    )
        elif tag == "delete":
            # Transcription lines with no detected counterpart: place them with
            # an interpolated box derived from neighbors.
            for ti in range(i1, i2):
                result.append(
                    PlacedLine(
                        text=transcription_lines[ti],
                        box_2d=_interp_box(result, detected),
                        is_interpolated=True,
                    )
                )
        elif tag == "insert":
            # Detected boxes with no transcription line: keep their own text so
            # the region is still represented.
            for dj in range(j1, j2):
                if not used_detected[dj]:
                    used_detected[dj] = True
                    result.append(
                        PlacedLine(
                            text=detected[dj].text,
                            box_2d=list(detected[dj].box_2d),
                            is_interpolated=False,
                        )
                    )

    return result


def _median_step(boxes, fallback: float) -> float:
    """Median y-step (``ymin_{i+1} - ymin_i``) between consecutive boxes."""
    if len(boxes) < 2:
        return fallback
    steps = [
        b.box_2d[0] - a.box_2d[0]
        for a, b in zip(boxes, boxes[1:])
        if b.box_2d[0] > a.box_2d[0]
    ]
    if not steps:
        return fallback
    return sorted(steps)[len(steps) // 2]


def _interp_box(placed, detected) -> list[int]:
    """Best-effort box for an unmatched transcription line: positioned below
    the most recent placement at the column's typical line cadence so that a
    run of extras extends the column at the same rhythm as the detected lines
    instead of stacking tightly on top of each other.

    Accepts either ``LineBox`` or ``PlacedLine`` entries — both expose
    ``box_2d``.
    """
    if placed:
        prev = placed[-1].box_2d
        height = max(20, prev[2] - prev[0])
        step = max(int(_median_step(placed, height + 5)), height + 1)
        ymin = min(990, prev[0] + step)
        ymax = min(1000, ymin + height)
        return [ymin, prev[1], ymax, prev[3]]
    if detected:
        first = detected[0].box_2d
        return list(first)
    return [0, 0, 40, 1000]


def _synthetic_layout(lines: list[str]) -> list[PlacedLine]:
    if not lines:
        return []
    n = len(lines)
    margin = 40
    usable = 1000 - 2 * margin
    step = usable / n
    out: list[PlacedLine] = []
    for i, text in enumerate(lines):
        ymin = int(margin + i * step)
        ymax = int(margin + (i + 1) * step) - 4
        out.append(
            PlacedLine(
                text=text,
                box_2d=[ymin, margin, ymax, 1000 - margin],
                is_interpolated=True,
            )
        )
    return out
