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

from .models import LineBox

# Gap between consecutive sorted left-edge coordinates that we treat as a
# column boundary. Coordinates are normalized 0-1000, so 100 ≈ 10 % of the
# page width — large enough to ignore within-column wobble but small enough
# to catch a real column gutter.
_COLUMN_GAP_THRESHOLD = 100


def sort_for_reading_order(detected: list[LineBox]) -> list[LineBox]:
    """Return ``detected`` re-ordered column-major (top-to-bottom within each
    detected column, columns ordered left-to-right)."""
    if len(detected) <= 1:
        return list(detected)

    # Group boxes by left edge (xmin). Within-column lines share a left margin;
    # a real column break shows up as a large gap in the sorted sequence.
    by_xmin = sorted(range(len(detected)), key=lambda i: detected[i].box_2d[1])
    columns: list[list[int]] = [[by_xmin[0]]]
    last_xmin = detected[by_xmin[0]].box_2d[1]
    for idx in by_xmin[1:]:
        xmin = detected[idx].box_2d[1]
        if xmin - last_xmin > _COLUMN_GAP_THRESHOLD:
            columns.append([])
        columns[-1].append(idx)
        last_xmin = xmin

    result: list[LineBox] = []
    for column in columns:
        column.sort(key=lambda i: detected[i].box_2d[0])  # ymin: top to bottom
        result.extend(detected[i] for i in column)
    return result


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def align_lines(
    transcription_lines: list[str], detected: list[LineBox]
) -> list[LineBox]:
    """Return ``LineBox`` items carrying the accurate transcription text.

    The returned boxes are still in normalized 0-1000 space (taken from the
    detected boxes); only the ``text`` is replaced with the authoritative
    transcription.
    """
    transcription_lines = [t for t in transcription_lines if t.strip()]
    detected = sort_for_reading_order(detected)

    if not detected:
        # No spatial information at all: stack lines into evenly spaced,
        # full-width boxes so the page is still usable/searchable.
        return _synthetic_layout(transcription_lines)
    if not transcription_lines:
        return list(detected)

    norm_t = [_norm(t) for t in transcription_lines]
    norm_d = [_norm(d.text) for d in detected]

    matcher = SequenceMatcher(a=norm_t, b=norm_d, autojunk=False)
    result: list[LineBox] = []
    used_detected = [False] * len(detected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                box = detected[j1 + k]
                used_detected[j1 + k] = True
                result.append(LineBox(text=transcription_lines[i1 + k], box_2d=box.box_2d))
        elif tag == "replace":
            # Map the block of transcription lines onto the block of detected
            # boxes positionally; if counts differ, share boxes by ratio.
            t_block = list(range(i1, i2))
            d_block = list(range(j1, j2))
            for idx, ti in enumerate(t_block):
                if d_block:
                    dj = d_block[min(idx, len(d_block) - 1)]
                    used_detected[dj] = True
                    result.append(
                        LineBox(text=transcription_lines[ti], box_2d=detected[dj].box_2d)
                    )
                else:
                    result.append(
                        LineBox(text=transcription_lines[ti], box_2d=_interp_box(result, detected))
                    )
        elif tag == "delete":
            # Transcription lines with no detected counterpart: place them with
            # an interpolated box derived from neighbors.
            for ti in range(i1, i2):
                result.append(
                    LineBox(text=transcription_lines[ti], box_2d=_interp_box(result, detected))
                )
        elif tag == "insert":
            # Detected boxes with no transcription line: keep their own text so
            # the region is still represented.
            for dj in range(j1, j2):
                if not used_detected[dj]:
                    used_detected[dj] = True
                    result.append(LineBox(text=detected[dj].text, box_2d=detected[dj].box_2d))

    return result


def _interp_box(placed: list[LineBox], detected: list[LineBox]) -> list[int]:
    """Best-effort box for an unmatched transcription line: just below the last
    placed box, or a default full-width strip near the top."""
    if placed:
        prev = placed[-1].box_2d
        height = max(20, prev[2] - prev[0])
        ymin = min(990, prev[2] + 5)
        ymax = min(1000, ymin + height)
        return [ymin, prev[1], ymax, prev[3]]
    if detected:
        first = detected[0].box_2d
        return list(first)
    return [0, 0, 40, 1000]


def _synthetic_layout(lines: list[str]) -> list[LineBox]:
    if not lines:
        return []
    n = len(lines)
    margin = 40
    usable = 1000 - 2 * margin
    step = usable / n
    out: list[LineBox] = []
    for i, text in enumerate(lines):
        ymin = int(margin + i * step)
        ymax = int(margin + (i + 1) * step) - 4
        out.append(LineBox(text=text, box_2d=[ymin, margin, ymax, 1000 - margin]))
    return out
