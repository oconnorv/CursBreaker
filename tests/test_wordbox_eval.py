"""Word-box localization eval harness — Step 0 of the word-localization plan.

Measures how accurately a word-localization *strategy* places per-word boxes, so
future improvements (smarter split, ink-projection segmentation, a real detector)
are a measured before/after rather than a guess. The strategy under test today is
the proportional ``split_line_into_words`` synthesis that feeds hOCR, ALTO, and
the searchable PDF.

Two metric families:
  * accuracy — per-word IoU vs ground truth. Ground truth is generated
    synthetically by laying words with a real proportional font at known
    positions (so each word's true horizontal extent is known), and an optional
    real labeled set (e.g. cursive) is loaded from
    ``tests/fixtures/wordbox_truth.json`` when present.
  * defects  — degenerate / overlapping / out-of-bounds boxes, computed from the
    output alone (the same classes the ALTO sanity-checker flags).

Vertical extent is held at the line box for both truth and output, so IoU
isolates *horizontal* word placement — the axis the synthesis (and the A-tier
cheap wins) actually control.

Run as a report:  python tests/test_wordbox_eval.py
Run as tests:      python -m pytest tests/test_wordbox_eval.py -s
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from cursbreaker.hocr import split_line_into_words
from cursbreaker.models import PixelBox

LABELED_PATH = Path(__file__).parent / "fixtures" / "wordbox_truth.json"
FONT_PATH = Path(__file__).resolve().parents[1] / "src" / "cursbreaker" / "fonts" / "DejaVuSans.ttf"

# Current measured mean IoU on the synthetic printed set is ~0.86. This floor
# guards against regressions while tolerating cross-version font-render jitter;
# A-tier improvements (A1 width-weighted split, A3 ink-projection) and a real
# cursive labeled set are where the headroom is.
BASELINE_MEAN_IOU = 0.80

# Representative lines (short/long words, numbers, names) echoing real records.
SAMPLES = [
    "Funeral of Jack Wright",
    "Date of Birth July 12 1908",
    "Cause of Death Pneumonia",
    "Place of Death Charlotte",
    "the quick brown fox jumps over",
    "Address 1925 Bently Avenue",
]


# ---- strategy under test ------------------------------------------------- #
Strategy = Callable[[str, PixelBox], "list[PixelBox]"]


def synthesis_strategy(text: str, box: PixelBox) -> list[PixelBox]:
    """The current default: proportional split (hocr.split_line_into_words)."""
    return [wb for _, wb in split_line_into_words(text, box)]


# ---- geometry ------------------------------------------------------------ #
def iou(a: PixelBox, b: PixelBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = a.width() * a.height() + b.width() * b.height() - inter
    return inter / union if union else 0.0


def _overlap_frac(a: PixelBox, b: PixelBox) -> float:
    inter = (max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
             * max(0, min(a.y1, b.y1) - max(a.y0, b.y0)))
    amin = min(a.width() * a.height(), b.width() * b.height())
    return inter / amin if amin else 0.0


# ---- datasets ------------------------------------------------------------ #
@dataclass
class EvalLine:
    text: str
    box: PixelBox            # the line box handed to the strategy
    truth: list[PixelBox]    # ground-truth word boxes, parallel to text.split()


def _font(size: int = 40):
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


def synthetic_lines() -> list[EvalLine]:
    """Lay each sample's words with a proportional font at known positions and
    record each word's true horizontal extent (vertical held to the line box)."""
    font = _font(40)
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    space = scratch.textlength(" ", font=font)
    lines: list[EvalLine] = []
    for text in SAMPLES:
        x = 20.0
        ink = []  # (left, top, right, bottom) per word, at its drawn position
        for w in text.split():
            left, top, right, bottom = scratch.textbbox((x, 10), w, font=font)
            ink.append((left, top, right, bottom))
            x += scratch.textlength(w, font=font) + space
        ly0 = int(min(t for _, t, _, _ in ink))
        ly1 = int(max(b for _, _, _, b in ink))
        line_box = PixelBox(
            x0=int(min(l for l, _, _, _ in ink)), y0=ly0,
            x1=int(max(r for _, _, r, _ in ink)), y1=ly1,
        )
        truth = [PixelBox(x0=int(l), y0=ly0, x1=int(r), y1=ly1) for (l, _, r, _) in ink]
        lines.append(EvalLine(text=text, box=line_box, truth=truth))
    return lines


def _box(seq) -> PixelBox:
    x0, y0, x1, y1 = seq
    return PixelBox(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))


def load_labeled_set(path: Path = LABELED_PATH) -> list[EvalLine]:
    """Optional real labeled set. JSON shape:

        {"pages": [{"lines": [
            {"text": "...", "box": [x0,y0,x1,y1], "words": [[x0,y0,x1,y1], ...]}
        ]}]}

    Returns [] when the file is absent, so the harness runs synthetic-only."""
    if not Path(path).is_file():
        return []
    data = json.loads(Path(path).read_text("utf-8"))
    out: list[EvalLine] = []
    for page in data.get("pages", []):
        for ln in page.get("lines", []):
            out.append(EvalLine(text=ln["text"], box=_box(ln["box"]),
                                truth=[_box(wb) for wb in ln["words"]]))
    return out


# ---- evaluation ---------------------------------------------------------- #
@dataclass
class EvalReport:
    name: str
    n_lines: int = 0
    n_words: int = 0
    mean_iou: float = 0.0
    pct_iou50: float = 0.0     # % of words with IoU >= 0.5
    degenerate: int = 0
    overlaps: int = 0
    out_of_bounds: int = 0


def count_defects(boxes, line_box, *, min_dim=2, overlap_thresh=0.5):
    degenerate = sum(1 for b in boxes if b.width() < min_dim or b.height() < min_dim)
    overlaps = sum(1 for a, b in zip(boxes, boxes[1:])
                   if _overlap_frac(a, b) > overlap_thresh)
    oob = sum(1 for b in boxes
              if b.x0 < line_box.x0 - 1 or b.y0 < line_box.y0 - 1
              or b.x1 > line_box.x1 + 1 or b.y1 > line_box.y1 + 1)
    return degenerate, overlaps, oob


def evaluate(name, lines, strategy: Strategy = synthesis_strategy,
             *, min_dim=2, overlap_thresh=0.5) -> EvalReport:
    rep = EvalReport(name=name)
    ious: list[float] = []
    for ln in lines:
        produced = strategy(ln.text, ln.box)
        n = min(len(produced), len(ln.truth))
        ious.extend(iou(produced[i], ln.truth[i]) for i in range(n))
        ious.extend([0.0] * abs(len(ln.truth) - len(produced)))  # unmatched -> 0
        d, o, b = count_defects(produced, ln.box, min_dim=min_dim,
                                overlap_thresh=overlap_thresh)
        rep.degenerate += d
        rep.overlaps += o
        rep.out_of_bounds += b
        rep.n_lines += 1
        rep.n_words += len(ln.truth)
    rep.mean_iou = sum(ious) / len(ious) if ious else 0.0
    rep.pct_iou50 = 100.0 * sum(1 for v in ious if v >= 0.5) / len(ious) if ious else 0.0
    return rep


def run_eval(strategy: Strategy = synthesis_strategy) -> list[EvalReport]:
    reports = [evaluate("synthetic (printed, auto-truth)", synthetic_lines(), strategy)]
    labeled = load_labeled_set()
    if labeled:
        reports.append(evaluate(f"labeled set ({LABELED_PATH.name})", labeled, strategy))
    return reports


def format_report(rep: EvalReport) -> str:
    return (f"  {rep.name}\n"
            f"    lines={rep.n_lines}  words={rep.n_words}\n"
            f"    mean word IoU = {rep.mean_iou:.3f}   words IoU>=0.5 = {rep.pct_iou50:.1f}%\n"
            f"    defects: degenerate={rep.degenerate}  "
            f"overlaps>50%={rep.overlaps}  out-of-bounds={rep.out_of_bounds}")


def main():
    print("Word-box localization eval — strategy: synthesis (proportional split)\n")
    for rep in run_eval():
        print(format_report(rep))
    if not LABELED_PATH.is_file():
        print(f"\n(No labeled set at {LABELED_PATH.relative_to(Path.cwd()) if LABELED_PATH.is_relative_to(Path.cwd()) else LABELED_PATH} "
              f"— drop real cursive pages there to measure handwriting word-IoU. "
              f"Synthetic-only for now.)")


# ---- tests --------------------------------------------------------------- #
def test_iou_basic():
    a = PixelBox(x0=0, y0=0, x1=10, y1=10)
    assert iou(a, a) == 1.0
    assert iou(a, PixelBox(x0=20, y0=0, x1=30, y1=10)) == 0.0
    half = PixelBox(x0=5, y0=0, x1=15, y1=10)  # inter=50, union=150 -> 1/3
    assert abs(iou(a, half) - (50 / 150)) < 1e-9


def test_defect_detection_flags_known_bad_boxes():
    line = PixelBox(x0=0, y0=0, x1=100, y1=20)
    boxes = [
        PixelBox(x0=0, y0=0, x1=1, y1=20),      # degenerate (width 1)
        PixelBox(x0=10, y0=0, x1=60, y1=20),
        PixelBox(x0=15, y0=0, x1=58, y1=20),    # ~fully inside the previous box
        PixelBox(x0=90, y0=0, x1=130, y1=20),   # out of bounds (x1 > 100)
    ]
    d, o, b = count_defects(boxes, line)
    assert d >= 1 and o >= 1 and b >= 1


def test_synthetic_eval_runs_and_meets_baseline():
    rep = evaluate("synthetic", synthetic_lines())
    print("\n" + format_report(rep))
    assert rep.n_words >= 20
    # Improvements should raise mean IoU; it must not regress below the floor.
    assert rep.mean_iou >= BASELINE_MEAN_IOU
    # The synthesis must not emit the defects the ALTO checker flags on clean input.
    assert rep.degenerate == 0
    assert rep.out_of_bounds == 0


def test_labeled_set_absent_returns_empty(tmp_path):
    assert load_labeled_set(tmp_path / "nope.json") == []


def test_labeled_set_parses_and_evaluates(tmp_path):
    p = tmp_path / "truth.json"
    p.write_text(json.dumps({"pages": [{"lines": [
        {"text": "hi there", "box": [0, 0, 100, 20],
         "words": [[0, 0, 40, 20], [50, 0, 100, 20]]},
    ]}]}))
    lines = load_labeled_set(p)
    assert len(lines) == 1 and lines[0].text == "hi there"
    assert len(lines[0].truth) == 2 and lines[0].truth[0].x1 == 40
    rep = evaluate("labeled", lines)
    assert rep.n_words == 2 and 0.0 <= rep.mean_iou <= 1.0


if __name__ == "__main__":
    main()
