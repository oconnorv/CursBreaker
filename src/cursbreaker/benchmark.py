"""Transcription-accuracy benchmark — Track A of the evaluation-suite plan.

Scores CursBreaker's transcription against a human ground-truth corpus, reported
by document stratum (type x difficulty x era), so prompts/models/settings are
tuned with evidence and regressions are caught.

Two scoring modes (the manifest's ``mode`` column):
  * ``sequential`` — CER + WER (standard word/character edit distance). For linear
    text: letters, postcards, newsletters, and **single newspaper columns** (crop
    a column into its own ``sequential`` doc rather than scoring a whole page).
  * ``spotting``   — order-free label precision/recall/F1. For maps, whose
    scattered, multi-orientation labels have no reading order.

CER/WER here are the standard edit-distance metrics (the same algorithm ``jiwer``
implements); no third-party dependency is required. Normalization is applied
identically to truth and hypothesis (see ``normalize``) -- CER/WER are meaningless
without a fixed, documented policy.

Corpus layout (default ``--corpus benchmark/corpus``):
    manifest.csv          id,image,type,subtype,era,difficulty,mode,source
    gt/<id>.txt           human ground-truth transcription
    images/<id>.<ext>     page image (only needed for --refresh)
    cache/<id>.<tag>.txt  cached model hypothesis (so scoring runs offline + free)

Usage:
    python -m cursbreaker.benchmark               # score cached hyps, print report
    python -m cursbreaker.benchmark --refresh     # run the model first (API cost), cache, then score
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_CORPUS = Path("benchmark/corpus")

# Bracketed transcriber markers ([illegible], [?], [blank], ...) survive whole.
_SPECIAL = re.compile(r"^\[.*\]$")
_PUNCT_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)


# --------------------------------------------------------------------------- #
# Normalization (applied IDENTICALLY to truth and hypothesis)
# --------------------------------------------------------------------------- #
def normalize(text: str, *, strict: bool = False) -> str:
    """Fixed normalization policy. NFC; long-s -> s; case-fold; collapse
    whitespace. Non-strict also strips punctuation at token edges (keeping
    intra-word apostrophes/hyphens); strict keeps all punctuation. Bracketed
    markers like ``[illegible]`` are preserved as single tokens."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ſ", "s")          # long s
    text = text.casefold()
    out: list[str] = []
    for tok in text.split():
        if _SPECIAL.match(tok):
            out.append(tok)
            continue
        if not strict:
            tok = _PUNCT_EDGES.sub("", tok)
        if tok:
            out.append(tok)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Edit distance (Wagner-Fischer) with S/D/I breakdown
# --------------------------------------------------------------------------- #
def _levenshtein(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Min edit distance between two token sequences + (subs, dels, ins).
    Deletions = reference tokens missing from the hypothesis; insertions = extra
    hypothesis tokens (the standard ASR/OCR convention)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # Backtrace for the substitution/deletion/insertion split.
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        cost = (0 if (i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]) else 1)
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost:
                S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return dp[n][m], S, D, I


@dataclass
class SeqScore:
    n_ref_words: int
    word_errors: int
    sub: int
    dele: int
    ins: int
    n_ref_chars: int
    char_errors: int

    @property
    def wer(self) -> float:
        return self.word_errors / self.n_ref_words if self.n_ref_words else 0.0

    @property
    def cer(self) -> float:
        return self.char_errors / self.n_ref_chars if self.n_ref_chars else 0.0


def score_sequential(ref_raw: str, hyp_raw: str, *, strict: bool = False) -> SeqScore:
    ref, hyp = normalize(ref_raw, strict=strict), normalize(hyp_raw, strict=strict)
    rw, hw = ref.split(), hyp.split()
    wdist, S, D, I = _levenshtein(rw, hw)
    cdist, *_ = _levenshtein(list(ref), list(hyp))
    return SeqScore(len(rw), wdist, S, D, I, len(ref), cdist)


# --------------------------------------------------------------------------- #
# Spotting (order-free, for maps)
# --------------------------------------------------------------------------- #
@dataclass
class SpotScore:
    n_ref: int
    n_hyp: int
    matched: int

    @property
    def precision(self) -> float:
        return self.matched / self.n_hyp if self.n_hyp else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.n_ref if self.n_ref else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_spotting(ref_raw: str, hyp_raw: str, *, strict: bool = False,
                   threshold: float = 1.0) -> SpotScore:
    """Order-free label match. Default ``threshold=1.0`` = exact match after
    normalization (a misread label counts against precision AND recall); a lower
    threshold allows fuzzy matches via difflib ratio."""
    ref = normalize(ref_raw, strict=strict).split()
    hyp = normalize(hyp_raw, strict=strict).split()
    matched = 0
    if threshold >= 1.0:
        remaining = Counter(ref)
        for t in hyp:
            if remaining.get(t, 0) > 0:
                remaining[t] -= 1
                matched += 1
    else:
        used = [False] * len(ref)
        for t in hyp:
            best, bi = 0.0, -1
            for k, rt in enumerate(ref):
                if used[k]:
                    continue
                ratio = SequenceMatcher(None, t, rt).ratio()
                if ratio > best:
                    best, bi = ratio, k
            if bi >= 0 and best >= threshold:
                used[bi] = True
                matched += 1
    return SpotScore(len(ref), len(hyp), matched)


# --------------------------------------------------------------------------- #
# Corpus I/O + scoring
# --------------------------------------------------------------------------- #
MANIFEST_COLUMNS = ["id", "image", "type", "subtype", "era", "difficulty", "mode", "source"]
CEILING = {"ceiling", "extreme"}


def load_manifest(corpus: Path) -> list[dict]:
    path = Path(corpus) / "manifest.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if (row.get("id") or "").strip()]


def _read(path: Path) -> str | None:
    return path.read_text("utf-8") if path.is_file() else None


def cache_path(corpus: Path, doc_id: str, tag: str) -> Path:
    return Path(corpus) / "cache" / f"{doc_id}.{tag}.txt"


@dataclass
class DocResult:
    id: str
    mode: str
    type: str = ""
    difficulty: str = ""
    era: str = ""
    seq: SeqScore | None = None
    spot: SpotScore | None = None
    missing: str | None = None


def score_corpus(corpus: Path, *, model_tag: str = "default", strict: bool = False,
                 spotting_threshold: float = 1.0) -> list[DocResult]:
    results: list[DocResult] = []
    for row in load_manifest(corpus):
        doc_id = row["id"].strip()
        mode = (row.get("mode") or "sequential").strip()
        dr = DocResult(id=doc_id, mode=mode, type=(row.get("type") or "").strip(),
                       difficulty=(row.get("difficulty") or "").strip(),
                       era=(row.get("era") or "").strip())
        gt = _read(Path(corpus) / "gt" / f"{doc_id}.txt")
        hyp = _read(cache_path(corpus, doc_id, model_tag))
        if gt is None:
            dr.missing = "no ground truth"
        elif hyp is None:
            dr.missing = f"no cached hypothesis for tag '{model_tag}' (run --refresh)"
        elif mode == "spotting":
            dr.spot = score_spotting(gt, hyp, strict=strict, threshold=spotting_threshold)
        else:
            dr.seq = score_sequential(gt, hyp, strict=strict)
        results.append(dr)
    return results


def aggregate(results: list[DocResult]):
    """Micro-average (sum errors / sum units) by stratum and overall; ceiling
    difficulties are reported separately, never in the headline."""
    def acc():
        return {"wer_num": 0, "wer_den": 0, "cer_num": 0, "cer_den": 0, "docs": 0}

    overall, strata = acc(), {}
    spot = {"ref": 0, "hyp": 0, "matched": 0, "docs": 0}
    ceiling: list[DocResult] = []
    for r in results:
        if r.missing:
            continue
        if r.difficulty in CEILING:
            ceiling.append(r)
            continue
        if r.seq:
            s = strata.setdefault((r.type, r.difficulty, r.era), acc())
            for tgt in (overall, s):
                tgt["wer_num"] += r.seq.word_errors
                tgt["wer_den"] += r.seq.n_ref_words
                tgt["cer_num"] += r.seq.char_errors
                tgt["cer_den"] += r.seq.n_ref_chars
                tgt["docs"] += 1
        if r.spot:
            spot["ref"] += r.spot.n_ref
            spot["hyp"] += r.spot.n_hyp
            spot["matched"] += r.spot.matched
            spot["docs"] += 1
    return overall, strata, spot, ceiling


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def format_report(results: list[DocResult]) -> str:
    overall, strata, spot, ceiling = aggregate(results)
    scored = [r for r in results if not r.missing]
    skipped = [r for r in results if r.missing]
    lines = ["Transcription accuracy benchmark",
             f"  scored {len(scored)} doc(s); skipped {len(skipped)}"]
    if overall["wer_den"]:
        lines.append(
            f"  OVERALL (sequential, excl. ceiling): "
            f"WER {_rate(overall['wer_num'], overall['wer_den'])*100:.1f}%  "
            f"CER {_rate(overall['cer_num'], overall['cer_den'])*100:.1f}%  "
            f"({overall['docs']} docs, {overall['wer_den']} ref words)")
    for key in sorted(strata):
        s = strata[key]
        lines.append(
            f"    {key[0]}/{key[1]}/{key[2]}: "
            f"WER {_rate(s['wer_num'], s['wer_den'])*100:.1f}%  "
            f"CER {_rate(s['cer_num'], s['cer_den'])*100:.1f}%  ({s['docs']} docs)")
    if spot["docs"]:
        p, r = _rate(spot["matched"], spot["hyp"]), _rate(spot["matched"], spot["ref"])
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        lines.append(f"  MAPS (spotting): P {p*100:.1f}%  R {r*100:.1f}%  "
                     f"F1 {f1*100:.1f}%  ({spot['docs']} docs)")
    if ceiling:
        lines.append(f"  CEILING (reported, excluded from headline): {len(ceiling)} doc(s)")
        for r in ceiling:
            if r.seq:
                lines.append(f"    {r.id}: WER {r.seq.wer*100:.1f}%  CER {r.seq.cer*100:.1f}%")
    for r in skipped:
        lines.append(f"  [skip] {r.id}: {r.missing}")
    return "\n".join(lines)


def _doc_json(r: DocResult) -> dict:
    d = {"id": r.id, "mode": r.mode, "type": r.type, "difficulty": r.difficulty, "era": r.era}
    if r.missing:
        d["missing"] = r.missing
    if r.seq:
        d.update(wer=round(r.seq.wer, 4), cer=round(r.seq.cer, 4),
                 n_ref_words=r.seq.n_ref_words, sub=r.seq.sub, dele=r.seq.dele, ins=r.seq.ins)
    if r.spot:
        d.update(precision=round(r.spot.precision, 4), recall=round(r.spot.recall, 4),
                 f1=round(r.spot.f1, 4), n_ref=r.spot.n_ref, n_hyp=r.spot.n_hyp)
    return d


# --------------------------------------------------------------------------- #
# --refresh: run the model on each image and cache its plain-text output
# --------------------------------------------------------------------------- #
def refresh(corpus: Path, *, model_tag: str = "default", force: bool = False,
            provider=None, settings=None) -> int:
    """Run the model on each doc's image and cache its ``.txt`` output. Needs an
    API key (or a passed provider, e.g. MockProvider for tests). Returns the
    number of hypotheses (re)generated."""
    import tempfile

    from .pipeline import process_file
    if settings is None:
        from .config import load_settings
        settings = load_settings()
    if provider is None:
        from .gemini_client import make_provider
        provider = make_provider(settings)

    corpus = Path(corpus)
    (corpus / "cache").mkdir(parents=True, exist_ok=True)
    n = 0
    for row in load_manifest(corpus):
        doc_id = row["id"].strip()
        out = cache_path(corpus, doc_id, model_tag)
        if out.is_file() and not force:
            continue
        image = corpus / (row.get("image") or f"images/{doc_id}.png").strip()
        if not image.is_file():
            continue
        with tempfile.TemporaryDirectory() as td:
            res = process_file(image, provider, settings, Path(td))
            txt = _read(Path(td) / res.txt_name) if res.txt_name else ""
        out.write_text(txt or "", "utf-8")
        n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CursBreaker transcription-accuracy benchmark")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--model-tag", default="default", help="cache tag, e.g. the model name")
    ap.add_argument("--refresh", action="store_true",
                    help="run the model + cache before scoring (uses the API, costs tokens)")
    ap.add_argument("--force", action="store_true", help="with --refresh, recompute cached hyps")
    ap.add_argument("--strict", action="store_true", help="strict normalization (keep punctuation)")
    ap.add_argument("--spotting-threshold", type=float, default=1.0)
    ap.add_argument("--json", type=Path, help="also write a JSON report to this path")
    args = ap.parse_args(argv)

    if args.refresh:
        made = refresh(args.corpus, model_tag=args.model_tag, force=args.force)
        print(f"(refreshed {made} hypothesis file(s))")
    results = score_corpus(args.corpus, model_tag=args.model_tag, strict=args.strict,
                           spotting_threshold=args.spotting_threshold)
    print(format_report(results))
    if args.json:
        args.json.write_text(json.dumps([_doc_json(r) for r in results], indent=2), "utf-8")


if __name__ == "__main__":
    main()
