from pathlib import Path

from cursbreaker import benchmark

FIX = Path(__file__).parent / "fixtures" / "benchmark_sample"


# ---- normalization ------------------------------------------------------- #
def test_normalize_policy():
    # NFC + long-s + case-fold + whitespace collapse + edge-punctuation strip.
    assert benchmark.normalize("The  ſun, ROSE.\n") == "the sun rose"
    # intra-word apostrophes/hyphens survive; bracketed markers stay whole.
    assert benchmark.normalize("don't well-known [illegible]") == "don't well-known [illegible]"
    # strict keeps surrounding punctuation.
    assert benchmark.normalize("rose.", strict=True) == "rose."


# ---- sequential CER/WER -------------------------------------------------- #
def test_wer_is_word_level_cer_is_char_level():
    # poppy->puppy and poppy->polly are BOTH one word substitution (WER 100%),
    # but differ in CER (1/5 vs 2/5) -- the distinction we hashed out.
    a = benchmark.score_sequential("poppy", "puppy")
    b = benchmark.score_sequential("poppy", "polly")
    assert a.wer == 1.0 and b.wer == 1.0
    assert abs(a.cer - 1 / 5) < 1e-9 and abs(b.cer - 2 / 5) < 1e-9


def test_wer_cer_on_sentence_examples():
    s1 = benchmark.score_sequential("This is our sample sentence", "This is oor simple sentence")
    s2 = benchmark.score_sequential("This is our sample sentence", "This is oop simrle sentence")
    assert s1.wer == 0.4 and s2.wer == 0.4               # 2 of 5 words substituted
    assert abs(s1.cer - 2 / 27) < 1e-9                   # 2 char errors / 27 chars
    assert abs(s2.cer - 4 / 27) < 1e-9                   # 4 char errors / 27 chars


def test_wer_handles_transposition_and_deletion():
    assert benchmark.score_sequential("the cat", "cat the").wer == 1.0   # both words wrong in place
    dropped = benchmark.score_sequential("This is our sample sentence", "is our sample sentence")
    assert dropped.wer == 0.2 and dropped.dele == 1 and dropped.sub == 0  # re-aligns: 1 deletion


# ---- spotting (maps) ----------------------------------------------------- #
def test_spotting_is_order_free():
    same = benchmark.score_spotting("North South East West Center", "West Center North East South")
    assert same.f1 == 1.0 and same.precision == 1.0 and same.recall == 1.0


def test_spotting_counts_misread_labels():
    s = benchmark.score_spotting("North South East West", "North Sauth East Wast")
    assert s.matched == 2 and s.f1 < 1.0  # north, east matched; sauth, wast missed


def test_spotting_fuzzy_threshold():
    # A near-miss counts as found when the threshold is relaxed.
    strict = benchmark.score_spotting("Charlotte", "Charlote")
    fuzzy = benchmark.score_spotting("Charlotte", "Charlote", threshold=0.8)
    assert strict.matched == 0 and fuzzy.matched == 1


# ---- corpus scoring + aggregation --------------------------------------- #
def test_score_corpus_fixture():
    results = benchmark.score_corpus(FIX, model_tag="fixture")
    by_id = {r.id: r for r in results}
    assert by_id["letter01"].seq.wer == 0.4
    assert by_id["map01"].spot.f1 == 1.0
    assert by_id["ceil01"].seq is not None  # scored, even though it's a ceiling doc


def test_aggregate_excludes_ceiling_and_micro_averages():
    results = benchmark.score_corpus(FIX, model_tag="fixture")
    overall, strata, spot, ceiling = benchmark.aggregate(results)
    # Only the non-ceiling sequential doc (letter01) feeds the headline.
    assert overall["docs"] == 1 and overall["wer_den"] == 5
    assert [r.id for r in ceiling] == ["ceil01"]
    assert spot["docs"] == 1 and spot["matched"] == 5
    report = benchmark.format_report(results)
    assert "OVERALL" in report and "MAPS" in report and "CEILING" in report


def test_missing_hypothesis_is_skipped_not_crashed():
    results = benchmark.score_corpus(FIX, model_tag="no-such-tag")
    assert all(r.missing for r in results)
    assert "no cached hypothesis" in results[0].missing


# ---- refresh wiring (run model -> cache), with the offline mock ---------- #
def test_refresh_smoke_with_mock(tmp_path):
    from PIL import Image

    from cursbreaker.config import Settings
    from cursbreaker.gemini_client import MockProvider

    (tmp_path / "images").mkdir()
    (tmp_path / "gt").mkdir()
    Image.new("RGB", (240, 90), "white").save(tmp_path / "images" / "d1.png")
    (tmp_path / "gt" / "d1.txt").write_text("ground truth", "utf-8")
    (tmp_path / "manifest.csv").write_text(
        "id,image,type,subtype,era,difficulty,mode,source\n"
        "d1,images/d1.png,letter,handwritten,1900-49,medium,sequential,fixture\n", "utf-8")

    made = benchmark.refresh(tmp_path, model_tag="mock",
                             provider=MockProvider(), settings=Settings())
    assert made == 1
    cached = benchmark.cache_path(tmp_path, "d1", "mock")
    assert cached.is_file() and cached.read_text("utf-8").strip()
    # It now scores cleanly (no longer "missing").
    results = benchmark.score_corpus(tmp_path, model_tag="mock")
    assert results[0].missing is None and results[0].seq is not None
