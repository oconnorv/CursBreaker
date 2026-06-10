# Transcription-accuracy benchmark (Track A)

Measures CursBreaker transcription quality (**CER** and **WER**) against a human
ground-truth corpus of 19th–20th-century record types, reported **by stratum**
(type × difficulty × era). Scorer: [`src/cursbreaker/benchmark.py`](../src/cursbreaker/benchmark.py).

This is a **periodic benchmark**, not a per-commit CI gate: generating hypotheses
calls the Gemini API (costs tokens), but **scoring runs against cached outputs**,
so it's free and deterministic once the cache exists.

## Layout
```
benchmark/corpus/
  manifest.csv          one row per document (schema below)
  gt/<id>.txt           human ground-truth transcription (you write this)
  images/<id>.<ext>     the page image (only needed to (re)generate hypotheses)
  cache/<id>.<tag>.txt  cached model output (so scoring is offline + free)
```

## `manifest.csv` columns
| column | meaning | values |
|---|---|---|
| `id` | unique id (also the gt/cache filename stem) | e.g. `letter_1893_01` |
| `image` | image path relative to `benchmark/corpus/` | `images/letter_1893_01.png` |
| `type` | document type | letter, postcard, newsletter, newspaper, map |
| `subtype` | medium | handwritten, typewriter, printed, microfilm |
| `era` | era bucket | `1850-99`, `1900-49`, `1950-99` |
| `difficulty` | handwriting difficulty / ceiling flag | easy, medium, hard, ceiling |
| `mode` | scoring mode | `sequential` (CER/WER) or `spotting` (maps) |
| `source` | provenance / credit | free text |

- **difficulty** — easy = hand-printing, medium = cursive, hard = marginalia /
  odd angles, **ceiling** = expected-impossible (e.g. cross-written): scored and
  reported, but **excluded from headline averages**.
- **newspapers** — don't score a whole multi-column page. Crop each column /
  article into its own image and add it as a `sequential` doc (within one column,
  reading order is unambiguous, so WER is fair).
- **maps** — use `spotting`: order-free label precision/recall/F1 (a misread place
  name counts against both; word order doesn't).

## Normalization (applied identically to truth and hypothesis)
Unicode NFC; long-s (ſ) → s; case-fold; collapse whitespace; strip punctuation at
token edges (keeping intra-word apostrophes/hyphens). `--strict` keeps all
punctuation. Bracketed markers (`[illegible]`, `[?]`, `[blank]`) are preserved as
single tokens. CER/WER are meaningless without a fixed policy — this is it.

## Adding a document
1. Drop the page image in `images/` and add a `manifest.csv` row.
2. Generate the model hypothesis:
   `python -m cursbreaker.benchmark --refresh` (needs your API key; costs tokens;
   caches to `cache/<id>.<tag>.txt`). Or run CursBreaker yourself and copy its
   `.txt` into `cache/<id>.default.txt`.
3. **Correct it into ground truth:** copy the cached hypothesis to `gt/<id>.txt`
   and fix it against the image. Correcting is far faster than transcribing blind
   — but correct *honestly* (don't rubber-stamp the model's errors). Transcribe a
   few docs independently to anchor the set.

## Running
```bash
python -m cursbreaker.benchmark                          # score cached hyps, print report
python -m cursbreaker.benchmark --model-tag gemini-3.1-pro --json report.json
python -m cursbreaker.benchmark --refresh                # (re)run the model first (API cost)
python -m cursbreaker.benchmark --strict                 # keep punctuation
```
