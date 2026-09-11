# CursBreaker

**Turn handwriting into searchable text — and know where every word sits on the page.**

CursBreaker is a small, local desktop application (it runs in your browser) that
sends document images to the AI service of your choice — **Google Gemini**,
**Anthropic Claude** or **OpenAI** — for high-accuracy handwriting
transcription, then produces, for each page:

1. a plain-text transcription (`.txt`);
2. a valid **hOCR** file (`.hocr`) that pairs the transcribed words with their
   pixel locations on the image — making the page keyword-searchable and
   findable, the same way Tesseract or ABBYY output works for printed text; and
3. an **ALTO XML** file (`.alto.xml`) — the same word geometry in the Library of
   Congress's preservation format, for ALTO/METS-based repositories.

You bring your **own** API key for whichever service you use; CursBreaker never
ships or phones home with anyone else's. Pick the service your institution has
approved — the pipeline, the settings and the output files are identical either
way, and each service keeps its own saved key, so switching back and forth
never means pasting a key again.

---

## Why this exists

Recent work (notably Mark Humphries' [*Gemini 3 Solves Handwriting
Recognition*](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition))
showed that a frontier model can transcribe historical cursive at near-human
accuracy. But a transcript alone can't tell you *where* a word is on the page.
CursBreaker adds that missing half: it asks the model for **line bounding
boxes** alongside the text and converts everything into standards-compliant
hOCR and ALTO XML.

### How localization works

Every provider is asked for the same thing, in the same words (the prompts live
in one place, `src/cursbreaker/prompts.py`): spatial coordinates as
`box_2d = [ymin, xmin, ymax, xmax]`, normalized to a 0–1000 grid (origin
top-left). CursBreaker:

1. converts those to real pixels using the page dimensions;
2. emits one hOCR `ocr_line` per detected line; and
3. **synthesizes per-word boxes** by splitting each line box horizontally in
   proportion to word length — so individual words stay searchable, without
   relying on per-character detection (which is unreliable for connected
   cursive).

How precisely a model places those boxes varies between services, and it's the
half most worth checking on your own material before committing to a large
batch. Gemini's spatial grounding is what the original recipe was built around;
if line boxes from another service land loosely, turn on **Refine word
positions with Tesseract**, which keeps the model's transcription as the text
and uses local OCR only to place the words.

---

## Install

Requires Python 3.10+.

```bash
# from source
git clone https://github.com/oconnorv/cursbreaker.git
cd cursbreaker
pip install .

# then run
cursbreaker
```

This starts a local server and opens `http://127.0.0.1:8765/` in your browser.
Use `cursbreaker --no-browser --port 9000` to change the defaults.

> Prefer not to install Python? See **Downloads / packaging** below for
> standalone builds.

### Tesseract (for printed text)

CursBreaker uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
locally for typeset text: it powers **Printed only** mode and the optional
*word-position refinement* in Handwriting mode. Tesseract is excellent on clean
printed text, runs without an API call, and gives *real* per-word boxes and
confidences — strictly better than the proportional word-box synthesis used for
handwriting alone. Handwriting mode itself never needs it.

**The standalone downloads bundle Tesseract** (engine + a default language set),
so end users need no separate, admin-requiring install — Printed-only works out
of the box.

Running **from source** (`pip install .`)? The `pytesseract` wrapper ships as a
dependency, but the engine itself isn't bundled — install it only if you want
Printed-only mode:

```bash
# Linux (Debian/Ubuntu)
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: use the UB-Mannheim installer
#   https://github.com/UB-Mannheim/tesseract/wiki
```

The Settings panel speaks up only when the engine is *missing*, and the Advanced
"Tesseract language" box lists the packs it can see (install e.g.
`tesseract-ocr-fra` to add French). If the engine is installed but not on your
`PATH` (common on Windows, where the UB-Mannheim installer doesn't always add
it), point CursBreaker at it with the `TESSERACT_CMD` environment variable:

```bash
# Windows (PowerShell)
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

CursBreaker also auto-checks the well-known install locations on each OS, so in
most cases no override is needed.

## Get an API key

**New to this? Don't worry.** CursBreaker uses a hosted AI service to read
handwriting, and that service needs to know the requests are coming from you.
An **API key** is how you do that: think of it as a long password that lets
CursBreaker use the service on your behalf. You create it once, paste it into
CursBreaker, and you're set. You bring your **own** key — CursBreaker never
ships or borrows anyone else's.

Pick whichever service you (or your institution) prefer in **Settings → AI
service**, then create a key for it:

| Service | Where to create a key | Environment variable |
| --- | --- | --- |
| Google Gemini | <https://aistudio.google.com/apikey> | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| Anthropic Claude | <https://console.anthropic.com/settings/keys> | `ANTHROPIC_API_KEY` |
| OpenAI | <https://platform.openai.com/api-keys> | `OPENAI_API_KEY` |

Sign in, create a key, copy it, then open **Settings** in CursBreaker, paste it
into the **API key** box and click **Save key**. The in-app help under that box
walks through the same steps for whichever service you've selected.

Keys are stored locally on your own computer (with owner-only file permissions)
and are sent only to the service you chose, when you transcribe — never to us
or anyone else. Each service has its own slot, so a saved Gemini key survives
switching to Claude and back. Remove one at any time with the **Clear** button,
which clears only the service you're currently on. Prefer not to paste a key
into the app at all? Set the environment variable from the table above before
launching; it overrides anything saved in the app.

### Does it cost money?

Yes — you pay the service directly for what you use, and CursBreaker adds
nothing on top. Google offers a **free tier** you can start with (no credit
card required, with daily limits); Anthropic and OpenAI are paid from the
start, though new accounts often come with trial credit.

CursBreaker helps you stay in control: it shows an **estimated cost before
every run** and the **actual token usage afterward**, and the model dropdown
lists each model's published price.

One caveat worth knowing: **OpenAI can't price the page images in advance.**
Gemini and Claude both expose a free token-counting endpoint, so their
pre-flight estimate measures the real input cost of your pages before anything
runs. OpenAI has no equivalent, so its estimate covers the output side only and
is shown as a minimum ("at least …"), with the shortfall spelled out — the
image half, usually the larger one, is missing rather than zero. The **actual**
cost reported after a run is complete for every provider, because the response
tells us the real input tokens.

Published rates change, so every in-app figure is an estimate, not a guarantee.
Rates live in `src/cursbreaker/pricing.py`; edit the numbers and bump
`PRICES_AS_OF` to refresh them. Where a rate is introductory and its expiry is
already published — Gemini 3.8 Flash doubles on 2027-01-01 — the catalog holds
both prices and quotes whichever is in force that day, so estimates stay right
across the change with nothing to remember.
Current rates: [Gemini](https://ai.google.dev/gemini-api/docs/pricing) ·
[Claude](https://claude.com/pricing#api) ·
[OpenAI](https://openai.com/api/pricing/).

> **Keep your key private.** Treat it like a password: anyone who has it can run
> up usage on your account. Don't paste it into emails, chats, or screenshots.
> If a key is ever exposed, delete it in that service's console and create a
> new one.

## Using it

1. **Settings** — choose your AI service, paste that service's API key, pick a
   model, and choose a mode.
2. **Documents** — drag in (or browse for) TIFF / JPEG / PNG / GIF / PDF files.
   Bulk import and multi-page PDFs are supported.
3. **Transcribe** — watch progress, then download a per-file searchable `.pdf`,
   `.txt`, `.hocr`, and `.alto.xml`, or everything (including the page images) as
   a `.zip`.
4. **Preview boxes** — overlay the detected line boxes on the page to verify the
   localization before you trust it.

### Content type (you choose per batch)

| Content type | What it does | When to use |
|---|---|---|
| **Handwriting** (default) | The AI service transcribes the whole page — printed text included — and its transcription is always the authoritative text. Optionally, **Tesseract refines word *positions*** where its reading agrees with the model's (real per-word boxes), without ever changing the wording. | Any page with handwriting, including mixed printed + handwritten (typeset letterhead + handwritten body, printed headers + handwritten entries). |
| **Printed only** | Tesseract OCRs the whole page locally. **No API call** (no cost). | Fully typeset documents. |

### Two modes for handwriting (Two-pass / One-pass)

| Mode | What it does | Trade-off |
|------|--------------|-----------|
| **Two-pass** (default) | One call for the most accurate transcription, a second for line boxes; the accurate text is aligned onto the boxes. | Best accuracy, ~2× API cost/time. |
| **One-pass** | A single structured call returns text + line boxes together. | ~½ the cost and latency; transcription may be slightly less accurate. |

### Accuracy settings (defaults)

- **Model:** each service starts on its flagship — `gemini-3.1-pro-preview`,
  `claude-opus-5`, `gpt-6-astra` — picked from a short curated dropdown
  (Gemini 3.1 Pro · 3.8 Flash; Claude Opus 5 · Sonnet 5 ·
  Haiku 4.5; GPT-6 Astra · GPT-5.6 Sol · Terra · Luna), with each model's
  published price shown and used to estimate cost automatically. The flagship
  is the default because it reads difficult hands best; for bulk work on
  legible material the lighter models cost a fraction as much, and the
  dropdown shows exactly how much.
- **Temperature:** `0.3`
- **Thinking budget:** `128` tokens — Humphries' finding is that extra
  reasoning *hurts* handwriting accuracy, so the default is deliberately
  minimal. The Advanced panel also exposes a coarser "thinking level"
  (`low` / `medium` / `high`); when that's set it overrides the budget. Claude
  takes no token budget, so the same setting maps onto its reasoning *effort*
  and defaults to `low` for the same reason.
- **Media resolution:** `high`
- **Preprocessing:** gentle orientation/denoise/brightness (toggleable).

## Using the coordinate output (hOCR / ALTO)

The `.hocr` + page `.png` pair is consumable by standard tooling. For example,
with [`hocr-tools`](https://github.com/ocropus/hocr-tools):

```bash
hocr-pdf /path/to/output_folder > searchable.pdf
```

(CursBreaker already writes a searchable `.pdf` for you; this is just one example
of what the coordinate layer enables.)

Many viewers, indexers and digital-library platforms ingest these coordinate
formats directly: **hOCR** for IIIF text-layer workflows (e.g. Islandora/Mirador
search-result highlighting), and **ALTO XML** for ALTO/METS-based repositories.

## Notes & limitations

- **Localization is line-level.** Word boxes are synthesized from line boxes, so
  they're approximate within a line but reliable for search/highlight.
- **Model names change.** If a default model is unavailable to your key, pick a
  current one from the dropdown. (On OpenAI, a rejected model id comes back
  with the list of models your key can actually see. On Gemini, a retired
  model falls back to another catalogued one so a long batch survives — and
  the cost is then reported at the price of the model that actually ran.)
- **Known failure modes** (from the underlying model): marginalia and text
  squeezed between lines transcribe poorly; very dense or multi-column pages may
  drop or merge lines.
- **Duplicate filenames** within a single batch will overwrite each other — give
  files unique names.
- **Privacy:** your images and key are sent to whichever AI service you select,
  when you transcribe. Nothing is sent anywhere else, and switching service
  changes who receives them — worth checking against your institution's policy
  before a run.
- **Box quality varies by service.** Gemini's spatial grounding is what this
  workflow was built around; other services transcribe well but may place line
  boxes less precisely. Check a sample page's box preview before a large batch,
  and consider **Refine word positions with Tesseract**.

## Downloads / packaging

A GitHub Actions workflow (`.github/workflows/build.yml`) builds standalone
executables for Windows, macOS and Linux with PyInstaller on tagged releases. It
bundles the static UI assets and the **Tesseract engine** (plus a default
language set) so end users need zero separate, admin-requiring install —
Printed-only mode works out of the box. The runtime resolver looks for the
bundled engine first and falls back to a system install; see
[`docs/bundling-tesseract.md`](docs/bundling-tesseract.md) for the approach.

## Credits & license

- Transcription approach inspired by Mark Humphries' *Generative History* work
  and the [Transcription Pearl](https://github.com/mhumphries2323/Transcription_Pearl)
  GUI.
- Licensed under **AGPL-3.0-or-later** (see `LICENSE`).
