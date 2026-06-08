# CursBreaker

**Turn handwriting into searchable text — and know where every word sits on the page.**

CursBreaker is a small, local desktop application (it runs in your browser) that
sends document images to **Google Gemini** for high-accuracy handwriting
transcription, then produces, for each page:

1. a plain-text transcription (`.txt`);
2. a valid **hOCR** file (`.hocr`) that pairs the transcribed words with their
   pixel locations on the image — making the page keyword-searchable and
   findable, the same way Tesseract or ABBYY output works for printed text; and
3. an **ALTO XML** file (`.alto.xml`) — the same word geometry in the Library of
   Congress's preservation format, for ALTO/METS-based repositories.

You bring your **own** Gemini API key; CursBreaker never ships or phones home
with anyone else's.

---

## Why this exists

Recent work (notably Mark Humphries' [*Gemini 3 Solves Handwriting
Recognition*](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition))
showed that Gemini can transcribe historical cursive at near-human accuracy.
But a transcript alone can't tell you *where* a word is on the page. CursBreaker
adds that missing half: it asks Gemini for **line bounding boxes** alongside the
text and converts everything into standards-compliant hOCR and ALTO XML.

### How localization works

Gemini returns spatial coordinates as `box_2d = [ymin, xmin, ymax, xmax]`,
normalized to a 0–1000 grid (origin top-left). CursBreaker:

1. converts those to real pixels using the page dimensions;
2. emits one hOCR `ocr_line` per detected line; and
3. **synthesizes per-word boxes** by splitting each line box horizontally in
   proportion to word length — so individual words stay searchable, without
   relying on per-character detection (which is unreliable for connected
   cursive).

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

## Get a Gemini API key

**New to this? Don't worry.** CursBreaker uses Google's Gemini AI to read
handwriting, and Google needs to know the requests are coming from you. An
**API key** is how you do that: think of it as a long password that lets
CursBreaker use Gemini on your behalf. You create it once, paste it into
CursBreaker, and you're set. You bring your **own** key — CursBreaker never
ships or borrows anyone else's.

You'll need a Google account (the same kind you use for Gmail). Then:

1. Go to **Google AI Studio**: <https://aistudio.google.com/apikey>
2. **Sign in** with your Google account if you're prompted.
3. Click **Create API key**. (If you don't see that button, click **Get API
   key** in the left-hand menu first.) If Google asks which project to use, let
   it **create a new one** for you — the name doesn't matter.
4. Google shows you the key — a long string of letters and numbers that starts
   with `AIza…`. Click to **copy** it.
5. Back in CursBreaker, open **Settings**, paste the key into the **Gemini API
   key** box, and click **Save key**. That's it.

Your key is stored locally on your own computer (with owner-only file
permissions) and is sent only to Google's Gemini API when you transcribe —
never to us or anyone else. You can remove it at any time with the **Clear**
button. (Prefer not to paste it into the app? Set the `GEMINI_API_KEY`
environment variable before launching instead.)

### Does it cost money?

Google offers a **free tier** you can start with — **no credit card required** —
so you can try CursBreaker without paying anything; the free tier just has daily
limits. For larger jobs you can turn on paid usage (billing) in your Google
account, and Google then charges based on how much text the model reads and
writes. CursBreaker helps you stay in control: it shows an **estimated cost
before every run** and the **actual usage afterward**, and the model dropdown
lists each model's published price. See Google's current rates at
<https://ai.google.dev/gemini-api/docs/pricing> — those come straight from
Google and can change, so the in-app figures are estimates, not a guarantee.

> **Keep your key private.** Treat it like a password: anyone who has it can run
> up usage on your account. Don't paste it into emails, chats, or screenshots.
> If a key is ever exposed, delete it in Google AI Studio and create a new one.

## Using it

1. **Settings** — paste your API key, pick a model, and choose a mode.
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
| **Handwriting** (default) | Gemini transcribes the whole page — printed text included — and its transcription is always the authoritative text. Optionally, **Tesseract refines word *positions*** where its reading agrees with Gemini's (real per-word boxes), without ever changing the wording. | Any page with handwriting, including mixed printed + handwritten (typeset letterhead + handwritten body, printed headers + handwritten entries). |
| **Printed only** | Tesseract OCRs the whole page locally. **No Gemini call** (no API cost). | Fully typeset documents. |

### Two modes for handwriting (Two-pass / One-pass)

| Mode | What it does | Trade-off |
|------|--------------|-----------|
| **Two-pass** (default) | One call for the most accurate transcription, a second for line boxes; the accurate text is aligned onto the boxes. | Best accuracy, ~2× API cost/time. |
| **One-pass** | A single structured call returns text + line boxes together. | ~½ the cost and latency; transcription may be slightly less accurate. |

### Accuracy settings (defaults)

- **Model:** `gemini-3.1-pro-preview` by default — pick from a short curated
  dropdown (Gemini 3.1 Pro · 3.5 Flash · 3.1 Flash-Lite); the app shows each
  model's published price and uses it to estimate cost automatically.
- **Temperature:** `0.3`
- **Thinking budget:** `128` tokens — Humphries' finding is that extra
  reasoning *hurts* handwriting accuracy, so the default is deliberately
  minimal. The Advanced panel also exposes a coarser "thinking level"
  (`low` / `medium` / `high`); when that's set it overrides the budget.
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
  current one from the dropdown.
- **Known failure modes** (from the underlying model): marginalia and text
  squeezed between lines transcribe poorly; very dense or multi-column pages may
  drop or merge lines.
- **Duplicate filenames** within a single batch will overwrite each other — give
  files unique names.
- **Privacy:** your images and key are sent to Google's Gemini API when you
  transcribe. Nothing is sent anywhere else.

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
