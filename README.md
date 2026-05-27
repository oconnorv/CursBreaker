# CurseBreaker

**Turn handwriting into searchable text — and know where every word sits on the page.**

CurseBreaker is a small, local desktop application (it runs in your browser) that
sends document images to **Google Gemini** for high-accuracy handwriting
transcription, then produces two things for each page:

1. a plain-text transcription (`.txt`), and
2. a valid **hOCR** file (`.hocr`) that pairs the transcribed words with their
   pixel locations on the image — making the page keyword-searchable and
   findable, the same way Tesseract or ABBYY output works for printed text.

You bring your **own** Gemini API key; CurseBreaker never ships or phones home
with anyone else's.

---

## Why this exists

Recent work (notably Mark Humphries' [*Gemini 3 Solves Handwriting
Recognition*](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition))
showed that Gemini can transcribe historical cursive at near-human accuracy.
But a transcript alone can't tell you *where* a word is on the page. CurseBreaker
adds that missing half: it asks Gemini for **line bounding boxes** alongside the
text and converts everything into standards-compliant hOCR.

### How localization works

Gemini returns spatial coordinates as `box_2d = [ymin, xmin, ymax, xmax]`,
normalized to a 0–1000 grid (origin top-left). CurseBreaker:

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
git clone https://github.com/oconnorv/cursebreaker.git
cd cursebreaker
pip install .

# then run
cursebreaker
```

This starts a local server and opens `http://127.0.0.1:8765/` in your browser.
Use `cursebreaker --no-browser --port 9000` to change the defaults.

> Prefer not to install Python? See **Downloads / packaging** below for
> standalone builds.

## Get a Gemini API key

Create a key at **Google AI Studio** (<https://aistudio.google.com/apikey>).
Paste it into CurseBreaker's **Settings → Gemini API key** (stored locally on
your machine with owner-only permissions), or set the `GEMINI_API_KEY`
environment variable.

## Using it

1. **Settings** — paste your API key, pick a model, and choose a mode.
2. **Documents** — drag in (or browse for) TIFF / JPEG / PNG / GIF / PDF files.
   Bulk import and multi-page PDFs are supported.
3. **Transcribe** — watch progress, then download per-file `.txt` and `.hocr`,
   the rendered page `.png`, or everything as a `.zip`.
4. **Preview boxes** — overlay the detected line boxes on the page to verify the
   localization before you trust it.

No API key handy? Flip on **Demo mode** to exercise the whole workflow with
sample output.

### Two modes (you choose per run)

| Mode | What it does | Trade-off |
|------|--------------|-----------|
| **Two-pass** (default) | One call for the most accurate transcription, a second for line boxes; the accurate text is aligned onto the boxes. | Best accuracy, ~2× API cost/time. |
| **One-pass** | A single structured call returns text + line boxes together. | ~½ the cost and latency; transcription may be slightly less accurate. |

### Accuracy settings (defaults follow the blog's recipe)

- **Model:** `gemini-3-pro-preview` (editable; the app also lists the models
  your key can actually use).
- **Temperature:** `0`
- **Thinking level:** `low` — extra reasoning was found to *hurt* handwriting
  accuracy.
- **Media resolution:** `high`
- **Preprocessing:** gentle orientation/denoise/brightness (toggleable).

## Turning hOCR into a searchable PDF

The `.hocr` + page `.png` pair is consumable by standard tooling. For example,
with [`hocr-tools`](https://github.com/ocropus/hocr-tools):

```bash
hocr-pdf /path/to/output_folder > searchable.pdf
```

Many viewers, indexers and digital-library platforms (e.g. IIIF text-layer
workflows) also ingest hOCR directly.

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

A starter GitHub Actions workflow (`.github/workflows/build.yml`) builds
standalone executables for Windows, macOS and Linux with PyInstaller on tagged
releases. It still needs a real CI run to validate per-OS bundling of the
static UI assets.

## Credits & license

- Transcription approach inspired by Mark Humphries' *Generative History* work
  and the [Transcription Pearl](https://github.com/mhumphries2323/Transcription_Pearl)
  GUI.
- Licensed under **AGPL-3.0-or-later** (see `LICENSE`).
