# Bundling the Tesseract engine with CursBreaker

> Status: **exploration**. This document lays out the options and a recommended
> path. The runtime *plumbing* needed to support a bundled engine already exists
> (see "How detection already supports a bundle" below); shipping the binary is a
> packaging task, not an app-code change. Windows is the priority platform.

## Goal & current state

CursBreaker uses Tesseract for printed text (the **Mixed** and **Printed only**
content types). Two pieces are required at runtime:

1. **`pytesseract`** — the thin Python wrapper. As of this change it is a
   declared dependency, so `pip install .` (and the PyInstaller build) always
   include it. This was the most common reason detection failed.
2. **The Tesseract engine binary** — a C++ program (`tesseract` /
   `tesseract.exe`) plus its language data (`tessdata/*.traineddata`). This is
   still installed separately by the user today.

The aim of bundling is to ship #2 *inside* the standalone executables that
`packaging/cursbreaker.spec` + `.github/workflows/build.yml` produce, so a
non-technical user downloads one artifact and Printed/Mixed modes "just work."

## How detection already supports a bundle

`src/cursbreaker/tesseract_client.py` resolves the binary in this order
(`resolve_tesseract`):

1. Explicit override — `TESSERACT_CMD` env var or the `tesseract_cmd` setting.
2. **Bundled binary** — `_app_root()/tesseract/tesseract(.exe)`, where
   `_app_root()` is `sys._MEIPASS/cursbreaker` in a PyInstaller build.
3. Well-known per-OS locations (e.g. `C:\Program Files\Tesseract-OCR`).
4. Pytesseract's own PATH lookup (the universal fallback).

It also sets `TESSDATA_PREFIX` to `_app_root()/tessdata` (or a `tessdata/` dir
next to the binary) when that directory exists. **Consequence:** if the build
drops `tesseract(.exe)` under `tesseract/` and the language data under
`tessdata/`, step 2 finds it with **no code change**. Everything below is about
getting those files into the bundle correctly per OS.

## Approaches

### A. True PyInstaller bundle (recommended end state)

Ship the engine binary, its transitive shared libraries, and `tessdata` inside
the executable. Wire them into `packaging/cursbreaker.spec` (today
`binaries = []`):

```python
# illustrative — paths come from the CI runner where tesseract was installed
binaries += [(r"C:\Program Files\Tesseract-OCR\tesseract.exe", "tesseract")]
binaries += [(dll, "tesseract") for dll in glob("C:/Program Files/Tesseract-OCR/*.dll")]
datas    += [(r"C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata", "tessdata")]
```

- **Pros:** single artifact; offline; deterministic version.
- **Cons:** must gather the binary's transitive shared libraries per OS (see
  "The hard part"); larger artifact.

### B. Auto-install via the OS package manager on first run

Detect a missing engine and offer to run `winget install UB-Mannheim.TesseractOCR`
(Windows), `brew install tesseract` (macOS), or `apt install tesseract-ocr`
(Linux).

- **Pros:** tiny artifact; system-managed updates.
- **Cons:** needs network and often elevation; behavior varies by machine.
  Good as a *fallback/offer*, not the default.

### C. Pure-pip OCR alternative (no system binary)

Swap Tesseract for an OCR engine that ships as wheels — e.g.
`rapidocr-onnxruntime` (ONNX) or `easyocr` (PyTorch). These install with `pip`
and need no separate binary.

- **Pros:** no native-binary bundling problem at all.
- **Cons:** different accuracy profile; `easyocr` pulls in a large PyTorch stack;
  would require an engine-abstraction layer behind the current `tesseract_client`
  surface. Out of scope for now, but worth revisiting if native bundling proves
  painful on a given OS.

## The hard part: transitive shared libraries

A system `tesseract` binary dynamically links several libraries that PyInstaller
will **not** discover automatically (it only analyzes the Python import graph):

- leptonica, libpng, libjpeg, libtiff, libwebp, zlib (and, on Linux, `libgomp`).

Per OS, enumerate them and add each to `binaries`:

- **Windows:** `dumpbin /dependents tesseract.exe` (or Dependencies.exe); in
  practice the UB-Mannheim install folder already contains the needed `*.dll`,
  so globbing that folder is usually enough.
- **macOS:** `otool -L $(which tesseract)`; copy the `*.dylib` and consider
  `install_name_tool` if load paths are absolute.
- **Linux:** `ldd $(which tesseract)`; copy the `*.so` files.

CI must install the engine on each runner **before** the PyInstaller step
(`.github/workflows/build.yml`, currently `pip install . pyinstaller` then
`pyinstaller packaging/cursbreaker.spec`):

```yaml
# Windows runner
- run: choco install --no-progress tesseract
# macOS runner
- run: brew install tesseract
# Linux runner
- run: sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
```

**UPX caveat:** the spec sets `upx=True`. UPX can corrupt some Windows DLLs (and
occasionally macOS dylibs). If the bundled engine crashes on launch, add the
offending files to `upx_exclude` or disable UPX for the engine binaries.

## Language data (`tessdata`) and `TESSDATA_PREFIX`

- Ship at least `eng.traineddata` under a bundled `tessdata/` directory; the
  resolver auto-sets `TESSDATA_PREFIX` to it.
- Language packs are large, so keep extra languages opt-in (a later "download
  language" feature, or document adding files to the user `tessdata`).
- There are three data flavors: `tessdata_fast` (smallest), `tessdata`
  (standard), `tessdata_best` (largest/most accurate). Prefer `fast` or
  standard for size.

## Licensing

- **Tesseract:** Apache License 2.0.
- **`tessdata` traineddata:** Apache License 2.0.
- **CursBreaker:** AGPL-3.0-or-later (`pyproject.toml`).

Apache-2.0 is one-way compatible into (A)GPL-3.0, so bundling is fine. Retain
Tesseract's `LICENSE`/`NOTICE` in the distribution (e.g. a `licenses/` folder in
the artifact) to satisfy the Apache attribution requirement.

## Approximate size impact

- Engine + core shared libraries: ~10–15 MB.
- `eng.traineddata`: ~4 MB (`tessdata_fast`) up to ~15 MB (`tessdata_best`).
- **Net:** roughly **+15–30 MB** per platform artifact for English-only.

## Recommended phased path

- **Phase 0 (done):** `pytesseract` is a dependency; the resolver searches
  bundled → well-known → PATH; diagnostics name the actual gap. No binary
  bundled yet.
- **Phase 1 — Windows only:** add a Windows `binaries`/`datas` block to the spec
  and a `choco install tesseract` step to the Windows CI job. Validate with a
  real CI run and a clean Windows VM (no separate Tesseract install).
- **Phase 2 — macOS & Linux:** repeat with `brew`/`apt` plus `otool`/`ldd`
  library gathering.
- **Phase 3 — evaluate alternatives:** if native bundling is too fragile on a
  platform, reconsider approach C (pure-pip OCR) behind an engine abstraction.

Every phase lands the engine in the same place the resolver already checks
(`_app_root()/tesseract`), so none of them require touching the transcription
code.
```

