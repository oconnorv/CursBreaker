# Bundling the Tesseract engine with CursBreaker

> Status: **Phase 1 implemented (Windows)**. The Windows build now bundles the
> engine, and the resolver auto-detects a bundled engine, a portable build in an
> app-managed folder, and per-user installs — all without administrator rights.
> macOS/Linux bundling (Phase 2) and a possible in-app downloader remain future
> work. Windows is the priority platform.

## No-admin options (important)

Some users cannot run installers or write to `C:\Program Files`. Phase 1 gives
them two admin-free paths, both discovered automatically by the resolver:

1. **Bundled engine** — the Windows `.exe` ships Tesseract inside it; nothing to
   install. This is the default for users who download a release build.
2. **Portable drop-in** — a user can unzip a portable Tesseract into the
   app-managed folder (`platformdirs.user_data_dir("CursBreaker")/tesseract`,
   i.e. `%LOCALAPPDATA%\CursBreaker\tesseract` on Windows) — the binary plus a
   `tessdata/` subfolder — and CursBreaker finds it on next launch.

The resolver also probes the per-user install location (`%LOCALAPPDATA%\Programs\
Tesseract-OCR`, where winget and non-admin UB-Mannheim installs land) before the
machine-wide `C:\Program Files` locations. Installing Tesseract is therefore
entirely **optional**: handwriting mode never needs it, and when a bundled build
is used, printed/mixed modes work out of the box.

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
(`resolve_tesseract`), no-admin options first:

1. Explicit override — `TESSERACT_CMD` env var or the `tesseract_cmd` setting.
2. **Bundled binary** — `_app_root()/tesseract/tesseract(.exe)`, where
   `_app_root()` is `sys._MEIPASS/cursbreaker` in a PyInstaller build.
3. **Portable drop-in** — `_managed_dir()/tesseract(.exe)` (the app-managed
   `user_data_dir`), so a user can add an engine without an installer.
4. Per-user install — `%LOCALAPPDATA%\Programs\Tesseract-OCR` on Windows.
5. Machine-wide locations (e.g. `C:\Program Files\Tesseract-OCR`).
6. Pytesseract's own PATH lookup (the universal fallback).

It also sets `TESSDATA_PREFIX` to the first existing `tessdata` directory among:
beside the binary, `_app_root()/tessdata`, and `_managed_dir()/tessdata`.
Tesseract 5.x requires this to point **directly at** the `tessdata` folder (the
parent-directory form errors out). **Consequence:** dropping `tesseract(.exe)`
under `tesseract/` and language data under `tessdata/` is picked up with **no
code change**. The status badge reports *how* the engine was found (bundled /
portable / system / PATH) via the `source` field.

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

- Ship language data under a bundled `tessdata/` directory; the resolver
  auto-sets `TESSDATA_PREFIX` to it.
- **Languages are bundled broadly by default, not opt-in.** A document in another
  language shouldn't break OCR, and the target users don't care about a few extra
  MB. The Windows CI job downloads a generous default set (European incl.
  historical/classical, Cyrillic, Greek, and the major world languages) into
  `tessdata` before packaging; the spec then ships **every** `*.traineddata`
  present. Edit the `$langs` list in `.github/workflows/build.yml` to adjust.
- There are three data flavors: `tessdata_fast` (smallest), `tessdata`
  (standard), `tessdata_best` (largest/most accurate). We use `tessdata_fast`
  for the downloaded packs so a broad set stays small.

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
- **Phase 1 — Windows (done):** the spec copies the engine binary, its DLLs and
  **all** installed language packs into the bundle (`cursbreaker/tesseract/` +
  `cursbreaker/tesseract/tessdata/`), with `upx_exclude` so UPX can't corrupt the
  DLLs. The Windows CI job runs `choco install tesseract` (eng + osd) and then
  downloads a broad default language set (`tessdata_fast`). The resolver adds
  no-admin discovery (bundled engine, portable drop-in folder, per-user
  `%LOCALAPPDATA%` install).
  *Remaining validation:* a real CI run and a smoke test on a clean Windows VM
  with no separate Tesseract install.
- **Phase 2 — macOS & Linux:** repeat with `brew`/`apt` plus `otool`/`ldd`
  library gathering (the spec's Tesseract block is currently gated to Windows).
- **Phase 3 — evaluate alternatives:** an in-app portable-engine downloader
  (no-admin, fetches into the managed folder), and/or approach C (pure-pip OCR)
  behind an engine abstraction if native bundling proves fragile.

Every phase lands the engine in the same place the resolver already checks
(`_app_root()/tesseract`), so none of them require touching the transcription
code.
```

