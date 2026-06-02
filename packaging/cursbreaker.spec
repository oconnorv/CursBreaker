# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller spec. Build with: pyinstaller packaging/cursbreaker.spec

It encodes:
  1. Bundle the static web UI under ``cursbreaker/static`` so the server finds it.
  2. Collect uvicorn/fastapi/google-genai submodules PyInstaller's analysis misses.
  3. Bundle the Tesseract engine + language packs so end users need no separate,
     admin-requiring install. The runtime resolver (``tesseract_client.py``) looks
     for the engine at ``<app>/tesseract/tesseract(.exe)`` -- i.e.
     ``sys._MEIPASS/cursbreaker/tesseract/...`` -- with language data in
     ``<app>/tesseract/tessdata``.

     * Windows: the engine's DLLs are copied next to the binary (Windows resolves
       DLLs from the executable's own folder).
     * macOS/Linux: PyInstaller follows the binary and auto-collects its whole
       shared-library dependency tree into the bundle root; the runtime resolver
       prepends that dir to LD_LIBRARY_PATH / DYLD_* so the libs load. (Validated:
       PyInstaller pulls the full chain and the engine runs self-contained.)

The engine bundling degrades to a no-op (with a warning) when no Tesseract is
found on the build machine, so a build always succeeds.
"""

import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("../src/cursbreaker/static", "cursbreaker/static")]
binaries = []
hiddenimports = collect_submodules("uvicorn")

for pkg in ("fastapi", "starlette", "google.genai", "pymupdf", "lxml", "pytesseract"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden


def _first(paths, kind):
    for p in paths:
        if kind == "file" and p.is_file():
            return p
        if kind == "dir" and p.is_dir():
            return p
    return None


def _windows_install_dirs():
    out = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        out.append(Path(local) / "Programs" / "Tesseract-OCR")
    out += [
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    ]
    return out


def _discover_engine():
    """Return ``(binary_path, tessdata_dir)`` for this build OS, or ``(None, None)``.

    CI sets ``CURSBREAKER_TESSERACT_BIN`` / ``CURSBREAKER_TESSDATA_DIR`` for the
    Unix runners; otherwise we probe the usual per-OS locations so a local dev
    build also bundles whatever Tesseract is installed.
    """
    is_win = sys.platform.startswith("win")
    env_bin = os.environ.get("CURSBREAKER_TESSERACT_BIN")
    env_data = os.environ.get("CURSBREAKER_TESSDATA_DIR")
    binary = Path(env_bin) if env_bin else None
    tessdata = Path(env_data) if env_data else None

    if binary is None:
        if is_win:
            binary = _first([d / "tesseract.exe" for d in _windows_install_dirs()], "file")
        else:
            which = shutil.which("tesseract")
            binary = Path(which) if which else _first(
                [Path(p) for p in (
                    "/opt/homebrew/bin/tesseract",
                    "/usr/local/bin/tesseract",
                    "/usr/bin/tesseract",
                )], "file")
    if binary is None or not binary.is_file():
        return None, None

    if tessdata is None:
        if is_win:
            tessdata = binary.parent / "tessdata"
        else:
            tessdata = _first([Path(p) for p in (
                "/usr/share/tesseract-ocr/5/tessdata",
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/usr/share/tessdata",
                "/usr/local/share/tessdata",
                "/opt/homebrew/share/tessdata",
            )], "dir")
    return binary, (tessdata if (tessdata and tessdata.is_dir()) else None)


def _allowed_languages():
    """The language allow-list from ``bundled_languages.txt`` (one code per line),
    or ``None`` to ship whatever is present -- a local-dev convenience and a guard
    against package managers (e.g. Homebrew) that pre-install a huge language set.
    """
    f = Path(SPECPATH) / "bundled_languages.txt"
    if not f.is_file():
        return None
    langs = set()
    for line in f.read_text().splitlines():
        code = line.split("#", 1)[0].strip()
        if code:
            langs.add(code)
    return langs or None


tess_binary, tessdata_dir = _discover_engine()
if tess_binary is None:
    print(
        "WARNING: no Tesseract found on this build machine; building WITHOUT a "
        "bundled engine.",
        file=sys.stderr,
    )
else:
    is_win = sys.platform.startswith("win")
    # Engine binary -> cursbreaker/tesseract (where the frozen resolver looks).
    binaries.append((str(tess_binary), "cursbreaker/tesseract"))
    if is_win:
        # Windows loads DLLs from the exe's own folder, so co-locate them.
        for dll in tess_binary.parent.glob("*.dll"):
            binaries.append((str(dll), "cursbreaker/tesseract"))
    # macOS/Linux: PyInstaller auto-collects the binary's shared-library
    # dependency tree into the bundle root; the resolver points the dynamic
    # loader there at runtime -- nothing to gather by hand.

    allow = _allowed_languages()
    shipped = []
    if tessdata_dir is None:
        print(
            f"WARNING: Tesseract at {tess_binary} but no tessdata directory "
            "found; the bundle will have NO language data.",
            file=sys.stderr,
        )
    else:
        for p in sorted(tessdata_dir.glob("*.traineddata")):
            if allow is None or p.stem in allow:
                datas.append((str(p), "cursbreaker/tesseract/tessdata"))
                shipped.append(p.stem)
    print(f"Bundling Tesseract {tess_binary} ({len(shipped)} languages: {shipped})")


a = Analysis(
    ["launch.py"],
    pathex=["../src"],  # find the cursbreaker package even without `pip install`
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="cursbreaker",
    console=True,
    disable_windowed_traceback=False,
    # UPX is off: compressing the bundled native engine / its shared libraries
    # can corrupt them, and the target users prioritize "it just works" over size.
    upx=False,
)
