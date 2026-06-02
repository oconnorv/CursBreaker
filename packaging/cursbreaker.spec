# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller spec. Build with: pyinstaller packaging/cursbreaker.spec

It encodes three things:
  1. Bundle the static web UI under ``cursbreaker/static`` so the server finds
     it at runtime.
  2. Collect uvicorn/fastapi/google-genai submodules that PyInstaller's static
     analysis misses.
  3. (Phase 1, Windows) Bundle the Tesseract engine itself -- the binary, its
     DLLs, and selected language packs -- so end users need no separate,
     admin-requiring install. The runtime resolver in ``tesseract_client.py``
     looks for the engine at ``<app>/tesseract/tesseract(.exe)`` with language
     data in ``<app>/tessdata`` first, so dropping the files there is all that
     is required -- no app-code change.

The Tesseract bundling is gated on the build OS and degrades to a no-op (with a
printed warning) when no engine is found on the build machine, so Linux/macOS
builds and Windows builds without Tesseract installed still succeed.
"""

import os
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


def _find_windows_tesseract_dir():
    """Locate an installed Tesseract on the (Windows) build machine."""
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Tesseract-OCR")
    candidates += [
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    ]
    for d in candidates:
        if (d / "tesseract.exe").is_file():
            return d
    return None


# Files we never want UPX to touch: compressing some native DLLs corrupts them
# and the bundled engine then fails to start.
upx_exclude = []

if sys.platform.startswith("win"):
    tess_dir = _find_windows_tesseract_dir()
    if tess_dir is None:
        print(
            "WARNING: no Tesseract install found on this build machine; the "
            "executable will ship WITHOUT a bundled engine. Install it first "
            "(e.g. 'choco install tesseract') to bundle it.",
            file=sys.stderr,
        )
    else:
        # The engine binary + every DLL beside it. Destinations MUST sit under
        # ``cursbreaker/`` because the frozen resolver's _app_root() is
        # ``sys._MEIPASS/cursbreaker`` -- it probes
        # ``cursbreaker/tesseract/tesseract.exe`` (mirrors the ``cursbreaker/
        # static`` datas entry above). Top-level ``tesseract/`` would ship the
        # engine but leave it undetectable.
        binaries.append((str(tess_dir / "tesseract.exe"), "cursbreaker/tesseract"))
        for dll in tess_dir.glob("*.dll"):
            binaries.append((str(dll), "cursbreaker/tesseract"))
            upx_exclude.append(dll.name)
        upx_exclude.append("tesseract.exe")

        # Ship every language pack present in the build machine's tessdata. CI
        # downloads a broad default set (see .github/workflows/build.yml) so a
        # foreign-language document never breaks OCR; a local dev build ships
        # whatever languages happen to be installed.
        src_tessdata = tess_dir / "tessdata"
        shipped = sorted(p.stem for p in src_tessdata.glob("*.traineddata"))
        for p in src_tessdata.glob("*.traineddata"):
            datas.append((str(p), "cursbreaker/tesseract/tessdata"))
        if not shipped:
            print(
                f"WARNING: Tesseract found but {src_tessdata} contains no "
                "language data (*.traineddata).",
                file=sys.stderr,
            )
        else:
            print(
                f"Bundling Tesseract {tess_dir} with {len(shipped)} "
                f"languages: {shipped}"
            )

a = Analysis(
    ["launch.py"],
    pathex=[],
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
    upx=True,
    upx_exclude=upx_exclude,
)
