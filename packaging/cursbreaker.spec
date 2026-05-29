# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller spec. Build with: pyinstaller packaging/cursbreaker.spec

NOTE: this is a starting point and has not yet been validated on a real CI
runner. The key requirements it encodes are (1) bundling the static web UI under
``cursbreaker/static`` so the server can find it at runtime, and (2) collecting
uvicorn/fastapi/google-genai submodules that PyInstaller's static analysis misses.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("../src/cursbreaker/static", "cursbreaker/static")]
binaries = []
hiddenimports = collect_submodules("uvicorn")

for pkg in ("fastapi", "starlette", "google.genai", "pymupdf", "lxml"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

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
)
