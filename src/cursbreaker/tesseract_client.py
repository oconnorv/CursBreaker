"""Local Tesseract OCR for printed text.

Tesseract is a C++ system binary (apt/brew/Windows installer); ``pytesseract``
is just a thin wrapper. So both the binary AND the Python package can be absent
in a CursBreaker install — when they are, the rest of the app still works in
handwriting-only mode and any path that needs Tesseract raises a clear error.

Why integrate Tesseract at all when Gemini already transcribes handwriting?
For *printed* text it is:

* Local and free (no API cost, no network).
* Near-state-of-the-art accuracy on clean printed scans.
* Returns **real per-word boxes and per-word confidences**, which is strictly
  better than the proportional word-box synthesis we do for Gemini lines.

This module exposes a small, side-effect-free surface used by the pipeline:

* :func:`is_available` — quick environmental check for the UI status badge.
* :func:`available_languages` — what language packs the local install supports.
* :func:`transcribe_region` — OCR a (possibly cropped) PIL image and return
  ``TranscribedLine`` objects whose ``words`` field carries real per-word data,
  with coordinates offset back to page space when caller passes ``offset``.
* :func:`transcribe_page` — convenience wrapper for the whole-page case.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .models import OcrWord, PixelBox, TranscribedLine


# pytesseract level codes (see the docs). We only care about word rows.
_WORD_LEVEL = 5


def _app_root() -> Path:
    """Base directory for bundled resources.

    When frozen by PyInstaller the app data lives under
    ``sys._MEIPASS/cursbreaker`` (matching the spec's ``datas`` destination);
    otherwise it's this package directory -- the same convention
    ``searchable_pdf.py`` uses to find its bundled font.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "cursbreaker"
    return Path(__file__).parent


def _point_loader_at_bundle(cmd: Optional[str]) -> None:
    """Make the bundled engine's co-bundled shared libraries loadable.

    On macOS/Linux PyInstaller collects the engine's whole dependency tree into
    the bundle root (``sys._MEIPASS``) but does not rpath the binary, so for the
    *bundled* engine we prepend that dir to the platform's dynamic-loader search
    path -- the tesseract subprocess then finds its libs there. Acts only in a
    frozen build, only for the bundled binary, and is a no-op on Windows (DLLs
    sit beside the .exe and load automatically)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass or not cmd:
        return
    if os.path.dirname(cmd) != str(_app_root() / "tesseract"):
        return  # not our bundled engine -- leave the loader path alone
    if sys.platform == "darwin":
        keys = ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")
    elif sys.platform.startswith("win"):
        return
    else:
        keys = ("LD_LIBRARY_PATH",)
    for key in keys:
        parts = [p for p in os.environ.get(key, "").split(os.pathsep) if p]
        if meipass not in parts:
            os.environ[key] = os.pathsep.join([meipass, *parts])


def _managed_dir() -> Path:
    """App-managed folder a user can drop a *portable* Tesseract into.

    This is the no-admin escape hatch: a user who cannot run installers can
    unzip a portable Tesseract here (binary + a ``tessdata`` subfolder) and
    CursBreaker will find it. We only ever *read* from this folder in Phase 1.
    """
    from platformdirs import user_data_dir

    return Path(user_data_dir("CursBreaker", appauthor=False)) / "tesseract"


def _is_file(path: str) -> bool:
    """Indirection point so tests can simulate which candidate paths exist."""
    return os.path.isfile(path)


def _candidate_binaries(platform: Optional[str] = None) -> list[str]:
    """Likely tesseract locations to probe before falling back to PATH.

    Ordered so the no-admin options win first: a binary bundled inside the app,
    then a portable build the user dropped in the managed folder, then per-user
    installs, then machine-wide installs. ``platform`` defaults to
    ``sys.platform`` but is a parameter so every OS branch is unit-testable from
    any host. PATH itself is intentionally absent here -- pytesseract handles
    that universal fallback.
    """
    plat = platform or sys.platform
    is_windows = plat.startswith("win")
    exe = "tesseract.exe" if is_windows else "tesseract"
    # (1) shipped inside the app bundle and (2) a portable build in the
    # app-managed folder -- both usable with no administrator rights.
    cands = [
        str(_app_root() / "tesseract" / exe),
        str(_managed_dir() / exe),
    ]
    if is_windows:
        # Per-user installs (winget, or a non-admin UB-Mannheim install) land
        # under %LOCALAPPDATA% and need no administrator.
        local = os.environ.get("LOCALAPPDATA")
        if local:
            cands.append(
                str(Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
            )
        # Machine-wide installs (admin to create, but free to read afterwards).
        cands += [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif plat == "darwin":
        cands += ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    else:  # linux and other unixes
        cands += ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    return cands


def _candidate_tessdata(cmd: Optional[str]) -> Optional[str]:
    """Return the ``tessdata`` directory to advertise via ``TESSDATA_PREFIX``.

    Tesseract 5.x wants this pointed straight at the ``tessdata`` folder (not
    its parent). We only ever return a directory that exists, so a system
    install -- whose data lives elsewhere and is located by the engine itself
    -- is correctly left untouched.
    """
    dirs: list[Path] = []
    if cmd:
        dirs.append(Path(cmd).parent / "tessdata")  # travels with the binary
    dirs.append(_app_root() / "tessdata")
    dirs.append(_managed_dir() / "tessdata")
    for d in dirs:
        if d.is_dir():
            return str(d)
    return None


def _classify_source(cmd: Optional[str], override: str) -> Optional[str]:
    """Label *how* the binary was found, for an informative status badge.

    One of: ``override`` (explicit setting/env), ``bundled`` (shipped in the
    app), ``managed`` (portable build in the app folder), ``path`` (found on
    PATH), or ``system`` (a well-known install location).
    """
    if not cmd:
        return None
    if override and cmd == override:
        return "override"
    if cmd == "tesseract":
        return "path"
    parent = os.path.dirname(cmd)
    if parent == str(_app_root() / "tesseract"):
        return "bundled"
    if parent == str(_managed_dir()):
        return "managed"
    return "system"


def _install_hint(platform: Optional[str] = None) -> str:
    """OS-appropriate one-line instruction for installing the engine."""
    plat = platform or sys.platform
    if plat.startswith("win"):
        return (
            "Install it with the UB-Mannheim installer "
            "(https://github.com/UB-Mannheim/tesseract/wiki)."
        )
    if plat == "darwin":
        return "Install it with: brew install tesseract"
    return "Install it with: sudo apt install tesseract-ocr"


def _override_cmd(settings=None) -> str:
    """The explicit binary path from the setting or ``TESSERACT_CMD`` env."""
    if settings is not None:
        return settings.resolved_tesseract_cmd()
    return os.environ.get("TESSERACT_CMD", "")


def resolve_tesseract(settings=None) -> Optional[str]:
    """Point pytesseract at the right binary + tessdata; return the command.

    Resolution order: (1) an explicit override from ``TESSERACT_CMD`` / the
    saved ``tesseract_cmd`` setting -- honored even if the file is missing so a
    later error can name the bad path; (2) a binary bundled with the app or
    dropped into the app-managed folder (both no-admin); (3) per-user then
    machine-wide install locations; (4) pytesseract's own PATH lookup, the
    universal fallback that works wherever ``tesseract`` is on PATH. Returns the
    resolved command, or ``None`` if the pytesseract wrapper itself is absent.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    override = _override_cmd(settings)

    cmd: Optional[str] = None
    if override:
        cmd = override
    else:
        for cand in _candidate_binaries():
            if _is_file(cand):
                cmd = cand
                break

    resolved = cmd or "tesseract"  # PATH fallback; deterministic on every call
    pytesseract.pytesseract.tesseract_cmd = resolved
    _point_loader_at_bundle(cmd)  # bundled engine: make its libs loadable

    if not os.environ.get("TESSDATA_PREFIX"):
        tessdata = _candidate_tessdata(cmd)
        if tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata
    return resolved


@dataclass
class TesseractStatus:
    """A precise, JSON-friendly account of whether Tesseract is usable.

    Splitting the two failure modes apart is the whole point: a missing Python
    wrapper and a missing engine binary need different fixes, and the old
    catch-all ``except`` could not tell them apart.
    """

    installed: bool = False        # wrapper present AND binary responded
    wrapper_present: bool = False  # ``import pytesseract`` succeeded
    binary_found: bool = False     # the engine answered a version probe
    cmd_path: Optional[str] = None  # the command we point pytesseract at
    source: Optional[str] = None   # how it was found: bundled/managed/system/path/override
    version: Optional[str] = None
    languages: list[str] = field(default_factory=list)
    error: Optional[str] = None
    install_hint: str = ""
    # Where a user without admin rights can drop a portable Tesseract build.
    managed_dir: str = ""


_PROBE_CACHE: dict[Optional[str], TesseractStatus] = {}


def status(settings=None, *, force: bool = False) -> TesseractStatus:
    """Resolve paths, then probe in stages and report exactly what's wrong.

    The version probe shells out to the engine, so its result is cached per
    resolved command path -- a batch job calling this once per page does not
    re-spawn tesseract every time. Pass ``force=True`` to re-probe (the status
    endpoint does, so the badge reflects a just-installed engine).
    """
    cmd = resolve_tesseract(settings)
    if not force and cmd in _PROBE_CACHE:
        return _PROBE_CACHE[cmd]

    st = TesseractStatus(
        cmd_path=cmd,
        source=_classify_source(cmd, _override_cmd(settings)),
        install_hint=_install_hint(),
        managed_dir=str(_managed_dir()),
    )
    try:
        import pytesseract
    except ImportError:
        st.error = (
            "The 'pytesseract' Python package is not installed. Reinstall "
            "CursBreaker (e.g. 'pip install .') to pull it in, then restart."
        )
        _PROBE_CACHE[cmd] = st
        return st

    st.wrapper_present = True
    try:
        st.version = str(pytesseract.get_tesseract_version())
        st.binary_found = True
    except Exception:
        where = f" (looked for '{cmd}')" if cmd else ""
        st.error = (
            f"Tesseract OCR engine not found{where}. {st.install_hint} "
            "No admin rights? Unzip a portable Tesseract into "
            f"'{st.managed_dir}' (a folder with the tesseract executable and a "
            "'tessdata' subfolder), or set the TESSERACT_CMD environment "
            "variable to the full path of the executable, then restart."
        )
        _PROBE_CACHE[cmd] = st
        return st

    try:
        st.languages = sorted(set(pytesseract.get_languages(config="")))
    except Exception:
        st.languages = []
    st.installed = True
    _PROBE_CACHE[cmd] = st
    return st


def is_available(settings=None) -> bool:
    """Return True iff the wrapper is importable AND the binary responds.

    Pass ``settings`` so a configured ``tesseract_cmd`` is honored, not just the
    environment override."""
    return status(settings).installed


def available_languages() -> list[str]:
    """Return the language codes the local install supports (e.g. ``["eng"]``)."""
    return status().languages


def transcribe_page(
    image: Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
) -> list[TranscribedLine]:
    """OCR the whole image; coordinates are already in page space."""
    return transcribe_region(image, lang=lang, psm=psm, offset=(0, 0))


def transcribe_region(
    image: Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
    offset: tuple[int, int] = (0, 0),
) -> list[TranscribedLine]:
    """OCR ``image`` (a possibly-cropped region) and return per-line results.

    ``offset`` is added to every returned coordinate so the caller can pass a
    crop of the page and still receive boxes in the original page's coordinate
    space — that is how word-box refinement maps a cropped line back to the page.
    """
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    return _data_to_lines(data, offset)


def _data_to_lines(
    data: dict, offset: tuple[int, int]
) -> list[TranscribedLine]:
    """Group word-level rows by (block, par, line) and build TranscribedLine."""
    ox, oy = offset
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    n = len(data["text"])
    for i in range(n):
        if data["level"][i] != _WORD_LEVEL:
            continue
        text = data["text"][i].strip()
        # pytesseract returns confidence as a string in some versions and -1
        # for non-text rows; coerce and drop blanks.
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            conf = -1
        if not text or conf < 0:
            continue
        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": conf,
                "left": int(data["left"][i]) + ox,
                "top": int(data["top"][i]) + oy,
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )

    lines: list[TranscribedLine] = []
    # Preserve detection order: sort by (block, par, line) so reading order
    # follows Tesseract's own page-layout analysis.
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        # Within a line, sort words left-to-right so the line text reads right.
        rows.sort(key=lambda r: r["left"])
        words = [
            OcrWord(
                text=r["text"],
                box=PixelBox(
                    x0=r["left"],
                    y0=r["top"],
                    x1=r["left"] + r["width"],
                    y1=r["top"] + r["height"],
                ),
                confidence=r["conf"],
            )
            for r in rows
        ]
        if not words:
            continue
        x0 = min(w.box.x0 for w in words)
        y0 = min(w.box.y0 for w in words)
        x1 = max(w.box.x1 for w in words)
        y1 = max(w.box.y1 for w in words)
        line_text = " ".join(w.text for w in words)
        avg_conf = round(sum(w.confidence for w in words) / len(words))
        lines.append(
            TranscribedLine(
                text=line_text,
                box=PixelBox(x0=x0, y0=y0, x1=x1, y1=y1),
                confidence=avg_conf,
                words=words,
            )
        )
    return lines


def require_available(settings=None) -> None:
    """Raise a clear, failure-specific error if Tesseract is not usable here.

    The message names the actual gap -- missing Python wrapper vs. missing or
    mislocated engine binary -- instead of a one-size-fits-all "install it".
    """
    st = status(settings)
    if st.installed:
        return
    raise RuntimeError(st.error or "Tesseract is not available.")
