"""FastAPI app for the local CursBreaker GUI.

Everything runs on the user's machine and binds to localhost. Uploaded files are
copied into a temp staging area; each processing run writes its outputs to a
per-job temp directory, which is offered back as individual downloads or a zip.
The user's original files are never modified.
"""

from __future__ import annotations

import io
import os
import signal
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from lxml import etree
from PIL import Image, ImageDraw
from pydantic import BaseModel

from . import __version__
from .config import load_settings, save_settings
from .gemini_client import SUGGESTED_MODELS, make_provider
from .hocr import XHTML_NS
from .images import SUPPORTED_EXT, count_content_pages, is_supported
from .pipeline import process_batch

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="CursBreaker", version=__version__)

# In-memory state for this single-user local session.
_BASE = Path(tempfile.mkdtemp(prefix="cursbreaker_"))
STAGE_DIR = _BASE / "stage"
JOBS_DIR = _BASE / "jobs"
STAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

STAGED: dict[str, Path] = {}
JOBS: dict[str, dict] = {}

# Browser heartbeat / auto-shutdown state. Tests don't start the watchdog; it is
# kicked off explicitly from __main__ when the CLI launches the server.
_LAST_PING_AT: float | None = None
_AUTOSHUTDOWN_STARTED = False


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@app.get("/api/settings")
def get_settings():
    return load_settings().public_dict()


@app.post("/api/settings")
async def update_settings(payload: dict):
    settings = load_settings()
    fields = type(settings).model_fields
    for key, value in payload.items():
        if key == "api_key":
            if value:  # never blank out a saved key on a no-op save
                settings.api_key = value
        elif key in fields:
            setattr(settings, key, value)
    save_settings(settings)
    return settings.public_dict()


@app.delete("/api/settings/api_key")
def clear_api_key():
    settings = load_settings()
    settings.api_key = ""
    save_settings(settings)
    return settings.public_dict()


@app.get("/api/models")
def list_models():
    settings = load_settings()
    if not settings.use_mock and not settings.resolved_api_key():
        return {"models": SUGGESTED_MODELS, "suggested": SUGGESTED_MODELS, "note": "no_key"}
    try:
        models = make_provider(settings).list_models()
        return {"models": models, "suggested": SUGGESTED_MODELS}
    except Exception as exc:
        return {"models": SUGGESTED_MODELS, "suggested": SUGGESTED_MODELS, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Upload / process / status
# --------------------------------------------------------------------------- #
def _page_count(path: Path) -> int:
    return count_content_pages(path)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    staged = []
    for f in files:
        if not is_supported(f.filename or ""):
            continue
        file_id = uuid.uuid4().hex
        # Stage in a per-upload subdir so the UUID never leaks into output names.
        sub = STAGE_DIR / file_id
        sub.mkdir(parents=True, exist_ok=True)
        dest = sub / Path(f.filename).name
        dest.write_bytes(await f.read())
        STAGED[file_id] = dest
        staged.append(
            {"id": file_id, "name": Path(f.filename).name, "pages": _page_count(dest)}
        )
    if not staged:
        raise HTTPException(400, f"No supported files. Allowed: {sorted(SUPPORTED_EXT)}")
    return {"files": staged}


class ProcessRequest(BaseModel):
    file_ids: list[str]
    mode: str | None = None
    use_mock: bool | None = None


@app.post("/api/process")
def process(req: ProcessRequest):
    paths = [STAGED[i] for i in req.file_ids if i in STAGED]
    if not paths:
        raise HTTPException(400, "No staged files to process.")

    settings = load_settings()
    if req.mode in ("one_pass", "two_pass"):
        settings.mode = req.mode
    if req.use_mock is not None:
        settings.use_mock = req.use_mock
    if not settings.use_mock and not settings.resolved_api_key():
        raise HTTPException(400, "No Gemini API key set. Add one in Settings or enable mock mode.")

    job_id = uuid.uuid4().hex
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {
        "status": "running",
        "done": 0,
        "total": len(paths),
        "current": "",
        "results": [],
        "out_dir": str(out_dir),
        "error": None,
    }
    threading.Thread(
        target=_run_job, args=(job_id, paths, settings, out_dir), daemon=True
    ).start()
    return {"job_id": job_id}


def _run_job(job_id, paths, settings, out_dir):
    job = JOBS[job_id]
    try:
        provider = make_provider(settings)

        def cb(done, total, name):
            job["done"], job["total"], job["current"] = done, total, name

        results = process_batch(paths, provider, settings, out_dir, cb)
        job["results"] = [
            {
                "source_name": r.source_name,
                "n_pages": r.n_pages,
                "n_lines": r.n_lines,
                "txt": _url(job_id, r.txt_name),
                "hocr": _url(job_id, r.hocr_name),
                "pdf": _url(job_id, r.pdf_name),
                "images": [
                    {"name": n, "download": _url(job_id, n), "preview": f"/api/preview/{job_id}/{n}"}
                    for n in r.image_names
                ],
                "error": r.error,
            }
            for r in results
        ]
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def _url(job_id, name):
    return f"/api/download/{job_id}/{name}" if name else None


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return job


# --------------------------------------------------------------------------- #
# Downloads / preview
# --------------------------------------------------------------------------- #
def _safe_output(job_id: str, name: str) -> Path:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Invalid file name.")
    path = Path(job["out_dir"]) / name
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/download/{job_id}/{name}")
def download(job_id: str, name: str):
    return FileResponse(_safe_output(job_id, name), filename=name)


@app.get("/api/download/{job_id}.zip")
def download_zip(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    out_dir = Path(job["out_dir"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="cursbreaker_{job_id[:8]}.zip"'},
    )


@app.get("/api/preview/{job_id}/{name}")
def preview(job_id: str, name: str):
    """Draw the hOCR line boxes over the page image so localization is visible."""
    img_path = _safe_output(job_id, name)
    out_dir = img_path.parent
    boxes = _line_boxes_for_image(out_dir, name)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im, "RGBA")
        for (x0, y0, x1, y1) in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 90, 255), width=3)
            draw.rectangle([x0, y0, x1, y1], fill=(220, 30, 90, 28))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _line_boxes_for_image(out_dir: Path, image_name: str) -> list[tuple[int, int, int, int]]:
    ns = {"x": XHTML_NS}
    for hocr in out_dir.glob("*.hocr"):
        root = etree.fromstring(hocr.read_bytes())
        for page in root.xpath("//x:div[@class='ocr_page']", namespaces=ns):
            if f'image "{image_name}"' not in (page.get("title") or ""):
                continue
            boxes = []
            for line in page.xpath(".//x:span[@class='ocr_line']", namespaces=ns):
                title = line.get("title") or ""
                for part in title.split(";"):
                    part = part.strip()
                    if part.startswith("bbox "):
                        x0, y0, x1, y1 = (int(v) for v in part[5:].split()[:4])
                        boxes.append((x0, y0, x1, y1))
            return boxes
    return []


# --------------------------------------------------------------------------- #
# Browser heartbeat / auto-shutdown
# --------------------------------------------------------------------------- #
@app.post("/api/heartbeat")
def heartbeat(bye: bool = False):
    """The page pings every few seconds while it's open. ``bye=true`` is sent
    via ``navigator.sendBeacon`` on tab close, which pulls the last-seen time
    back so the watchdog fires shortly (unless another tab is still pinging)."""
    global _LAST_PING_AT
    now = time.time()
    if bye:
        _LAST_PING_AT = now - max(0.0, _SHUTDOWN_GRACE - 3.0)
    else:
        _LAST_PING_AT = now
    return {"ok": True}


_SHUTDOWN_GRACE = 15.0  # seconds without a ping before we quit
_SHUTDOWN_POLL = 2.0  # how often the watchdog checks


def _any_jobs_running() -> bool:
    return any(j.get("status") == "running" for j in JOBS.values())


def _should_shutdown(
    last_ping: float | None,
    grace: float,
    *,
    now: float | None = None,
    jobs_running: bool = False,
) -> bool:
    if jobs_running or last_ping is None:
        return False
    return ((time.time() if now is None else now) - last_ping) > grace


def _quit_process() -> None:
    """Polite shutdown first (uvicorn handles SIGINT gracefully); hard-exit as
    a backstop if uvicorn doesn't pick it up promptly."""
    try:
        signal.raise_signal(signal.SIGINT)
    except Exception:
        pass
    threading.Timer(2.5, lambda: os._exit(0)).start()


def start_autoshutdown(
    grace_seconds: float = _SHUTDOWN_GRACE, poll_seconds: float = _SHUTDOWN_POLL
) -> None:
    """Begin the watchdog thread. Safe to call multiple times (no-op after the
    first call)."""
    global _LAST_PING_AT, _AUTOSHUTDOWN_STARTED, _SHUTDOWN_GRACE, _SHUTDOWN_POLL
    if _AUTOSHUTDOWN_STARTED:
        return
    _AUTOSHUTDOWN_STARTED = True
    _SHUTDOWN_GRACE = grace_seconds
    _SHUTDOWN_POLL = poll_seconds
    _LAST_PING_AT = time.time()  # initial grace until the first browser ping

    def loop():
        while True:
            time.sleep(poll_seconds)
            if _should_shutdown(
                _LAST_PING_AT, grace_seconds, jobs_running=_any_jobs_running()
            ):
                _quit_process()
                return

    threading.Thread(target=loop, name="cb-autoshutdown", daemon=True).start()


def install_access_log_filter() -> None:
    """Drop heartbeat hits from uvicorn's access log so the CLI stays quiet
    while the browser is just keeping the server alive."""
    import logging

    class _HideHeartbeat(logging.Filter):
        _cursbreaker_heartbeat = True

        def filter(self, record):  # noqa: A003 (logging API name)
            try:
                for arg in (record.args or ()):
                    if isinstance(arg, str) and "/api/heartbeat" in arg:
                        return False
                if "/api/heartbeat" in record.getMessage():
                    return False
            except Exception:
                pass
            return True

    logger = logging.getLogger("uvicorn.access")
    if not any(getattr(f, "_cursbreaker_heartbeat", False) for f in logger.filters):
        logger.addFilter(_HideHeartbeat())


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/favicon.ico")
def favicon():
    # Drop a favicon.ico into src/cursbreaker/static/ to use a custom one;
    # otherwise reply 204 so browsers stop asking and we don't log a 404.
    f = STATIC_DIR / "favicon.ico"
    if f.is_file():
        return FileResponse(f, media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text("utf-8")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
