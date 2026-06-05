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
import sys
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
from .gemini_client import make_provider
from .hocr import XHTML_NS
from .images import SUPPORTED_EXT, count_content_pages, is_supported
from .pipeline import estimate_usage, process_batch
from .pricing import (
    CATALOG,
    PRICES_AS_OF,
    PRICING_URL,
    cost_for,
    effective_rates,
    pricing_for,
)

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
    settings.normalize_content()  # migrate a posted legacy "mixed" value
    settings.sync_models()  # detection (two-pass) always follows the chosen model
    save_settings(settings)
    return settings.public_dict()


@app.delete("/api/settings/api_key")
def clear_api_key():
    settings = load_settings()
    settings.api_key = ""
    save_settings(settings)
    return settings.public_dict()


@app.get("/api/tesseract")
def tesseract_status():
    """Detailed Tesseract status: which piece (if any) is missing and where we
    looked, so the UI can offer a fix specific to the actual failure."""
    from . import tesseract_client

    st = tesseract_client.status(load_settings(), force=True)
    return {
        "available": st.installed,  # kept for back-compat with older clients
        "languages": st.languages,
        "wrapper_present": st.wrapper_present,
        "binary_found": st.binary_found,
        "cmd_path": st.cmd_path,
        "source": st.source,
        "version": st.version,
        "error": st.error,
        "install_hint": st.install_hint,
        "managed_dir": st.managed_dir,
    }


@app.get("/api/key-status")
def key_status():
    """Cheap, generation-free check that the stored Gemini key still works, so a
    revoked/expired key surfaces in Settings before a transcription fails. Uses
    the free ListModels endpoint (no token/quota cost)."""
    from .gemini_client import check_api_key

    st = check_api_key(load_settings())
    return {"state": st.state, "message": st.message}


@app.get("/api/models")
def list_models():
    """The curated model catalog with published prices, for the dropdown +
    automatic cost estimate. A fixed list (not the key's live ListModels) so the
    selectable models and their prices stay in lockstep."""
    return {
        "models": [
            {
                "id": m.model,
                "label": m.label,
                "input_per_mtok": m.input_per_mtok,
                "output_per_mtok": m.output_per_mtok,
                "tier_threshold": m.tier_threshold,
                "input_per_mtok_high": m.input_per_mtok_high,
                "output_per_mtok_high": m.output_per_mtok_high,
            }
            for m in CATALOG
        ],
        "prices_as_of": PRICES_AS_OF,
        "pricing_url": PRICING_URL,
    }


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


class EstimateRequest(BaseModel):
    file_ids: list[str]
    mode: str | None = None


@app.post("/api/estimate")
def estimate(req: EstimateRequest):
    """Free pre-flight estimate of the tokens, and the rough dollars, a run will
    cost -- before any transcription happens. Uses Gemini's no-charge
    ``count_tokens`` for input; output is a labelled assumption, and the cost is
    derived automatically from the selected model's published price."""
    paths = [STAGED[i] for i in req.file_ids if i in STAGED]
    if not paths:
        raise HTTPException(400, "No staged files to estimate.")

    settings = load_settings()
    if req.mode in ("one_pass", "two_pass"):
        settings.mode = req.mode

    content = (settings.content_type or "handwriting").lower()
    # Printed-only runs locally (Tesseract) with no Gemini call -> no token cost.
    if content == "text":
        return {
            "billable": False,
            "reason": "Printed-only mode",
            "files": len(paths),
            "input": 0,
            "output": 0,
            "total": 0,
            "calls": 0,
            "cost": None,
        }

    if not settings.resolved_api_key():
        raise HTTPException(
            400, "No Gemini API key set. Add one in Settings to estimate cost."
        )

    try:
        provider = make_provider(settings)
        data = estimate_usage(paths, provider, settings)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't estimate right now: {exc}")
    data["billable"] = True
    return data


@app.post("/api/process")
def process(req: ProcessRequest):
    paths = [STAGED[i] for i in req.file_ids if i in STAGED]
    if not paths:
        raise HTTPException(400, "No staged files to process.")

    settings = load_settings()
    if req.mode in ("one_pass", "two_pass"):
        settings.mode = req.mode
    if not settings.resolved_api_key():
        raise HTTPException(400, "No Gemini API key set. Add one in Settings.")

    job_id = uuid.uuid4().hex
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # The model is captured at job start so the live cost figure stays anchored
    # to the model actually used, even if the user switches models mid-run.
    model = settings.transcription_model
    JOBS[job_id] = {
        "status": "running",
        "done": 0,                 # legacy file counters (kept for back-compat)
        "total": len(paths),
        "current": "",             # latest step message
        "stage": "",               # latest step's machine tag
        "log": [],                 # append-only activity log (capped)
        "done_units": 0,           # pages completed across the whole job (bar)
        "total_units": 0,          # total pages across the whole job (bar)
        "results": [],
        "out_dir": str(out_dir),
        "error": None,
        "model": model,
        "tokens": _usage_to_dict(None, model),  # zeros until the provider exists
        "_cancel": False,          # private flag flipped by /api/jobs/{id}/cancel
    }
    threading.Thread(
        target=_run_job, args=(job_id, paths, settings, out_dir), daemon=True
    ).start()
    return {"job_id": job_id}


_LOG_CAP = 500  # keep the most recent N activity-log lines (bounds memory + JSON)


def _append_capped(log: list, message: str, cap: int = _LOG_CAP) -> None:
    """Append a line, trimming to the most recent ``cap`` entries."""
    log.append(message)
    if len(log) > cap:
        del log[: len(log) - cap]


def _run_job(job_id, paths, settings, out_dir):
    job = JOBS[job_id]
    try:
        provider = make_provider(settings)
        # Expose the provider so a status poll can read its running token total
        # live (per page, as each call returns), not just at file boundaries.
        job["_provider"] = provider

        # Page count up front (cheap; same function /api/upload uses) so the bar
        # is page-driven and actually fills. count_content_pages already returns
        # 1 on any read error, so the sum is always >= the file count.
        total_units = sum(count_content_pages(p) for p in paths)
        job["total_units"] = total_units

        # The worker thread writes these scalar keys and appends to job["log"];
        # the request thread reads them in _public_job. Simple int/str writes and
        # list.append are atomic under the GIL, and no keys are added after the
        # job dict is created, so no lock is needed.
        def report(ev):
            job["current"] = ev.message
            job["stage"] = ev.stage
            job["done_units"] = ev.units_done
            job["total_units"] = ev.units_total or job["total_units"]
            job["done"], job["total"] = ev.file_index, ev.file_total
            _append_capped(job["log"], ev.message)

        results = process_batch(
            paths, provider, settings, out_dir, report, units_total=total_units,
            should_cancel=lambda: bool(job.get("_cancel")),
        )
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
                "tokens": _usage_to_dict(r.token_usage, job["model"]),
            }
            for r in results
        ]
        job["tokens"] = _usage_to_dict(provider.usage, job["model"])
        # Completed files (if any) are kept and downloadable even on cancel.
        job["status"] = "cancelled" if job.get("_cancel") else "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def _usage_to_dict(usage, model=None) -> dict:
    """Serialize a ``TokenUsage`` (or None -> zeros) for the browser. The dollar
    cost is derived automatically from the selected model's published price; it
    is ``None`` when the model isn't in the catalog (so no dollars are implied),
    and the (tier-aware) rates used are echoed back for transparency."""
    if usage is None:
        d = {"input": 0, "output": 0, "thinking": 0, "total": 0, "calls": 0}
    else:
        d = {
            "input": usage.input,
            "output": usage.output,
            "thinking": usage.thinking,
            "total": usage.total,
            "calls": usage.calls,
        }
    pricing = pricing_for(model)
    if pricing:
        if usage is not None:
            in_rate, out_rate = effective_rates(pricing, usage)
            d["cost"] = cost_for(pricing, usage)
        else:
            in_rate, out_rate = pricing.input_per_mtok, pricing.output_per_mtok
            d["cost"] = 0.0
        d["model"] = pricing.model
        d["model_label"] = pricing.label
        d["price_input_per_mtok"] = in_rate
        d["price_output_per_mtok"] = out_rate
        d["prices_as_of"] = PRICES_AS_OF
    else:
        d["price_input_per_mtok"] = 0.0
        d["price_output_per_mtok"] = 0.0
        d["cost"] = None
    return d


def _public_job(job: dict) -> dict:
    """A copy of a job's state safe to send to the browser: private keys (the
    provider handle) dropped, and the token counter refreshed live from the
    provider so it ticks up while the job runs."""
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    # The log list is shared by reference with the worker thread, which may keep
    # appending after this returns and before Starlette serializes; copy it so a
    # stable snapshot is sent.
    if isinstance(out.get("log"), list):
        out["log"] = list(out["log"])
    provider = job.get("_provider")
    usage = getattr(provider, "usage", None) if provider is not None else None
    if usage is not None:
        out["tokens"] = _usage_to_dict(usage, job.get("model"))
    return out


def _url(job_id, name):
    return f"/api/download/{job_id}/{name}" if name else None


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return _public_job(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Request cooperative cancellation of a running job. The worker stops at the
    next page/file boundary (an in-flight Gemini call can't be interrupted), so
    the status flips to ``cancelled`` shortly after; files already finished stay
    downloadable. A no-op on a job that's already finished."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job["status"] == "running" and not job.get("_cancel"):
        # Acknowledge in the activity log immediately (before the worker reaches
        # a cancel boundary) and explain why it isn't instant. "page" stage means
        # a Gemini/Tesseract call is in flight and can't be interrupted.
        if job.get("stage") == "page":
            msg = ("Cancellation requested — the current page is already being "
                   "processed by Gemini and can't be interrupted, so this will "
                   "take effect once the current step finishes.")
        else:
            msg = "Cancellation requested — stopping after the current step finishes."
        _append_capped(job["log"], msg)
        job["current"] = msg
        job["_cancel"] = True  # set last, so the message is logged before the worker stops
    return {"status": job["status"], "cancelling": bool(job.get("_cancel"))}


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


import logging as _logging


class PrettyAccessFormatter(_logging.Formatter):
    """Drop-in replacement for uvicorn's access-log formatter that styles each
    line by HTTP status class, so successful requests don't visually read like
    warnings:

      * 2xx, 3xx -> green, prefixed with ``ok``
      * 4xx       -> yellow, prefixed with ``warn``
      * 5xx       -> red, prefixed with ``err``

    Colors auto-detect a TTY; the standard ``NO_COLOR`` env var disables them.
    Non-access log records (anything that isn't a 5-arg access tuple) fall
    through to the default Formatter so we never break unrelated log lines.
    """

    _RESET = "\x1b[0m"
    _GREEN = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED = "\x1b[31m"

    def __init__(self, *, use_colors=None):
        super().__init__()
        if use_colors is None:
            use_colors = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.use_colors = bool(use_colors)

    def format(self, record):  # noqa: A003 (logging API name)
        args = record.args or ()
        try:
            client_addr, method, full_path, http_version, status_code = args
            status = int(status_code)
        except (ValueError, TypeError):
            return super().format(record)

        if 200 <= status < 400:
            color, marker = self._GREEN, " ok "
        elif 400 <= status < 500:
            color, marker = self._YELLOW, "warn"
        else:
            color, marker = self._RED, " err"

        try:
            from http import HTTPStatus

            phrase = HTTPStatus(status).phrase
        except ValueError:
            phrase = ""

        line = (
            f'{marker}     {client_addr} - '
            f'"{method} {full_path} HTTP/{http_version}" '
            f'{status} {phrase}'
        ).rstrip()
        if self.use_colors:
            return f"{color}{line}{self._RESET}"
        return line


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
