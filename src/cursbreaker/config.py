"""Persistent user settings.

Settings live in a small JSON file in the per-user config directory. The Gemini
API key belongs to the *user* (never bundled), so it is stored locally with
owner-only permissions and can always be overridden by the ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY`` environment variables.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from platformdirs import user_config_dir
from pydantic import BaseModel

APP_NAME = "CursBreaker"


class Settings(BaseModel):
    # --- Gemini API ---
    api_key: str = ""
    # The UI offers a curated dropdown of these models (see pricing.CATALOG and
    # /api/models) so cost can be priced automatically. One picker drives both:
    # ``detection_model`` (used by the two-pass flow) is kept in sync with
    # ``transcription_model``.
    transcription_model: str = "gemini-3.1-pro-preview"
    detection_model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.3
    # Two ways to constrain reasoning: a coarse "thinking level" string (used
    # by some models) and a numeric token budget. When ``thinking_level`` is
    # set we send that; an empty string falls through to ``thinking_budget``.
    # Default ships with budget=128 active so the model spends minimal tokens
    # on reasoning (Humphries' finding: more thinking hurts handwriting).
    thinking_level: str = ""
    thinking_budget: int = 128
    media_resolution: str = "high"  # high | medium | low
    max_output_tokens: int = 8192

    # --- Pipeline ---
    # Content type chooses which engine(s) run per page:
    #   handwriting -> Gemini transcribes the whole page (printed + handwritten);
    #                  its transcription is always the authoritative text.
    #   text        -> Tesseract only; no API call.
    # ("mixed" was retired -- it let Tesseract's text degrade the output. Its
    # replacement is handwriting with ``refine_word_boxes`` on; see
    # ``normalize_content``.)
    content_type: str = "handwriting"  # handwriting | text
    # When True (and Tesseract is available), refine per-word *positions* on the
    # Gemini transcription using Tesseract's real word boxes -- adopted only
    # where Tesseract's text agrees with Gemini's. Tesseract text is never
    # emitted, so this improves word location without ever changing what was
    # transcribed. Off by default (adds local OCR work; needs Tesseract).
    refine_word_boxes: bool = False
    tesseract_language: str = "eng"  # any 3-letter code installed locally
    # Optional explicit path to the tesseract binary. "" = auto-detect (bundled
    # binary, well-known locations, then PATH). Overridable by the TESSERACT_CMD
    # environment variable, the same way the API key can come from the env.
    tesseract_cmd: str = ""
    mode: str = "two_pass"  # two_pass | one_pass (handwriting flow)
    pdf_dpi: int = 300
    max_dimension: int = 0  # 0 = keep original size; else resize longest side
    preprocess: bool = True

    # --- Output ---
    word_confidence: int = 95  # x_wconf for words on detected lines
    interpolated_confidence: int = 60  # x_wconf for words on interpolated lines
    language: str = "en"  # used for xml:lang and per-line "lang" in hOCR

    def public_dict(self) -> dict:
        """Settings safe to send to the browser (API key presence only)."""
        data = self.model_dump()
        data.pop("api_key", None)
        resolved = self.resolved_api_key()
        data["api_key_set"] = bool(resolved)
        data["api_key_hint"] = (
            f"••••{resolved[-4:]}" if len(resolved) >= 4
            else ("••••" if resolved else "")
        )
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            data["api_key_source"] = "env"
        elif self.api_key:
            data["api_key_source"] = "config"
        else:
            data["api_key_source"] = None
        if os.environ.get("TESSERACT_CMD"):
            data["tesseract_cmd_source"] = "env"
        elif self.tesseract_cmd:
            data["tesseract_cmd_source"] = "config"
        else:
            data["tesseract_cmd_source"] = None
        return data

    def resolved_api_key(self) -> str:
        return (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or self.api_key
        )

    def resolved_tesseract_cmd(self) -> str:
        """Path to the tesseract binary: env override first, then the setting."""
        return os.environ.get("TESSERACT_CMD") or self.tesseract_cmd

    def normalize_content(self) -> "Settings":
        """Migrate the retired 'mixed' content type in place.

        'Mixed' meant "Gemini text plus Tesseract"; that is now Handwriting with
        ``refine_word_boxes`` enabled. Idempotent; returns ``self`` for chaining
        so callers can ``load(...).normalize_content()``."""
        if self.content_type == "mixed":
            self.content_type = "handwriting"
            self.refine_word_boxes = True
        return self

    def sync_models(self) -> "Settings":
        """Single-model UX: one dropdown drives everything, so the two-pass
        detection step always uses the transcription model. Enforced here (not
        just in the browser) so a stale config or a direct API call can't leave
        detection running on a different model than the one priced/reported.
        Idempotent; returns ``self`` for chaining."""
        self.detection_model = self.transcription_model
        return self


def config_path() -> Path:
    override = os.environ.get("CURSBREAKER_CONFIG")
    if override:
        return Path(override)
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


def load_settings() -> Settings:
    path = config_path()
    if path.exists():
        try:
            return (
                Settings.model_validate_json(path.read_text("utf-8"))
                .normalize_content()
                .sync_models()
            )
        except Exception:
            # A corrupt config should never brick the app; fall back to defaults.
            return Settings()
    return Settings()


def save_settings(settings: Settings) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.model_dump(), indent=2), "utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: the file holds an API key
    except OSError:
        pass
