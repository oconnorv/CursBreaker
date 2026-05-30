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
    # Model names change often; these are sensible defaults but the UI lets the
    # user pick any model their key exposes (see /api/models).
    transcription_model: str = "gemini-3-pro-preview"
    detection_model: str = "gemini-3-pro-preview"
    temperature: float = 0.0
    # The reference blog found that *minimal* reasoning gives the best
    # handwriting accuracy. On Gemini 3 this maps to a low "thinking level";
    # older models use a token budget. We send whichever the SDK accepts.
    thinking_level: str = "low"
    thinking_budget: int = 128
    media_resolution: str = "high"  # high | medium | low
    max_output_tokens: int = 8192

    # --- Pipeline ---
    # Content type chooses which engine(s) run per page:
    #   handwriting -> Gemini only (existing recipe).
    #   text        -> Tesseract only; no API call.
    #   mixed       -> Gemini classifies each line printed/handwritten, then
    #                  printed lines go to Tesseract (real per-word boxes) and
    #                  handwritten lines use Gemini transcription. Outputs are
    #                  merged in our code -- Gemini never sees Tesseract's text.
    content_type: str = "handwriting"  # handwriting | text | mixed
    tesseract_language: str = "eng"  # any 3-letter code installed locally
    mode: str = "two_pass"  # two_pass | one_pass (handwriting flow)
    pdf_dpi: int = 300
    max_dimension: int = 0  # 0 = keep original size; else resize longest side
    preprocess: bool = True
    use_mock: bool = False  # exercise the full app without a real API key

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
        return data

    def resolved_api_key(self) -> str:
        return (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or self.api_key
        )


def config_path() -> Path:
    override = os.environ.get("CURSBREAKER_CONFIG")
    if override:
        return Path(override)
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


def load_settings() -> Settings:
    path = config_path()
    if path.exists():
        try:
            return Settings.model_validate_json(path.read_text("utf-8"))
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
