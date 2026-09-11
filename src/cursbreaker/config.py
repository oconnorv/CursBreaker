"""Persistent user settings.

Settings live in a small JSON file in the per-user config directory. API keys
belong to the *user* (never bundled), so they are stored locally with
owner-only permissions and can always be overridden by environment variables.

One key is stored per provider, not one key overall: a user may hold keys for
several services, and switching provider must not make them re-paste anything.
The provider table (``providers.PROVIDERS``) owns which field and which
environment variables back each one; this module only reads that table, so
adding a provider never means editing key handling here.
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
    # --- Provider ---
    # Which hosted service transcribes: gemini | anthropic | openai. Users pick
    # whichever their institution has approved; the pipeline is identical.
    provider: str = "gemini"

    # --- API keys (one per provider; see providers.PROVIDERS) ---
    # ``api_key`` is Gemini's and keeps its original name so existing config
    # files -- written before other providers existed -- keep working untouched.
    api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
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
        """Settings safe to send to the browser (key *presence* only, never a
        key itself).

        ``keys`` carries one entry per provider so the UI can show at a glance
        which services are ready; the flat ``api_key_*`` fields describe the
        *active* provider and predate the others, so they stay for older
        clients and for the tests that assert on them."""
        from .providers import PROVIDERS, provider_info

        data = self.model_dump()
        for info in PROVIDERS.values():
            data.pop(info.key_field, None)

        data["keys"] = {
            pid: self._key_state(info) for pid, info in PROVIDERS.items()
        }
        active = data["keys"][provider_info(self.provider).id]
        data["api_key_set"] = active["set"]
        data["api_key_hint"] = active["hint"]
        data["api_key_source"] = active["source"]
        if os.environ.get("TESSERACT_CMD"):
            data["tesseract_cmd_source"] = "env"
        elif self.tesseract_cmd:
            data["tesseract_cmd_source"] = "config"
        else:
            data["tesseract_cmd_source"] = None
        return data

    def _key_state(self, info) -> dict:
        """Presence, masked hint and origin for one provider's key."""
        resolved = self.resolved_api_key(info.id)
        if any(os.environ.get(v) for v in info.env_vars):
            source = "env"
        elif getattr(self, info.key_field, ""):
            source = "config"
        else:
            source = None
        return {
            "set": bool(resolved),
            "hint": (
                f"••••{resolved[-4:]}" if len(resolved) >= 4
                else ("••••" if resolved else "")
            ),
            "source": source,
        }

    def resolved_api_key(self, provider: str | None = None) -> str:
        """The key for ``provider`` (default: the active one), environment
        first. An env var always wins so a site can provision a key centrally
        without touching each user's config file."""
        from .providers import provider_info

        info = provider_info(provider or self.provider)
        for var in info.env_vars:
            value = os.environ.get(var)
            if value:
                return value
        return getattr(self, info.key_field, "") or ""

    def resolved_tesseract_cmd(self) -> str:
        """Path to the tesseract binary: env override first, then the setting."""
        return os.environ.get("TESSERACT_CMD") or self.tesseract_cmd

    def normalize_content(self) -> "Settings":
        """Migrate the retired 'mixed' content type in place.

        'Mixed' meant "Gemini text plus Tesseract"; that is now Handwriting with
        ``refine_word_boxes`` enabled. Idempotent; returns ``self`` for chaining
        so callers can ``load(...).normalize_content()``."""
        from .providers import provider_info

        if self.content_type == "mixed":
            self.content_type = "handwriting"
            self.refine_word_boxes = True
        # An unknown provider (hand-edited config, or one from a newer build)
        # falls back to the default rather than crashing every call.
        self.provider = provider_info(self.provider).id
        return self

    def sync_models(self) -> "Settings":
        """Single-model UX: one dropdown drives everything, so the two-pass
        detection step always uses the transcription model. Enforced here (not
        just in the browser) so a stale config or a direct API call can't leave
        detection running on a different model than the one priced/reported.

        Also keeps the model and the provider consistent: switching provider
        leaves the previous provider's model id behind, which would 404 on the
        first call, so a model that doesn't belong to the active provider is
        replaced -- with its named successor if it is a retired model, else
        with that provider's default. Idempotent; returns ``self``."""
        from .pricing import default_model_for, owns_model, replacement_for

        if not owns_model(self.provider, self.transcription_model):
            # A retired model moves to its named successor where there is one,
            # so a deliberate choice of the cheap tier isn't silently upgraded
            # to the flagship's price.
            successor = replacement_for(self.transcription_model)
            self.transcription_model = (
                successor if owns_model(self.provider, successor)
                else default_model_for(self.provider)
            )
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
