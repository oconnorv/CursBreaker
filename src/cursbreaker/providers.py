"""Provider registry: which transcription services the app can use.

CursBreaker sends page images to a hosted model and gets text (and, for the
box-producing calls, line boxes) back. Three services can do that job, and an
institution will often have approved exactly one of them, so the provider is a
user setting rather than a build-time choice:

* ``gemini``    -- Google Gemini (the original; the only one that can price an
                   estimate *and* count input tokens before a run)
* ``anthropic`` -- Anthropic Claude
* ``openai``    -- OpenAI

Everything provider-specific lives behind ``TranscriptionProvider`` (declared in
``gemini_client``): the pipeline, server and hOCR export never learn which
service produced a line. This module holds the parts that are genuinely shared
-- the metadata table, the error classification every SDK needs, and the
dispatch that turns a ``Settings`` into a live provider.

Error classification is shared deliberately. All three SDKs raise exceptions
carrying an HTTP status and a message, and the decisions we make from them
(is the key dead? is the model gone? is this worth retrying?) are identical, so
one conservative implementation serves all three -- and stays conservative in
the same way: a throttled or offline key is never reported as invalid.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class ProviderInfo:
    """Everything the app and UI need to know about one service.

    ``key_field`` is the ``Settings`` attribute holding that provider's key;
    ``env_vars`` are the environment variables that override it, in precedence
    order. ``counts_input_tokens`` records whether the service exposes a
    free token-counting endpoint -- Gemini and Claude do, so their pre-flight
    estimate measures the real image cost; OpenAI does not, so its estimate
    reports output tokens only and says so.
    """

    id: str
    label: str            # user-facing name, e.g. "Anthropic Claude"
    short_label: str      # for tight UI spots, e.g. "Claude"
    key_field: str
    env_vars: tuple[str, ...]
    console_url: str      # where a user creates a key
    pricing_url: str
    counts_input_tokens: bool = True
    # Providers whose selectable models come from the user's own key rather
    # than our curated price list (see pricing.CATALOG).
    lists_models_live: bool = False
    notes: str = ""


PROVIDERS: dict[str, ProviderInfo] = {
    "gemini": ProviderInfo(
        id="gemini",
        label="Google Gemini",
        short_label="Gemini",
        key_field="api_key",
        env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        console_url="https://aistudio.google.com/apikey",
        pricing_url="https://ai.google.dev/gemini-api/docs/pricing",
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        label="Anthropic Claude",
        short_label="Claude",
        key_field="anthropic_api_key",
        env_vars=("ANTHROPIC_API_KEY",),
        console_url="https://console.anthropic.com/settings/keys",
        pricing_url="https://claude.com/pricing#api",
    ),
    "openai": ProviderInfo(
        id="openai",
        label="OpenAI",
        short_label="OpenAI",
        key_field="openai_api_key",
        env_vars=("OPENAI_API_KEY",),
        console_url="https://platform.openai.com/api-keys",
        pricing_url="https://openai.com/api/pricing/",
        counts_input_tokens=False,
        lists_models_live=True,
        notes=(
            "Models are read from your own key, and OpenAI has no free "
            "token-counting endpoint, so cost is not estimated before a run."
        ),
    ),
}

DEFAULT_PROVIDER = "gemini"


def provider_info(provider: str | None) -> ProviderInfo:
    """Metadata for a provider id, falling back to the default for anything
    unrecognized (a config from a newer build, or a hand-edited file)."""
    return PROVIDERS.get((provider or "").lower(), PROVIDERS[DEFAULT_PROVIDER])


def provider_ids() -> list[str]:
    return list(PROVIDERS)


# --------------------------------------------------------------------------- #
# Error classification (shared by every client)
# --------------------------------------------------------------------------- #
# Substrings that mark a genuine authentication failure (bad/revoked/expired
# key), independent of SDK and provider.
_AUTH_MARKERS = (
    "API_KEY_INVALID", "API KEY NOT VALID", "API KEY EXPIRED",
    "PERMISSION_DENIED", "UNAUTHENTICATED", "UNAUTHORIZED",
    "INVALID AUTHENTICATION", "INVALID_API_KEY", "INCORRECT API KEY",
    "AUTHENTICATION_ERROR",
)

# Substrings marking a model that can't be called (retired/renamed/not granted
# for this key) -- as opposed to a key or network problem.
_MODEL_GONE_MARKERS = (
    "NOT_FOUND", "NO LONGER AVAILABLE", "IS NOT FOUND", "NOT SUPPORTED",
    "DOES NOT EXIST", "UNKNOWN MODEL", "NOT FOUND FOR API VERSION",
    "MODEL_NOT_FOUND", "DOES NOT HAVE ACCESS TO MODEL",
)

# Transient failures worth retrying with backoff: momentarily unavailable or
# overloaded, a deadline, or a network blip -- common with large, dense images.
_TRANSIENT_MARKERS = (
    "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED", "INTERNAL", "OVERLOADED",
    "TIMEOUT", "TIMED OUT", "TEMPORARILY", "CONNECTION", "RESET BY PEER",
    "RATE_LIMIT", "RATE LIMIT",
)


def _status_code(exc: Exception):
    """The HTTP status an SDK exception carries, whatever it calls it."""
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _blob(exc: Exception) -> str:
    return f"{getattr(exc, 'message', '')} {exc}".upper()


def is_auth_error(exc: Exception) -> bool:
    """True only when an exception clearly means a bad/revoked key.

    Deliberately conservative: a transient network error, a 5xx, or a 429
    rate-limit must NOT be classified as 'invalid', or we would tell a user
    their good key is dead."""
    if _status_code(exc) in (401, 403):
        return True
    return any(m in _blob(exc) for m in _AUTH_MARKERS)


def is_model_unavailable(exc: Exception) -> bool:
    """True when an error means the *model* is gone (vs. a key/network
    problem), so a caller can fall back instead of failing the whole job."""
    if _status_code(exc) == 404:
        return True
    return any(m in _blob(exc) for m in _MODEL_GONE_MARKERS)


def is_transient(exc: Exception) -> bool:
    """True for retryable service/timeout errors (503, deadline exceeded, 429,
    5xx, network blips) -- but never for auth or model-gone."""
    if is_auth_error(exc) or is_model_unavailable(exc):
        return False
    if _status_code(exc) in (408, 409, 429, 500, 502, 503, 504):
        return True
    blob = f"{getattr(exc, 'message', '')} {type(exc).__name__} {exc}".upper()
    return any(m in blob for m in _TRANSIENT_MARKERS)


def short_error(exc: Exception) -> str:
    """A compact one-line version of an SDK error (drops the JSON blob)."""
    code = _status_code(exc) or ""
    msg = str(getattr(exc, "message", "") or exc).split("{", 1)[0].strip()
    return f"{code} {msg}".strip()[:140] or "error"


# A dense scan can legitimately take minutes, and a one-off 503/deadline
# usually clears on a retry.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0          # seconds; doubles each attempt (2, 4, 8)
REQUEST_TIMEOUT_S = 300.0       # 5 minutes, for slow/large images
# Metadata calls (listing models) back an interactive control, so they get a
# short deadline: a hung list must not freeze the Settings panel behind the
# transcription timeout.
METADATA_TIMEOUT_S = 20.0


def transient_message(exc: Exception, label: str) -> str:
    """Actionable guidance when retries are exhausted on a transient error."""
    return (
        f"{label} was unavailable or timed out ({short_error(exc)}). This often "
        "happens with very large or dense images -- a high-resolution map or "
        "scan is a common cause. Try again; if it keeps happening, lower 'Max "
        "image dimension' in Advanced (e.g. 3000-4000) so each request is "
        "lighter. For a printed page you can also use 'Printed only' mode, "
        "which runs locally with no API call."
    )


def call_with_retries(fn, label: str):
    """Run ``fn`` with exponential backoff on transient failures.

    Shared by the Claude and OpenAI clients (Gemini has its own copy wired into
    its model-fallback path). A retried success keeps a one-off timeout from
    killing a long batch; anything non-transient is raised straight away."""
    delay = RETRY_BASE_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= MAX_RETRIES or not is_transient(exc):
                raise
            print(
                f"WARNING: {label} call failed ({short_error(exc)}); retrying "
                f"in {delay:.0f}s ({attempt + 1}/{MAX_RETRIES})…",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------- #
# Parsing model replies
# --------------------------------------------------------------------------- #
def strip_code_fence(text: str) -> str:
    """Drop a ```-fenced wrapper. Every provider is told not to use fences and
    every provider occasionally does anyway."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def lines_from_json_text(text: str):
    """Line boxes from a raw JSON reply: a bare array, or the ``{"lines": [...]}``
    envelope the object-only structured-output modes produce, fenced or not.

    The text path is a fallback behind each provider's schema-checked mode, so
    it stays forgiving -- a malformed element is skipped rather than costing the
    user the whole page."""
    from .models import LineBox

    text = strip_code_fence(text)
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("lines") or []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict):
            try:
                out.append(LineBox(**item))
            except Exception:  # noqa: BLE001 -- skip a malformed element
                continue
    return out


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
@dataclass
class KeyStatus:
    """Result of a cheap, generation-free check that an API key still works."""

    state: str          # valid | invalid | unknown | no_key
    message: str = ""
    provider: str = ""


def make_provider(settings: Settings):
    """Build the client for the configured provider.

    Raises ``RuntimeError`` with a provider-specific message when no key is
    set, which the server turns into a 400 the Settings panel can act on."""
    pid = provider_info(settings.provider).id
    if pid == "anthropic":
        from .anthropic_client import AnthropicProvider

        return AnthropicProvider(settings)
    if pid == "openai":
        from .openai_client import OpenAIProvider

        return OpenAIProvider(settings)
    from .gemini_client import GeminiProvider

    return GeminiProvider(settings)


def check_api_key(settings: Settings) -> KeyStatus:
    """Verify the active provider's stored key without spending generation
    quota, so a revoked or mistyped key surfaces in Settings rather than
    mid-transcription. Every provider exposes a free "list models" call, which
    is what each client's probe uses."""
    info = provider_info(settings.provider)
    if not settings.resolved_api_key():
        return KeyStatus("no_key", "No API key is stored.", info.id)

    if info.id == "anthropic":
        from .anthropic_client import probe_key
    elif info.id == "openai":
        from .openai_client import probe_key
    else:
        from .gemini_client import probe_key

    try:
        probe_key(settings.resolved_api_key())
        return KeyStatus("valid", "Key verified -- it's active.", info.id)
    except Exception as exc:  # noqa: BLE001 -- classify, never propagate
        if is_auth_error(exc):
            return KeyStatus(
                "invalid",
                f"This {info.short_label} key was rejected -- it may have been "
                "revoked, expired, or mistyped. Paste a current key.",
                info.id,
            )
        return KeyStatus(
            "unknown",
            "Couldn't verify the key right now (a network or service issue); "
            "it may still be fine.",
            info.id,
        )
