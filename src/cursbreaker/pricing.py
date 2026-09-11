"""Curated transcription models and their published prices.

The app offers a fixed dropdown of models (rather than free-text entry) so the
cost estimate can be computed *automatically* from each model's published
per-million-token price -- the user never types a price. Prices are
point-in-time and do change, so ``PRICES_AS_OF`` is shown in the UI next to a
link to live pricing, and every dollar figure stays labelled an estimate, never
a guarantee.

Each entry names the provider it belongs to, so switching provider swaps the
dropdown and the prices together.

To refresh prices: edit the numbers below and bump ``PRICES_AS_OF``.

An entry can also declare a price change that has already been *announced*
(``scheduled_from`` and the ``*_from`` rates). ``current_rates`` picks whichever
price is in force on the day, so an introductory rate stops being quoted the
moment it expires instead of waiting for someone to notice.

A model with no entry here still runs -- ``pricing_for`` returns ``None`` and the
UI reports token counts without a dollar figure -- so a stale saved model
degrades to "no published price" rather than to a wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Bump whenever the numbers below change; surfaced in the UI for transparency.
PRICES_AS_OF = "2026-09-11"

# Per-provider "where these numbers come from" links, shown beside the estimate.
PRICING_URLS = {
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
    "anthropic": "https://claude.com/pricing#api",
    "openai": "https://openai.com/api/pricing/",
}
# Kept for older callers that assume a single provider.
PRICING_URL = PRICING_URLS["gemini"]


@dataclass(frozen=True)
class ModelPricing:
    """A selectable model and its USD price per million tokens.

    Some models are *tiered*: the rate depends on how large a single request's
    prompt is. ``tier_threshold`` (in input tokens) marks the boundary; the
    ``*_high`` rates apply to a request whose prompt exceeds it. A threshold of
    0 means flat pricing and the ``*_high`` fields are unused.

    A model can also carry an *already-announced* future price. Introductory
    rates expire on a published date, and an app that kept quoting the intro
    rate afterwards would halve every estimate until someone noticed and
    edited this file. ``scheduled_from`` (ISO date) plus ``*_from`` rates let
    the entry describe both prices at once; ``current_rates`` picks whichever
    applies today, with no code change on the day it flips. Scheduled changes
    apply to the base rates only -- see ``current_rates``.
    """

    model: str
    label: str
    input_per_mtok: float
    output_per_mtok: float
    tier_threshold: int = 0
    input_per_mtok_high: float = 0.0
    output_per_mtok_high: float = 0.0
    provider: str = "gemini"
    # An announced future price. Empty date = the rates above are the only ones.
    scheduled_from: str = ""
    input_per_mtok_from: float = 0.0
    output_per_mtok_from: float = 0.0


# The dropdown, in display order, grouped by provider. The first entry for a
# provider is its default model: the most accurate for handwriting, with
# lighter/cheaper options after it.
CATALOG: list[ModelPricing] = [
    # --- Google Gemini -----------------------------------------------------
    ModelPricing(
        "gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)",
        input_per_mtok=2.00, output_per_mtok=12.00,
        tier_threshold=200_000,
        input_per_mtok_high=4.00, output_per_mtok_high=18.00,
        provider="gemini",
    ),
    # $0.75/$3.75 is an introductory rate that expires: Google has it doubling
    # on 2027-01-01. Both prices are declared here, so estimates switch to the
    # standard rate on the day by themselves.
    ModelPricing(
        "gemini-3.8-flash", "Gemini 3.8 Flash",
        input_per_mtok=0.75, output_per_mtok=3.75,
        provider="gemini",
        scheduled_from="2027-01-01",
        input_per_mtok_from=1.50, output_per_mtok_from=7.50,
    ),
    # --- Anthropic Claude --------------------------------------------------
    # Prices verified against claude.com/pricing#api on PRICES_AS_OF. Claude
    # bills a flat rate across its full context window, so no tiering.
    ModelPricing(
        "claude-opus-5", "Claude Opus 5",
        input_per_mtok=5.00, output_per_mtok=25.00,
        provider="anthropic",
    ),
    ModelPricing(
        "claude-sonnet-5", "Claude Sonnet 5",
        input_per_mtok=2.00, output_per_mtok=10.00,
        provider="anthropic",
    ),
    ModelPricing(
        "claude-haiku-4-5", "Claude Haiku 4.5",
        input_per_mtok=1.00, output_per_mtok=5.00,
        provider="anthropic",
    ),
    # --- OpenAI ------------------------------------------------------------
    # Rates recorded on PRICES_AS_OF; OpenAI's own pricing page is the live
    # source (PRICING_URLS below), and like every other figure here they are
    # point-in-time. Flat-rate across the context window, as above.
    ModelPricing(
        "gpt-6-astra", "GPT-6 Astra",
        input_per_mtok=10.00, output_per_mtok=50.00,
        provider="openai",
    ),
    ModelPricing(
        "gpt-5.6-sol", "GPT-5.6 Sol",
        input_per_mtok=4.00, output_per_mtok=20.00,
        provider="openai",
    ),
    ModelPricing(
        "gpt-5.6-terra", "GPT-5.6 Terra",
        input_per_mtok=2.00, output_per_mtok=12.00,
        provider="openai",
    ),
    ModelPricing(
        "gpt-5.6-luna", "GPT-5.6 Luna",
        input_per_mtok=0.20, output_per_mtok=1.20,
        provider="openai",
    ),
]

_BY_MODEL = {m.model: m for m in CATALOG}

# Models dropped from the catalog, and what to use instead. Without this a
# retired id falls back to its provider's *default* -- which is the flagship,
# so a user who deliberately picked the cheap fast model would be moved onto
# the expensive one at 8x the price without being asked. Map a retirement to
# its nearest equivalent instead, and only fall back to the default when there
# isn't one.
REPLACED_MODELS: dict[str, str] = {
    # The 3.x Flash line, superseded by 3.8 Flash (still the cheap, fast tier).
    "gemini-3.5-flash": "gemini-3.8-flash",
    "gemini-3.1-flash-lite": "gemini-3.8-flash",
}


def replacement_for(model: str | None) -> str:
    """The catalogued successor to a retired model id, or ``""``."""
    successor = REPLACED_MODELS.get(model or "")
    return successor if pricing_for(successor) else ""


def pricing_for(model: str | None) -> ModelPricing | None:
    """The catalog entry for a model id, or ``None`` if it isn't one we price
    (a stale saved model, or any model from a live-listing provider) -- in
    which case no dollar figure is shown."""
    return _BY_MODEL.get(model or "")


def catalog_for(provider: str | None) -> list[ModelPricing]:
    """Every priced model belonging to one provider, in display order."""
    return [m for m in CATALOG if m.provider == (provider or "")]


def default_model_for(provider: str | None) -> str:
    """The model a provider starts on: its first catalog entry (the most
    capable; lighter, cheaper options follow it in the dropdown)."""
    entries = catalog_for(provider)
    return entries[0].model if entries else ""


def owns_model(provider: str | None, model: str | None) -> bool:
    """Whether ``model`` can be used under ``provider``.

    A catalogued model belongs to exactly the provider that lists it, and an
    uncatalogued id belongs to nobody -- so switching provider can never leave
    another provider's model id selected, and ``sync_models`` replaces it with
    a model that will actually answer."""
    from .providers import provider_info

    entry = pricing_for(model)
    return entry is not None and entry.provider == provider_info(provider).id


def current_rates(
    pricing: ModelPricing, *, today: date | None = None
) -> tuple[float, float]:
    """The (input, output) base rates in effect on ``today``.

    An entry with a ``scheduled_from`` date carries two prices: the one in
    force now and the announced one that replaces it. Resolving by date here
    means an introductory rate stops being quoted the day it expires, rather
    than the day someone remembers to edit this file -- and until then the
    cheaper rate is still what gets quoted, so nothing is overstated either.

    The date is the machine's local one. Billing boundaries are the provider's,
    so a run in the hours around midnight on a change-over can be priced on the
    wrong side of it; every figure the app shows is labelled an estimate, and
    being a few hours early or late on one day is a far smaller error than
    quoting a withdrawn price for months.

    Scheduled changes apply to the base rates only, never to the ``*_high``
    tier -- no catalog entry uses both, and a test enforces that, because a
    scheduled change on a tiered model would silently leave the tier stale."""
    if pricing.scheduled_from:
        starts = date.fromisoformat(pricing.scheduled_from)
        if (today or date.today()) >= starts:
            return pricing.input_per_mtok_from, pricing.output_per_mtok_from
    return pricing.input_per_mtok, pricing.output_per_mtok


def effective_rates(
    pricing: ModelPricing, usage, *, today: date | None = None
) -> tuple[float, float]:
    """The (input, output) per-million rates that apply to ``usage``.

    For a tiered model the higher rates kick in once a single request's prompt
    exceeds the threshold. This app sends one page per call, so the average
    prompt size per call (``input / calls``) *is* the per-request prompt size
    the threshold is defined against -- in practice always the lower tier, since
    a page image is far under 200K tokens, but modelled correctly regardless."""
    if (
        pricing.tier_threshold
        and usage is not None
        and usage.calls
        and (usage.input / usage.calls) > pricing.tier_threshold
    ):
        return pricing.input_per_mtok_high, pricing.output_per_mtok_high
    return current_rates(pricing, today=today)


def cost_for(pricing: ModelPricing, usage, *, today: date | None = None) -> float:
    """USD cost for ``usage`` under ``pricing`` (thinking billed at the output
    rate, per ``TokenUsage.cost``), at the rates in force on ``today``."""
    in_rate, out_rate = effective_rates(pricing, usage, today=today)
    return usage.cost(in_rate, out_rate)
