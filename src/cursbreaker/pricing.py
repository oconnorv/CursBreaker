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

**OpenAI carries no entries here.** Its models are read live from the user's own
key instead (``providers.PROVIDERS['openai'].lists_models_live``), because this
file's promise is that a listed price is a *published* price someone checked --
and no verified OpenAI price list was available when this was written. The
consequence is visible and already handled everywhere: with no catalog entry
``pricing_for`` returns ``None``, the UI says there's no published price for the
model, and token counts are reported without a dollar figure. Adding verified
``ModelPricing(..., provider="openai")`` rows below is all it takes to switch
OpenAI onto the same automatic estimate as the others.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """

    model: str
    label: str
    input_per_mtok: float
    output_per_mtok: float
    tier_threshold: int = 0
    input_per_mtok_high: float = 0.0
    output_per_mtok_high: float = 0.0
    provider: str = "gemini"


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
    ModelPricing(
        "gemini-3.5-flash", "Gemini 3.5 Flash",
        input_per_mtok=1.50, output_per_mtok=9.00,
        provider="gemini",
    ),
    ModelPricing(
        "gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite",
        input_per_mtok=0.25, output_per_mtok=1.50,
        provider="gemini",
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
    # Intentionally empty; see the module docstring.
]

_BY_MODEL = {m.model: m for m in CATALOG}


def pricing_for(model: str | None) -> ModelPricing | None:
    """The catalog entry for a model id, or ``None`` if it isn't one we price
    (a stale saved model, or any model from a live-listing provider) -- in
    which case no dollar figure is shown."""
    return _BY_MODEL.get(model or "")


def catalog_for(provider: str | None) -> list[ModelPricing]:
    """Every priced model belonging to one provider, in display order."""
    return [m for m in CATALOG if m.provider == (provider or "")]


def default_model_for(provider: str | None) -> str:
    """The model a provider starts on: the first catalog entry, or ``""`` for a
    provider whose models are listed live (the user picks from their own key,
    and guessing an id here would 404 on the first call)."""
    entries = catalog_for(provider)
    return entries[0].model if entries else ""


def owns_model(provider: str | None, model: str | None) -> bool:
    """Whether ``model`` can be used under ``provider``.

    A catalogued model belongs to exactly the provider that lists it. A model
    we don't price is accepted only by a live-listing provider -- that's how
    OpenAI ids (and nothing else) pass -- so switching provider can never leave
    another provider's model id selected."""
    from .providers import provider_info

    entry = pricing_for(model)
    info = provider_info(provider)
    if entry is not None:
        return entry.provider == info.id
    return bool(model) and info.lists_models_live


def effective_rates(pricing: ModelPricing, usage) -> tuple[float, float]:
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
    return pricing.input_per_mtok, pricing.output_per_mtok


def cost_for(pricing: ModelPricing, usage) -> float:
    """USD cost for ``usage`` under ``pricing`` (thinking billed at the output
    rate, per ``TokenUsage.cost``)."""
    in_rate, out_rate = effective_rates(pricing, usage)
    return usage.cost(in_rate, out_rate)
