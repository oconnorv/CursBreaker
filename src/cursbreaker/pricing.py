"""Curated transcription models and their published prices.

The app offers a fixed dropdown of models (rather than free-text entry) so the
cost estimate can be computed *automatically* from each model's published
per-million-token price -- the user never types a price. Prices are
point-in-time and do change, so ``PRICES_AS_OF`` is shown in the UI next to a
link to live pricing, and every dollar figure stays labelled an estimate, never
a guarantee.

To refresh prices: edit the numbers below and bump ``PRICES_AS_OF``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump whenever the numbers below change; surfaced in the UI for transparency.
PRICES_AS_OF = "2026-06-03"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


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


# The dropdown, in display order. gemini-3.1-pro-preview stays the saved default
# (most accurate for handwriting); the lighter models trade some accuracy for a
# much lower price.
CATALOG: list[ModelPricing] = [
    ModelPricing(
        "gemini-3.5-flash", "Gemini 3.5 Flash",
        input_per_mtok=1.50, output_per_mtok=9.00,
    ),
    ModelPricing(
        "gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite",
        input_per_mtok=0.25, output_per_mtok=1.50,
    ),
    ModelPricing(
        "gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)",
        input_per_mtok=2.00, output_per_mtok=12.00,
        tier_threshold=200_000,
        input_per_mtok_high=4.00, output_per_mtok_high=18.00,
    ),
]

_BY_MODEL = {m.model: m for m in CATALOG}


def pricing_for(model: str | None) -> ModelPricing | None:
    """The catalog entry for a model id, or ``None`` if it isn't one we price
    (e.g. a stale saved model) -- in which case no dollar figure is shown."""
    return _BY_MODEL.get(model or "")


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
