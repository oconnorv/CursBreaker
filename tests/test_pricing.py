from cursbreaker.models import TokenUsage
from cursbreaker.pricing import (
    CATALOG,
    cost_for,
    effective_rates,
    pricing_for,
)


def test_catalog_has_the_three_curated_models():
    ids = [m.model for m in CATALOG]
    assert ids == [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    ]


def test_pricing_for_unknown_model_is_none():
    assert pricing_for("not-a-real-model") is None
    assert pricing_for(None) is None


def test_flat_model_cost():
    p = pricing_for("gemini-3.5-flash")  # $1.50 in / $9.00 out
    usage = TokenUsage(input=1_000_000, output=500_000, thinking=500_000, calls=2)
    # input: 1M * $1.50 ; output+thinking: 1M * $9.00
    assert cost_for(p, usage) == 1.50 + 9.00


def test_tiered_model_uses_low_tier_for_small_prompts():
    p = pricing_for("gemini-3.1-pro-preview")
    # One page per call, well under the 200K-token threshold -> low tier.
    usage = TokenUsage(input=4000, output=1000, calls=2)
    in_rate, out_rate = effective_rates(p, usage)
    assert (in_rate, out_rate) == (2.00, 12.00)


def test_tiered_model_uses_high_tier_when_prompt_exceeds_threshold():
    p = pricing_for("gemini-3.1-pro-preview")
    # Average prompt per call above 200K input tokens -> high tier.
    usage = TokenUsage(input=500_000, output=1000, calls=1)
    in_rate, out_rate = effective_rates(p, usage)
    assert (in_rate, out_rate) == (4.00, 18.00)


def test_tiered_cost_low_tier():
    p = pricing_for("gemini-3.1-pro-preview")
    usage = TokenUsage(input=2000, output=1000, calls=2)
    expected = 2000 / 1_000_000 * 2.00 + 1000 / 1_000_000 * 12.00
    assert cost_for(p, usage) == expected


def test_effective_rates_with_zero_calls_falls_back_to_base():
    # No calls yet -> can't be in the high tier; base rates apply.
    p = pricing_for("gemini-3.1-pro-preview")
    assert effective_rates(p, TokenUsage()) == (2.00, 12.00)
