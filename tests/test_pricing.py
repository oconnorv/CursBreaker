from cursbreaker.models import TokenUsage
from cursbreaker.pricing import (
    CATALOG,
    catalog_for,
    cost_for,
    default_model_for,
    effective_rates,
    owns_model,
    pricing_for,
)


def test_gemini_catalog_lists_the_three_curated_models_pro_first():
    # Pro is first in the dropdown (and the saved default); lighter models follow.
    assert [m.model for m in catalog_for("gemini")] == [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]


def test_anthropic_catalog_is_priced_and_opus_first():
    entries = catalog_for("anthropic")
    assert [m.model for m in entries] == [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ]
    # Every catalogued model must carry a real price, or the estimate it backs
    # would quietly read $0.00 rather than "no published price".
    assert all(m.input_per_mtok > 0 and m.output_per_mtok > 0 for m in entries)


def test_openai_has_no_curated_prices_so_models_are_listed_live():
    # Deliberate: no verified OpenAI price list, so nothing is invented here.
    # The UI path for an unpriced model (tokens, no dollar figure) covers it.
    assert catalog_for("openai") == []
    assert default_model_for("openai") == ""


def test_every_catalog_entry_belongs_to_a_known_provider():
    from cursbreaker.providers import PROVIDERS

    assert {m.provider for m in CATALOG} <= set(PROVIDERS)


def test_default_model_is_the_first_entry_for_priced_providers():
    assert default_model_for("gemini") == "gemini-3.1-pro-preview"
    assert default_model_for("anthropic") == "claude-opus-5"


def test_owns_model_keeps_providers_from_inheriting_each_others_models():
    assert owns_model("gemini", "gemini-3.5-flash")
    assert not owns_model("anthropic", "gemini-3.5-flash")
    assert owns_model("anthropic", "claude-opus-5")
    assert not owns_model("gemini", "claude-opus-5")
    # A model we don't price is acceptable only where models are listed live.
    assert owns_model("openai", "some-new-openai-model")
    assert not owns_model("gemini", "some-new-openai-model")
    assert not owns_model("openai", "")


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
