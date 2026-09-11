from datetime import date

import pytest

from cursbreaker.models import TokenUsage
from cursbreaker.pricing import (
    CATALOG,
    REPLACED_MODELS,
    catalog_for,
    cost_for,
    current_rates,
    default_model_for,
    effective_rates,
    owns_model,
    pricing_for,
    replacement_for,
)


def test_gemini_catalog_is_pro_first_then_flash():
    # Pro is first in the dropdown (and the saved default); the lighter model
    # follows.
    assert [m.model for m in catalog_for("gemini")] == [
        "gemini-3.1-pro-preview",
        "gemini-3.8-flash",
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


def test_openai_catalog_is_priced_and_flagship_first():
    entries = catalog_for("openai")
    assert [m.model for m in entries] == [
        "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"
    ]
    assert all(m.input_per_mtok > 0 and m.output_per_mtok > 0 for m in entries)


def test_catalogued_models_are_unique_across_providers():
    # pricing_for() is a flat lookup by id, so a duplicate id would silently
    # price one provider's model at another's rate.
    ids = [m.model for m in CATALOG]
    assert len(ids) == len(set(ids))


def test_every_catalog_entry_belongs_to_a_known_provider():
    from cursbreaker.providers import PROVIDERS

    assert {m.provider for m in CATALOG} <= set(PROVIDERS)


def test_default_model_is_the_first_entry_for_every_provider():
    assert default_model_for("gemini") == "gemini-3.1-pro-preview"
    assert default_model_for("anthropic") == "claude-opus-5"
    assert default_model_for("openai") == "gpt-6-astra"


def test_owns_model_keeps_providers_from_inheriting_each_others_models():
    assert owns_model("gemini", "gemini-3.8-flash")
    assert not owns_model("anthropic", "gemini-3.8-flash")
    assert owns_model("anthropic", "claude-opus-5")
    assert not owns_model("gemini", "claude-opus-5")
    assert owns_model("openai", "gpt-5.6-terra")
    assert not owns_model("gemini", "gpt-5.6-terra")
    # An id nobody catalogues belongs to nobody, so sync_models replaces it
    # with one that will actually answer.
    assert not owns_model("openai", "some-unknown-model")
    assert not owns_model("openai", "")


def test_pricing_for_unknown_model_is_none():
    assert pricing_for("not-a-real-model") is None
    assert pricing_for(None) is None


def test_flat_model_cost():
    p = pricing_for("gemini-3.8-flash")  # $0.75 in / $3.75 out
    usage = TokenUsage(input=1_000_000, output=500_000, thinking=500_000, calls=2)
    # input: 1M * $0.75 ; output+thinking: 1M * $3.75
    assert cost_for(p, usage) == 0.75 + 3.75


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


def test_every_retirement_points_at_a_model_that_exists():
    # A successor that isn't in the catalog would send the user to the
    # flagship's price instead of the cheap tier they picked -- exactly what
    # the mapping exists to prevent.
    for retired, successor in REPLACED_MODELS.items():
        assert pricing_for(retired) is None, f"{retired} is still catalogued"
        assert pricing_for(successor) is not None, successor
        assert replacement_for(retired) == successor


def test_retirements_stay_within_one_provider():
    from cursbreaker.providers import provider_info

    for retired, successor in REPLACED_MODELS.items():
        # Both sides are Gemini ids today; a cross-provider mapping would be a
        # bug, since the key for the new provider may not even be set.
        assert provider_info("gemini").id == pricing_for(successor).provider
        assert retired.split("-")[0] == successor.split("-")[0]


# --- scheduled price changes ---------------------------------------------- #
# An introductory rate expires on a published date. The catalog declares both
# prices so estimates switch by themselves; these pin that switch.
INTRO = date(2026, 12, 31)      # last day of the introductory rate
STANDARD = date(2027, 1, 1)     # first day of the standard rate


def test_intro_rate_applies_up_to_and_including_its_last_day():
    flash = pricing_for("gemini-3.8-flash")
    assert current_rates(flash, today=date(2026, 9, 11)) == (0.75, 3.75)
    assert current_rates(flash, today=INTRO) == (0.75, 3.75)


def test_standard_rate_applies_from_the_day_it_takes_effect():
    flash = pricing_for("gemini-3.8-flash")
    assert current_rates(flash, today=STANDARD) == (1.50, 7.50)
    assert current_rates(flash, today=date(2028, 6, 1)) == (1.50, 7.50)


def test_a_run_after_the_change_is_costed_at_the_standard_rate():
    # The whole point: without this the estimate would read half the real cost
    # from January until someone noticed and edited the catalog.
    flash = pricing_for("gemini-3.8-flash")
    usage = TokenUsage(input=1_000_000, output=1_000_000, calls=1)
    assert cost_for(flash, usage, today=INTRO) == pytest.approx(0.75 + 3.75)
    assert cost_for(flash, usage, today=STANDARD) == pytest.approx(1.50 + 7.50)


def test_effective_rates_carries_the_date_through():
    flash = pricing_for("gemini-3.8-flash")
    usage = TokenUsage(input=1000, output=1000, calls=1)
    assert effective_rates(flash, usage, today=INTRO) == (0.75, 3.75)
    assert effective_rates(flash, usage, today=STANDARD) == (1.50, 7.50)


def test_models_without_a_scheduled_change_are_unaffected_by_the_date():
    for model in ("gemini-3.1-pro-preview", "claude-opus-5", "gpt-6-astra"):
        p = pricing_for(model)
        assert current_rates(p, today=date(2030, 1, 1)) == (
            p.input_per_mtok, p.output_per_mtok
        ), model


def test_no_entry_combines_tiering_with_a_scheduled_change():
    # current_rates resolves the base rates only. A tiered model with a
    # scheduled change would keep quoting its stale *_high rates after the
    # change, so the combination is disallowed rather than half-supported.
    for m in CATALOG:
        assert not (m.tier_threshold and m.scheduled_from), m.model


def test_a_scheduled_change_declares_real_rates():
    for m in CATALOG:
        if not m.scheduled_from:
            continue
        date.fromisoformat(m.scheduled_from)          # parses, or this fails
        assert m.input_per_mtok_from > 0, m.model
        assert m.output_per_mtok_from > 0, m.model
