from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from meshagent.llm_proxy.pricing import is_pricing_available, pricing, preprocess


@pytest.mark.parametrize(
    "model,input_price,cached_price,output_price",
    [
        ("gpt-6.2-astra", 8.00, 0.40, 40.00),
        ("gpt-6.2-sol", 2.00, 0.10, 10.00),
        ("gpt-6.1-sol", 2.00, 0.10, 10.00),
        ("gpt-6.1-terra", 1.00, 0.05, 5.00),
        ("gpt-6-astra", 10.00, 1.00, 50.00),
        ("gpt-6-sol", 2.00, 0.20, 10.00),
        ("gpt-6-luna", 0.10, 0.01, 0.50),
    ],
)
@pytest.mark.parametrize("tier,multiplier", [(None, 1), ("flex", 0.5), ("fast", 2)])
@pytest.mark.parametrize("total_input", [272_000, 272_001])
def test_gpt_6_billing_tiers(
    model, input_price, cached_price, output_price, tier, multiplier, total_input
):
    # Source: https://developers.openai.com/api/docs/pricing
    if model.startswith("gpt-6.2-") and tier == "flex":
        assert not is_pricing_available(
            provider="openai", model=model, service_tier=tier
        )
        return

    tokens = preprocess(
        provider="openai",
        model=model,
        service_tier=tier,
        usage={
            "input_tokens": total_input,
            "input_tokens_details": {
                "cached_tokens": 100_000,
                "cache_write_tokens": 50_000,
            },
            "output_tokens": 1_000,
        },
    )
    assert tokens is not None
    assert is_pricing_available(provider="openai", model=model, service_tier=tier)
    input_multiplier = 2 if total_input > 272_000 else 1
    output_multiplier = 1.5 if total_input > 272_000 else 1
    expected = (
        (
            (total_input - 150_000) * input_price * input_multiplier
            + 100_000 * cached_price * input_multiplier
            + 50_000 * input_price * 1.25 * input_multiplier
            + 1_000 * output_price * output_multiplier
        )
        * multiplier
        / 1_000_000
    )
    assert sum(
        quantity * pricing["openai"][model][key] for key, quantity in tokens.items()
    ) == pytest.approx(expected)


def test_astra_ultrafast_bills_long_context_and_cache_writes():
    tokens = preprocess(
        provider="openai",
        model="gpt-6-astra",
        service_tier="ultrafast",
        usage={
            "input_tokens": 300_000,
            "input_tokens_details": {
                "cached_tokens": 100_000,
                "cache_write_tokens": 50_000,
            },
            "output_tokens": 1_000,
        },
    )
    assert tokens == {
        "input_tokens_ultrafast_long": 150_000,
        "cached_tokens_ultrafast_long": 100_000,
        "cache_write_tokens_ultrafast_long": 50_000,
        "output_tokens_ultrafast_long": 1_000,
    }
    assert is_pricing_available(
        provider="openai", model="gpt-6-astra", service_tier="ultrafast"
    )
    assert is_pricing_available(
        provider="openai", model="gpt-6.1-sol", service_tier="ultrafast"
    )
    assert sum(
        quantity * pricing["openai"]["gpt-6-astra"][key]
        for key, quantity in tokens.items()
    ) == pytest.approx(27.15)


@pytest.mark.parametrize(
    "model,standard_cost", [("gpt-6.2-astra", 3.54), ("gpt-6.2-sol", 0.885)]
)
@pytest.mark.parametrize(
    "now,multiplier",
    [
        (datetime(2027, 3, 1, 7, 59, 59, tzinfo=UTC), 3),
        (datetime(2027, 3, 1, 8, tzinfo=UTC), 6),
    ],
)
def test_gpt_6_2_ultrafast_promotion(model, standard_cost, now, multiplier):
    assert is_pricing_available(
        provider="openai", model=model, service_tier="ultrafast"
    )
    with patch("meshagent.llm_proxy.pricing.datetime") as clock:
        clock.now.return_value = now
        tokens = preprocess(
            provider="openai",
            model=model,
            service_tier="ultrafast",
            usage={
                "input_tokens": 300_000,
                "input_tokens_details": {
                    "cached_tokens": 100_000,
                    "cache_write_tokens": 50_000,
                },
                "output_tokens": 1_000,
            },
        )
    assert tokens is not None
    assert sum(
        quantity * pricing["openai"][model][key] for key, quantity in tokens.items()
    ) == pytest.approx(standard_cost * multiplier)


@pytest.mark.parametrize(
    "model,input_price,cached_price,output_price",
    [
        ("claude-opus-5", 5.00, 0.50, 25.00),
        ("claude-opus-5-5", 4.00, 0.20, 20.00),
        ("claude-sonnet-5-5", 2.00, 0.10, 10.00),
        ("claude-fable-5", 10.00, 1.00, 50.00),
        ("claude-mythos-5", 10.00, 1.00, 50.00),
        ("claude-fable-5-1", 10.00, 0.25, 50.00),
        ("claude-mythos-5-1", 10.00, 0.25, 50.00),
        ("claude-fable-5-2", 8.00, 0.20, 40.00),
        ("claude-mythos-5-2", 8.00, 0.20, 40.00),
    ],
)
def test_new_claude_models_bill_standard_rates_across_full_context(
    model, input_price, cached_price, output_price
):
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    tokens = preprocess(
        provider="anthropic",
        model=model,
        usage={
            "input_tokens": 700_000,
            "cache_creation_input_tokens": 50_000,
            "cache_read_input_tokens": 100_000,
            "output_tokens": 1_000,
        },
    )
    assert tokens is not None
    expected = (
        700_000 * input_price
        + 50_000 * input_price * 1.25
        + 100_000 * cached_price
        + 1_000 * output_price
    ) / 1_000_000
    assert sum(
        quantity * pricing["anthropic"][model][key] for key, quantity in tokens.items()
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "input_tokens,suffix,multiplier", [(50_000, "", 1), (50_001, "_long", 5)]
)
def test_claude_haiku_5_5_counts_cache_tokens_toward_long_context(
    input_tokens, suffix, multiplier
):
    model = "claude-haiku-5-5"
    tokens = preprocess(
        provider="anthropic",
        model=model,
        usage={
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 10_000,
            "cache_read_input_tokens": 40_000,
            "output_tokens": 1_000,
        },
    )
    assert tokens == {
        f"input_tokens{suffix}": input_tokens,
        f"cache_creation_input_tokens{suffix}": 10_000,
        f"cache_read_input_tokens{suffix}": 40_000,
        f"output_tokens{suffix}": 1_000,
    }
    assert sum(
        quantity * pricing["anthropic"][model][key] for key, quantity in tokens.items()
    ) == pytest.approx(
        (input_tokens * 0.10 + 10_000 * 0.125 + 40_000 * 0.01 + 1_000 * 0.50)
        * multiplier
        / 1_000_000
    )


@pytest.mark.parametrize(
    "model,rate",
    [("gpt-live-transcribe", 0.017), ("gpt-transcribe", 0.0045), ("gpt-live-1", 0.05)],
)
def test_new_audio_models_bill_fractional_minutes(model, rate):
    tokens = preprocess(provider="openai", model=model, usage={"duration_seconds": 30})
    assert tokens == {"audio_minutes": 0.5}
    assert pricing["openai"][model]["audio_minutes"] * tokens[
        "audio_minutes"
    ] == pytest.approx(rate / 2)


@pytest.mark.parametrize("model", ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare"])
def test_new_image_model_prices(model):
    assert pricing["openai"][model] == pricing["openai"]["gpt-image-2"]


def test_new_realtime_mini_model_prices():
    assert (
        pricing["openai"]["gpt-realtime-2.1-mini"]
        == pricing["openai"]["gpt-realtime-mini"]
    )


def test_cyber_model_prices():
    assert pricing["openai"]["gpt-5.6-cyber"] == pytest.approx(
        {
            "input_tokens": 12.50 / 1_000_000,
            "cached_tokens": 1.25 / 1_000_000,
            "cache_write_tokens": 15.625 / 1_000_000,
            "output_tokens": 75.00 / 1_000_000,
        }
    )
