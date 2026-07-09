from dataclasses import dataclass

from meshagent.anthropic.usage import preprocess_anthropic_usage
from meshagent.openai.tools.usage import preprocess_openai_usage


def per_million(n):
    return n / 1000000


def per_thousand(n):
    return n / 1000


LLM_PROXY_SURCHARGE_RATE = 0.05
LLM_PROXY_SURCHARGED_PROVIDERS = frozenset({"openai", "anthropic"})


@dataclass(frozen=True)
class UsagePricingLineItem:
    type: str
    quantity: float
    unit_price: float
    amount: float
    description: str
    is_surcharge: bool = False


@dataclass(frozen=True)
class ChargeSpec:
    amount: float
    description: str


OPENAI_LONG_CONTEXT_THRESHOLDS = {
    # Source: https://platform.openai.com/docs/pricing
    # Context-length pricing starts at 272K input tokens.
    "gpt-5.4": 272000,
    "gpt-5.4-2026-03-05": 272000,
}


def _openai_input_tokens_for_context_pricing(tokens: dict[str, float]) -> float:
    total = 0.0
    for key, value in tokens.items():
        if key.startswith("input_tokens") or key.startswith("cached_tokens"):
            total += float(value)
    return total


def _apply_openai_context_length_tier(
    *,
    model: str,
    tokens: dict[str, float],
) -> dict[str, float]:
    threshold = OPENAI_LONG_CONTEXT_THRESHOLDS.get(model)
    if threshold is None:
        return tokens

    total_input_tokens = _openai_input_tokens_for_context_pricing(tokens)
    if total_input_tokens < threshold:
        return tokens

    model_pricing = pricing.get("openai", {}).get(model)
    if not isinstance(model_pricing, dict):
        return tokens

    out = dict(tokens)
    for key, value in list(out.items()):
        long_key = f"{key}_long"
        if long_key in model_pricing:
            out.pop(key)
            out[long_key] = value

    return out


def _apply_openai_service_tier(
    *,
    model: str,
    tokens: dict[str, float],
    service_tier: str | None,
) -> dict[str, float]:
    """Map OpenAI usage into tier-specific token keys.

    OpenAI supports different service tiers with different pricing (e.g. `flex`,
    `priority`). We encode the tier into token type keys so aggregated reporting
    remains correct.

    Source: https://platform.openai.com/docs/pricing
    """

    if service_tier is None:
        return tokens

    tier = str(service_tier).lower()
    if tier in {"standard", "default", ""}:
        return tokens

    if tier not in {"flex", "priority"}:
        return tokens

    model_pricing = pricing.get("openai", {}).get(model)
    if not isinstance(model_pricing, dict):
        return tokens

    out = dict(tokens)

    for k, v in list(out.items()):
        tier_key = f"{k}_{tier}"
        if tier_key in model_pricing:
            out.pop(k)
            out[tier_key] = v

    return out


# Preprocessors are keyed by provider then model.
# Use "*" for a provider/model default.
def _to_float(value) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def preprocess_openai_audio_minutes_usage(
    *, model: str, usage: dict
) -> dict[str, float] | None:
    del model

    if not isinstance(usage, dict):
        return None

    audio_minutes = _to_float(usage.get("audio_minutes"))
    if audio_minutes is not None:
        return {"audio_minutes": audio_minutes}

    audio_seconds = _to_float(usage.get("audio_seconds"))
    if audio_seconds is None:
        audio_seconds = _to_float(usage.get("duration_seconds"))
    if audio_seconds is None:
        audio_seconds = _to_float(usage.get("audio_duration_seconds"))

    if audio_seconds is not None:
        return {"audio_minutes": audio_seconds / 60.0}

    audio_ms = _to_float(usage.get("audio_ms"))
    if audio_ms is None:
        audio_ms = _to_float(usage.get("duration_ms"))
    if audio_ms is None:
        audio_ms = _to_float(usage.get("audio_duration_ms"))

    if audio_ms is not None:
        return {"audio_minutes": audio_ms / 60000.0}

    return preprocess_openai_usage(model="", usage=usage)


preprocessors = {
    "openai": {
        "*": preprocess_openai_usage,
        "gpt-realtime-translate": preprocess_openai_audio_minutes_usage,
        "gpt-realtime-whisper": preprocess_openai_audio_minutes_usage,
        "whisper-1": preprocess_openai_audio_minutes_usage,
    },
    "anthropic": {
        "*": preprocess_anthropic_usage,
    },
}


def _drop_zero_usage_values(tokens: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in tokens.items() if float(value) != 0.0}


def is_pricing_available(
    *, provider: str, model: str, service_tier: str | None = None
) -> bool:
    provider_table = pricing.get(provider)
    if not isinstance(provider_table, dict):
        return False

    model_table = provider_table.get(model)
    if not isinstance(model_table, dict):
        return False

    if provider != "openai":
        return True

    if service_tier is None:
        return True

    tier = str(service_tier).lower()
    if tier in {"standard", "default", ""}:
        return True

    if tier not in {"flex", "priority"}:
        return False

    # Tier is specified: require tier-specific base token pricing.
    return (
        f"input_tokens_{tier}" in model_table and f"output_tokens_{tier}" in model_table
    )


def preprocess(
    *,
    provider: str,
    model: str,
    usage: dict,
    service_tier: str | None = None,
) -> dict[str, float] | None:
    """Normalize/flatten usage to a pricing-compatible token dict.

    This is designed to work for:
    - non-streaming responses (`response["usage"]`)
    - streaming / SSE style usage where usage is tracked incrementally

    Callers should pass only the `usage` object.
    """

    provider_table = preprocessors.get(provider)
    if provider_table is None:
        return None

    pre = provider_table.get(model) or provider_table.get("*")
    if pre is None:
        return None

    tokens = pre(model=model, usage=usage)
    if tokens is None:
        return None

    if provider == "openai":
        tier_tokens = _apply_openai_service_tier(
            model=model,
            tokens=tokens,
            service_tier=service_tier,
        )
        tokens = _apply_openai_context_length_tier(
            model=model,
            tokens=tier_tokens,
        )
    filtered_tokens = _drop_zero_usage_values(tokens)
    if len(filtered_tokens) == 0:
        return None

    return filtered_tokens


gpt_4_1_pricing = {
    "input_tokens": per_million(2.00),
    "cached_tokens": per_million(0.50),
    "output_tokens": per_million(8.00),
    # Priority
    "input_tokens_priority": per_million(3.50),
    "cached_tokens_priority": per_million(0.875),
    "output_tokens_priority": per_million(14.00),
}

gpt_4_1_mini_pricing = {
    "input_tokens": per_million(0.40),
    "cached_tokens": per_million(0.10),
    "output_tokens": per_million(1.60),
    # Priority
    "input_tokens_priority": per_million(0.70),
    "cached_tokens_priority": per_million(0.175),
    "output_tokens_priority": per_million(2.80),
}

gpt_4_1_nano_pricing = {
    "input_tokens": per_million(0.10),
    "cached_tokens": per_million(0.025),
    "output_tokens": per_million(0.40),
    # Priority
    "input_tokens_priority": per_million(0.20),
    "cached_tokens_priority": per_million(0.05),
    "output_tokens_priority": per_million(0.80),
}


o1_pricing = {
    # Standard
    "input_tokens": per_million(15.00),
    "cached_tokens": per_million(7.50),
    "output_tokens": per_million(60.00),
    "reasoning_tokens": per_million(60.00),
}

o1_mini_pricing = {
    "input_tokens": per_million(1.10),
    "cached_tokens": per_million(0.55),
    "output_tokens": per_million(4.40),
    "reasoning_tokens": per_million(4.40),
}

# o3 standard pricing.
# Source: https://platform.openai.com/docs/pricing
o3_pricing = {
    # Standard
    "input_tokens": per_million(2.00),
    "cached_tokens": per_million(0.50),
    "output_tokens": per_million(8.00),
    "reasoning_tokens": per_million(8.00),
    # Flex
    "input_tokens_flex": per_million(1.00),
    "cached_tokens_flex": per_million(0.25),
    "output_tokens_flex": per_million(4.00),
    "reasoning_tokens_flex": per_million(4.00),
    # Priority
    "input_tokens_priority": per_million(3.50),
    "cached_tokens_priority": per_million(0.875),
    "output_tokens_priority": per_million(14.00),
    "reasoning_tokens_priority": per_million(14.00),
}

# o3-deep-research pricing.
# Source: https://platform.openai.com/docs/pricing
o3_deep_research_pricing = {
    "input_tokens": per_million(10.00),
    "cached_tokens": per_million(2.50),
    "output_tokens": per_million(40.00),
    "reasoning_tokens": per_million(40.00),
}

o3_mini_pricing = {
    "input_tokens": per_million(1.10),
    "cached_tokens": per_million(0.55),
    "output_tokens": per_million(4.40),
    "reasoning_tokens": per_million(4.40),
}

o4_mini_pricing = {
    # Standard
    "input_tokens": per_million(1.10),
    "cached_tokens": per_million(0.275),
    "output_tokens": per_million(4.40),
    "reasoning_tokens": per_million(4.40),
    # Flex
    "input_tokens_flex": per_million(0.55),
    "cached_tokens_flex": per_million(0.138),
    "output_tokens_flex": per_million(2.20),
    "reasoning_tokens_flex": per_million(2.20),
    # Priority
    "input_tokens_priority": per_million(2.00),
    "cached_tokens_priority": per_million(0.50),
    "output_tokens_priority": per_million(8.00),
    "reasoning_tokens_priority": per_million(8.00),
}

gpt_5_pricing = {
    # Standard
    "input_tokens": per_million(1.25),
    "cached_tokens": per_million(0.125),
    "output_tokens": per_million(10.00),
    # Flex
    "input_tokens_flex": per_million(0.625),
    "cached_tokens_flex": per_million(0.0625),
    "output_tokens_flex": per_million(5.00),
    # Priority
    "input_tokens_priority": per_million(2.50),
    "cached_tokens_priority": per_million(0.25),
    "output_tokens_priority": per_million(20.00),
}

gpt_5_1_pricing = {
    # Standard
    "input_tokens": per_million(1.25),
    "cached_tokens": per_million(0.125),
    "output_tokens": per_million(10.00),
    # Flex
    "input_tokens_flex": per_million(0.625),
    "cached_tokens_flex": per_million(0.0625),
    "output_tokens_flex": per_million(5.00),
    # Priority
    "input_tokens_priority": per_million(2.50),
    "cached_tokens_priority": per_million(0.25),
    "output_tokens_priority": per_million(20.00),
}

gpt_5_2_pricing = {
    # Standard
    "input_tokens": per_million(1.75),
    "cached_tokens": per_million(0.175),
    "output_tokens": per_million(14.00),
    # Flex
    "input_tokens_flex": per_million(0.875),
    "cached_tokens_flex": per_million(0.0875),
    "output_tokens_flex": per_million(7.00),
    # Priority
    "input_tokens_priority": per_million(3.50),
    "cached_tokens_priority": per_million(0.35),
    "output_tokens_priority": per_million(28.00),
}

gpt_5_3_pricing = {
    # Standard
    "input_tokens": per_million(1.75),
    "cached_tokens": per_million(0.175),
    "output_tokens": per_million(14.00),
    # Flex
    "input_tokens_flex": per_million(0.875),
    "cached_tokens_flex": per_million(0.0875),
    "output_tokens_flex": per_million(7.00),
    # Priority
    "input_tokens_priority": per_million(3.50),
    "cached_tokens_priority": per_million(0.35),
    "output_tokens_priority": per_million(28.00),
}

gpt_5_4_pricing = {
    # Standard
    "input_tokens": per_million(2.50),
    "cached_tokens": per_million(0.25),
    "output_tokens": per_million(15.00),
    # Flex
    "input_tokens_flex": per_million(1.25),
    "cached_tokens_flex": per_million(0.125),
    "output_tokens_flex": per_million(7.50),
    # Priority
    "input_tokens_priority": per_million(5.00),
    "cached_tokens_priority": per_million(0.50),
    "output_tokens_priority": per_million(30.00),
    # Context-length pricing (>=272K input tokens in session)
    "input_tokens_long": per_million(5.00),
    "cached_tokens_long": per_million(0.50),
    "output_tokens_long": per_million(22.50),
    "input_tokens_flex_long": per_million(2.50),
    "cached_tokens_flex_long": per_million(0.25),
    "output_tokens_flex_long": per_million(11.25),
}

gpt_5_4_mini_pricing = {
    # Standard
    "input_tokens": per_million(0.75),
    "cached_tokens": per_million(0.075),
    "output_tokens": per_million(4.50),
    # Flex
    "input_tokens_flex": per_million(0.375),
    "cached_tokens_flex": per_million(0.0375),
    "output_tokens_flex": per_million(2.25),
}

gpt_5_4_nano_pricing = {
    # Standard
    "input_tokens": per_million(0.20),
    "cached_tokens": per_million(0.02),
    "output_tokens": per_million(1.25),
}

# Source: https://openai.com/index/introducing-gpt-5-5/
# The release announcement publishes input/output pricing and tier multipliers.
# Cached-input pricing follows the existing GPT-5 family pricing pattern.
gpt_5_5_pricing = {
    # Standard
    "input_tokens": per_million(5.00),
    "cached_tokens": per_million(0.50),
    "output_tokens": per_million(30.00),
    # Image generation tool tokens use the default GPT Image 2 pricing.
    "image_input_tokens": per_million(8.00),
    "image_cached_tokens": per_million(2.00),
    "image_output_tokens": per_million(30.00),
    # Flex
    "input_tokens_flex": per_million(2.50),
    "cached_tokens_flex": per_million(0.25),
    "output_tokens_flex": per_million(15.00),
    # Priority
    "input_tokens_priority": per_million(12.50),
    "cached_tokens_priority": per_million(1.25),
    "output_tokens_priority": per_million(75.00),
}


# Source: https://openai.com/index/gpt-5-6/
# Cache reads cost 10% of uncached input; cache writes cost 125%.
def gpt_5_6_pricing(*, input_price: float, output_price: float):
    return {
        "input_tokens": per_million(input_price),
        "cached_tokens": per_million(input_price * 0.10),
        "cache_write_tokens": per_million(input_price * 1.25),
        "output_tokens": per_million(output_price),
        # Image generation tool tokens use the default GPT Image 2 pricing.
        "image_input_tokens": per_million(8.00),
        "image_cached_tokens": per_million(2.00),
        "image_output_tokens": per_million(30.00),
    }


gpt_5_6_sol_pricing = gpt_5_6_pricing(input_price=5.00, output_price=30.00)
gpt_5_6_terra_pricing = gpt_5_6_pricing(input_price=2.50, output_price=15.00)
gpt_5_6_luna_pricing = gpt_5_6_pricing(input_price=1.00, output_price=6.00)

# Codex mini pricing.
# Source: https://platform.openai.com/docs/pricing
codex_mini_pricing = {
    "input_tokens": per_million(1.50),
    "cached_tokens": per_million(0.375),
    "output_tokens": per_million(6.00),
}

gpt_mini_pricing = {
    # Standard
    "input_tokens": per_million(0.25),
    "cached_tokens": per_million(0.025),
    "output_tokens": per_million(2.00),
    # Flex
    "input_tokens_flex": per_million(0.125),
    "cached_tokens_flex": per_million(0.0125),
    "output_tokens_flex": per_million(1.00),
    # Priority
    "input_tokens_priority": per_million(0.45),
    "cached_tokens_priority": per_million(0.045),
    "output_tokens_priority": per_million(3.60),
}

gpt_nano_pricing = {
    # Standard
    "input_tokens": per_million(0.05),
    "cached_tokens": per_million(0.005),
    "output_tokens": per_million(0.40),
    # Flex
    "input_tokens_flex": per_million(0.025),
    "cached_tokens_flex": per_million(0.0025),
    "output_tokens_flex": per_million(0.20),
}

gpt_realtime_pricing = {
    "input_tokens": per_million(4.00),
    "cached_tokens": per_million(0.40),
    "output_tokens": per_million(16.00),
    "audio_input_tokens": per_million(32.00),
    "audio_cached_tokens": per_million(0.40),
    "audio_output_tokens": per_million(64.00),
    "image_input_tokens": per_million(5.00),
    "image_cached_tokens": per_million(0.50),
}

gpt_realtime_2_pricing = {
    # Text tokens
    "input_tokens": per_million(4.00),
    "cached_tokens": per_million(0.40),
    "output_tokens": per_million(24.00),
    "reasoning_tokens": per_million(24.00),
    # Audio tokens
    "audio_input_tokens": per_million(32.00),
    "audio_cached_tokens": per_million(0.40),
    "audio_output_tokens": per_million(64.00),
    # Image tokens
    "image_input_tokens": per_million(5.00),
    "image_cached_tokens": per_million(0.50),
}

# Anthropic pricing (Claude API).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Notes:
# - `cache_creation_input_tokens` == prompt cache writes (default 5m TTL).
# - `cache_read_input_tokens` == prompt cache reads/hits.

claude_opus_4_6_pricing = {
    "input_tokens": per_million(5.00),
    "cache_creation_input_tokens": per_million(6.25),
    "cache_read_input_tokens": per_million(0.50),
    "output_tokens": per_million(25.00),
}

claude_opus_4_7_pricing = claude_opus_4_6_pricing

claude_opus_4_8_pricing = claude_opus_4_6_pricing

claude_opus_4_5_pricing = {
    "input_tokens": per_million(5.00),
    "cache_creation_input_tokens": per_million(6.25),
    "cache_read_input_tokens": per_million(0.50),
    "output_tokens": per_million(25.00),
}

claude_sonnet_4_5_pricing = {
    # Standard pricing (<= 200K total input tokens)
    "input_tokens": per_million(3.00),
    "cache_creation_input_tokens": per_million(3.75),
    "cache_read_input_tokens": per_million(0.30),
    "output_tokens": per_million(15.00),
    # Long-context pricing (> 200K total input tokens)
    "input_tokens_long": per_million(6.00),
    "cache_creation_input_tokens_long": per_million(7.50),
    "cache_read_input_tokens_long": per_million(0.60),
    "output_tokens_long": per_million(22.50),
}

claude_sonnet_4_6_pricing = {
    "input_tokens": per_million(3.00),
    "cache_creation_input_tokens": per_million(3.75),
    "cache_read_input_tokens": per_million(0.30),
    "output_tokens": per_million(15.00),
}

# Introductory pricing through August 31, 2026.
claude_sonnet_5_pricing = {
    "input_tokens": per_million(2.00),
    "cache_creation_input_tokens": per_million(2.50),
    "cache_read_input_tokens": per_million(0.20),
    "output_tokens": per_million(10.00),
}

claude_haiku_4_5_pricing = {
    "input_tokens": per_million(1.00),
    "cache_creation_input_tokens": per_million(1.25),
    "cache_read_input_tokens": per_million(0.10),
    "output_tokens": per_million(5.00),
}

claude_opus_4_pricing = {
    "input_tokens": per_million(15.00),
    "cache_creation_input_tokens": per_million(18.75),
    "cache_read_input_tokens": per_million(1.50),
    "output_tokens": per_million(75.00),
}

claude_sonnet_4_pricing = {
    # Standard pricing (<= 200K total input tokens)
    "input_tokens": per_million(3.00),
    "cache_creation_input_tokens": per_million(3.75),
    "cache_read_input_tokens": per_million(0.30),
    "output_tokens": per_million(15.00),
    # Long-context pricing (> 200K total input tokens)
    "input_tokens_long": per_million(6.00),
    "cache_creation_input_tokens_long": per_million(7.50),
    "cache_read_input_tokens_long": per_million(0.60),
    "output_tokens_long": per_million(22.50),
}

claude_haiku_3_5_pricing = {
    "input_tokens": per_million(0.80),
    "cache_creation_input_tokens": per_million(1.00),
    "cache_read_input_tokens": per_million(0.08),
    "output_tokens": per_million(4.00),
}

claude_haiku_3_pricing = {
    "input_tokens": per_million(0.25),
    "cache_creation_input_tokens": per_million(0.30),
    "cache_read_input_tokens": per_million(0.03),
    "output_tokens": per_million(1.25),
}

computer_use_preview_pricing = {
    "input_tokens": per_million(3.0),
    "output_tokens": per_million(12.0),
}


pricing = {
    "openai": {
        # Image models
        # Source: https://developers.openai.com/api/docs/pricing
        "gpt-image-2": {
            # Text tokens
            "input_tokens": per_million(5.00),
            "cached_tokens": per_million(1.25),
            # Image tokens
            "image_input_tokens": per_million(8.00),
            "image_cached_tokens": per_million(2.00),
            "image_output_tokens": per_million(30.00),
        },
        "gpt-image-1.5": {
            # Text tokens
            "input_tokens": per_million(5.00),
            "cached_tokens": per_million(1.25),
            "output_tokens": per_million(10.00),
            # Image tokens
            "image_input_tokens": per_million(8.00),
            "image_cached_tokens": per_million(2.00),
            "image_output_tokens": per_million(32.00),
        },
        "chatgpt-image-latest": {
            "input_tokens": per_million(5.00),
            "cached_tokens": per_million(1.25),
            "output_tokens": per_million(10.00),
            "image_input_tokens": per_million(8.00),
            "image_cached_tokens": per_million(2.00),
            "image_output_tokens": per_million(32.00),
        },
        "gpt-image-1": {
            "input_tokens": per_million(5.00),
            "cached_tokens": per_million(1.25),
            # Image token rates
            "image_input_tokens": per_million(10.00),
            "image_cached_tokens": per_million(2.50),
            "image_output_tokens": per_million(40.00),
        },
        "gpt-image-1-mini": {
            "input_tokens": per_million(2.00),
            "cached_tokens": per_million(0.20),
            "image_input_tokens": per_million(2.50),
            "image_cached_tokens": per_million(0.25),
            "image_output_tokens": per_million(8.00),
        },
        # Embeddings
        "text-embedding-3-small": {
            "input_tokens": per_million(0.02),
        },
        "text-embedding-3-large": {
            "input_tokens": per_million(0.13),
        },
        "text-embedding-ada-002": {
            "input_tokens": per_million(0.10),
        },
        # Video (Sora)
        "sora-2": {
            "video_seconds_720p": 0.10,
        },
        "sora-2-pro": {
            "video_seconds_720p": 0.30,
            "video_seconds_1024": 0.50,
        },
        # Audio
        "gpt-audio": {
            "input_tokens": per_million(2.50),
            "output_tokens": per_million(10.00),
            "audio_input_tokens": per_million(32.00),
            "audio_output_tokens": per_million(64.00),
        },
        "gpt-audio-mini": {
            "input_tokens": per_million(0.60),
            "output_tokens": per_million(2.40),
            "audio_input_tokens": per_million(10.00),
            "audio_output_tokens": per_million(20.00),
        },
        "gpt-realtime": gpt_realtime_pricing,
        "gpt-realtime-1.5": gpt_realtime_pricing,
        "gpt-realtime-mini": {
            "input_tokens": per_million(0.60),
            "cached_tokens": per_million(0.06),
            "output_tokens": per_million(2.40),
            "audio_input_tokens": per_million(10.00),
            "audio_cached_tokens": per_million(0.30),
            "audio_output_tokens": per_million(20.00),
            "image_input_tokens": per_million(0.80),
            "image_cached_tokens": per_million(0.08),
        },
        # Realtime 2
        # Source: https://openai.com/api/pricing/
        "gpt-realtime-2.1": gpt_realtime_2_pricing,
        "gpt-realtime-2": gpt_realtime_2_pricing,
        "gpt-realtime-translate": {
            "audio_minutes": 0.034,
        },
        "gpt-realtime-whisper": {
            "audio_minutes": 0.017,
        },
        # Speech / transcription (STT + TTS)
        # Source: https://platform.openai.com/docs/pricing
        "gpt-4o-mini-tts": {
            "input_tokens": per_million(0.60),
            "audio_output_tokens": per_million(12.00),
        },
        "gpt-4o-transcribe": {
            "input_tokens": per_million(2.50),
            "output_tokens": per_million(10.00),
            "audio_input_tokens": per_million(2.50),
        },
        "gpt-4o-transcribe-diarize": {
            "input_tokens": per_million(2.50),
            "output_tokens": per_million(10.00),
            "audio_input_tokens": per_million(2.50),
        },
        "gpt-4o-mini-transcribe": {
            "input_tokens": per_million(1.25),
            "output_tokens": per_million(5.00),
            "audio_input_tokens": per_million(1.25),
        },
        # Legacy speech models
        "whisper-1": {
            "audio_minutes": 0.006,
        },
        "tts-1": {
            "input_characters": per_million(15.00),
        },
        "tts-1-hd": {
            "input_characters": per_million(30.00),
        },
        "gpt-5": gpt_5_pricing,
        "gpt-5.6": gpt_5_6_sol_pricing,
        "gpt-5.6-sol": gpt_5_6_sol_pricing,
        "gpt-5.6-terra": gpt_5_6_terra_pricing,
        "gpt-5.6-luna": gpt_5_6_luna_pricing,
        "gpt-5.5": gpt_5_5_pricing,
        "gpt-5.5-2026-04-23": gpt_5_5_pricing,
        "gpt-5.4": gpt_5_4_pricing,
        "gpt-5.4-2026-03-05": gpt_5_4_pricing,
        "gpt-5.4-mini": gpt_5_4_mini_pricing,
        "gpt-5.4-mini-2026-03-17": gpt_5_4_mini_pricing,
        "gpt-5.2": gpt_5_2_pricing,
        "computer-use-preview": computer_use_preview_pricing,
        "gpt-5.3-codex": gpt_5_3_pricing,
        "gpt-5.2-codex": gpt_5_2_pricing,
        "gpt-5.2-chat-latest": gpt_5_2_pricing,
        "gpt-5.2-2025-12-11": gpt_5_2_pricing,
        "gpt-5.1": gpt_5_1_pricing,
        "gpt-5.1-2025-11-13": gpt_5_1_pricing,
        "gpt-5.1-codex": gpt_5_1_pricing,
        "gpt-5.1-codex-max": gpt_5_1_pricing,
        "gpt-5.1-codex-mini": gpt_mini_pricing,
        "gpt-5.1-mini": gpt_mini_pricing,
        "gpt-5-2025-08-07": gpt_5_pricing,
        "gpt-5-codex": gpt_5_pricing,
        "gpt-5-mini": gpt_mini_pricing,
        "gpt-5-mini-2025-08-07": gpt_mini_pricing,
        "gpt-5.4-nano": gpt_5_4_nano_pricing,
        "gpt-5.4-nano-2026-03-17": gpt_5_4_nano_pricing,
        "gpt-5-nano": gpt_nano_pricing,
        "gpt-5-nano-2025-08-07": gpt_nano_pricing,
        "gpt-4.5-preview": {
            "input_tokens": per_million(75.00),
            "cached_tokens": per_million(37.50),
            "output_tokens": per_million(150.00),
        },
        "gpt-4o": {
            # Standard
            "input_tokens": per_million(2.50),
            "cached_tokens": per_million(1.25),
            "output_tokens": per_million(10.00),
            # Priority
            "input_tokens_priority": per_million(4.25),
            "cached_tokens_priority": per_million(2.125),
            "output_tokens_priority": per_million(17.00),
        },
        # Older snapshot had different rates.
        # Source: https://platform.openai.com/docs/pricing
        "gpt-4o-2024-05-13": {
            # Standard
            "input_tokens": per_million(5.00),
            "output_tokens": per_million(15.00),
            # Priority
            "input_tokens_priority": per_million(8.75),
            "output_tokens_priority": per_million(26.25),
        },
        "gpt-4.1": gpt_4_1_pricing,
        "gpt-4.1-2025-04-14": gpt_4_1_pricing,
        "gpt-4.1-mini": gpt_4_1_mini_pricing,
        "gpt-4.1-mini-2025-04-14": gpt_4_1_mini_pricing,
        "gpt-4.1-nano": gpt_4_1_nano_pricing,
        "gpt-4.1-nano-2025-04-14": gpt_4_1_nano_pricing,
        "gpt-4o-audio-preview": {
            # Text tokens (cached input is not offered for this model)
            "input_tokens": per_million(2.50),
            "output_tokens": per_million(10.00),
            # Audio tokens (cached input is not offered for this model)
            "audio_input_tokens": per_million(40.00),
            "audio_output_tokens": per_million(80.00),
        },
        "gpt-4o-realtime-preview": {
            # Text tokens
            "input_tokens": per_million(5.00),
            "cached_tokens": per_million(2.50),
            "output_tokens": per_million(20.00),
            # Audio tokens
            "audio_input_tokens": per_million(40.00),
            "audio_cached_tokens": per_million(2.50),
            "audio_output_tokens": per_million(80.00),
        },
        "gpt-4o-mini": {
            # Standard
            "input_tokens": per_million(0.15),
            "cached_tokens": per_million(0.075),
            "output_tokens": per_million(0.60),
            # Priority
            "input_tokens_priority": per_million(0.25),
            "cached_tokens_priority": per_million(0.125),
            "output_tokens_priority": per_million(1.00),
        },
        "gpt-4o-mini-audio-preview": {
            # Text tokens (cached input is not offered for this model)
            "input_tokens": per_million(0.15),
            "output_tokens": per_million(0.60),
            # Audio tokens (cached input is not offered for this model)
            "audio_input_tokens": per_million(10.00),
            "audio_output_tokens": per_million(20.00),
        },
        "gpt-4o-mini-realtime-preview": {
            # Text tokens
            "input_tokens": per_million(0.60),
            "cached_tokens": per_million(0.30),
            "output_tokens": per_million(2.40),
            # Audio tokens
            "audio_input_tokens": per_million(10.00),
            "audio_cached_tokens": per_million(0.30),
            "audio_output_tokens": per_million(20.00),
        },
        # Codex
        # Source: https://platform.openai.com/docs/pricing
        "codex-mini-latest": codex_mini_pricing,
        "o1": o1_pricing,
        "o1-2024-12-17": o1_pricing,
        "o1-preview": o1_pricing,
        "o1-preview-2024-09-12": o1_pricing,
        "o1-mini": o1_mini_pricing,
        "o1-mini-2024-09-12": o1_mini_pricing,
        "o3": o3_pricing,
        "o3-2025-04-16": o3_pricing,
        "o3-deep-research": o3_deep_research_pricing,
        "o3-mini": o3_mini_pricing,
        "o3-mini-2025-01-31": o3_mini_pricing,
        "o4-mini": o4_mini_pricing,
        "o4-mini-2025-01-31": o4_mini_pricing,
    },
    "anthropic": {
        # Latest model aliases
        "claude-opus-4-8": claude_opus_4_8_pricing,
        "claude-opus-4-7": claude_opus_4_7_pricing,
        "claude-opus-4-6": claude_opus_4_6_pricing,
        "claude-opus-4-5": claude_opus_4_5_pricing,
        "claude-sonnet-5": claude_sonnet_5_pricing,
        "claude-sonnet-4-6": claude_sonnet_4_6_pricing,
        "claude-sonnet-4-5": claude_sonnet_4_5_pricing,
        "claude-haiku-4-5": claude_haiku_4_5_pricing,
        # Latest model snapshots (from models overview)
        "claude-opus-4-5-20251101": claude_opus_4_5_pricing,
        "claude-sonnet-4-5-20250929": claude_sonnet_4_5_pricing,
        "claude-haiku-4-5-20251001": claude_haiku_4_5_pricing,
        # Legacy models / aliases
        "claude-opus-4-1": claude_opus_4_pricing,
        "claude-opus-4-1-20250805": claude_opus_4_pricing,
        "claude-opus-4-0": claude_opus_4_pricing,
        "claude-opus-4": claude_opus_4_pricing,
        "claude-opus-4-20250514": claude_opus_4_pricing,
        "claude-sonnet-4-0": claude_sonnet_4_pricing,
        "claude-sonnet-4": claude_sonnet_4_pricing,
        "claude-sonnet-4-20250514": claude_sonnet_4_pricing,
        "claude-3-7-sonnet-latest": claude_sonnet_4_5_pricing,
        "claude-3-7-sonnet-20250219": claude_sonnet_4_5_pricing,
        # Back-compat for old SDK defaults
        "claude-3-5-sonnet-latest": claude_sonnet_4_5_pricing,
        "claude-3-5-haiku-latest": claude_haiku_3_5_pricing,
        # Claude 3 era
        "claude-3-opus-20240229": claude_opus_4_pricing,
        "claude-3-opus": claude_opus_4_pricing,
        "claude-3-sonnet-20240229": claude_sonnet_4_pricing,
        "claude-3-sonnet": claude_sonnet_4_pricing,
        "claude-3-haiku-20240307": claude_haiku_3_pricing,
        "claude-3-haiku": claude_haiku_3_pricing,
        "claude-haiku-3": claude_haiku_3_pricing,
    },
    "meshagent.fal": {
        "fal-ai/flux/dev": {"megapixels": 0.025},
        "fal-ai/flux/dev/image-to-image": {"megapixels": 0.03},
        "fal-ai/flux/dev/redux": {"megapixels": 0.025},
        "fal-ai/flux/schnell": {"megapixels": 0.003},
        "fal-ai/flux/schnell/redux": {"megapixels": 0.025},
        "fal-ai/flux-pro/new": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1.1": {"megapixels": 0.04},
        "fal-ai/flux-pro/v1.1-ultra": {"images": 0.06},
        # "fal-ai/flux-pro/v1.1-ultra-finetuned" : {
        #   "images": 0.07
        # },
        "fal-ai/flux-pro/v1.1-ultra/redux": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1.1/redux": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1/canny": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1/canny-finetuned": {"megapixels": 0.06},
        "fal-ai/flux-pro/v1/depth": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1/depth-finetuned": {"megapixels": 0.06},
        "fal-ai/flux-pro/v1/fill": {"megapixels": 0.05},
        "fal-ai/flux-pro/v1/fill-finetuned": {"megapixels": 0.06},
        "fal-ai/flux-pro/v1/redux": {"megapixels": 0.05},
        "fal-ai/stable-video": {"videos": 0.075},
        "fal-ai/hunyuan-video": {"videos": 0.4},
        "fal-ai/hunyuan-video-img2vid-lora": {"videos": 0.3},
        "fal-ai/minimax/video-01-live": {"videos": 0.5},
        "fal-ai/minimax/video-01-live/image-to-video": {"videos": 0.5},
        "fal-ai/minimax/video-01-director": {"videos": 0.5},
        "fal-ai/minimax/video-01-director/image-to-video": {"videos": 0.5},
        "fal-ai/minimax/video-01-subject-reference": {"videos": 0.5},
        "fal-ai/veo2": {"video-seconds": 0.5},
        "fal-ai/veo2/image-to-video": {"video-seconds": 0.5},
        "fal-ai/kling-video/v1.5/pro/effects": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.5/pro/image-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.5/pro/text-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.6/pro/effects": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.6/pro/image-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.6/pro/text-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1.6/standard/effects": {"video-seconds": 0.03},
        "fal-ai/kling-video/v1.6/standard/image-to-video": {"video-seconds": 0.03},
        "fal-ai/kling-video/v1.6/standard/text-to-video": {"video-seconds": 0.03},
        "fal-ai/kling-video/v1/pro/effects": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1/pro/image-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1/pro/text-to-video": {"video-seconds": 0.1},
        "fal-ai/kling-video/v1/standard/effects": {"video-seconds": 0.03},
        "fal-ai/kling-video/v1/standard/image-to-video": {"video-seconds": 0.03},
        "fal-ai/kling-video/v1/standard/text-to-video": {"video-seconds": 0.03},
        "fal-ai/wan-i2v": {
            "videos": 0.4,
        },
        "fal-ai/wan-t2v": {
            "videos": 0.4,
        },
        "fal-ai/wan-pro/image-to-video": {
            "videos": 0.8,
        },
        "fal-ai/wan-pro/text-to-video": {
            "videos": 0.8,
        },
        "fal-ai/wan/v2.1/1.3b/text-to-video": {
            "videos": 0.2,
        },
    },
    "meshagent.firecrawl": {
        "firecrawl": {"firecrawl_credits": 0.01},
        "firecrawl_queue": {"firecrawl_credits": 0.01},
    },
    "meshagent": {
        "room": {"minutes": 0.01},
        "daily-storage": {"gibibyte_days": 0.05},
    },
    "meshagent.perplexity": {
        "sonar-deep-research": {
            "input_tokens": per_million(2),
            "output_tokens": per_million(8),
            "reasoning_tokens": per_million(3),
            "invocations": per_thousand(5),
        },
        "sonar-reasoning-pro": {
            "input_tokens": per_million(2),
            "output_tokens": per_million(8),
            "reasoning_tokens": 0,
            "invocations": per_thousand(5),
        },
        "sonar-reasoning": {
            "input_tokens": per_million(1),
            "output_tokens": per_million(5),
            "reasoning_tokens": 0,
            "invocations": per_thousand(5),
        },
        "sonar-pro": {
            "input_tokens": per_million(3),
            "output_tokens": per_million(15),
            "reasoning_tokens": 0,
            "invocations": per_thousand(5),
        },
        "sonar": {
            "input_tokens": per_million(1),
            "output_tokens": per_million(1),
            "reasoning_tokens": 0,
            "invocations": per_thousand(5),
        },
        "r1-1776": {
            "input_tokens": per_million(2),
            "output_tokens": per_million(8),
        },
    },
}


def supports_llm_proxy_surcharge(provider: str) -> bool:
    return provider in LLM_PROXY_SURCHARGED_PROVIDERS


def build_usage_pricing_line_items(
    *,
    provider: str,
    model: str,
    usage: dict[str, float],
    surcharge_rate: float = LLM_PROXY_SURCHARGE_RATE,
) -> list[UsagePricingLineItem]:
    line_items = list[UsagePricingLineItem]()

    provider_pricing = pricing.get(provider)
    if not isinstance(provider_pricing, dict):
        return line_items

    model_pricing = provider_pricing.get(model)
    if not isinstance(model_pricing, dict):
        return line_items

    subtotal = 0.0
    for usage_type, total in usage.items():
        surcharge_only = usage_type.startswith("custom_")
        priced_usage_type = (
            usage_type.removeprefix("custom_") if surcharge_only else usage_type
        )
        unit_price = model_pricing.get(priced_usage_type)
        if unit_price is None:
            continue

        quantity = float(total)
        amount = quantity * float(unit_price)
        subtotal += amount
        if surcharge_only:
            continue
        line_items.append(
            UsagePricingLineItem(
                type=usage_type,
                quantity=quantity,
                unit_price=float(unit_price),
                amount=amount,
                description=f"{provider} - {model} - {usage_type}",
            )
        )

    if subtotal > 0 and supports_llm_proxy_surcharge(provider):
        surcharge_amount = subtotal * surcharge_rate
        line_items.append(
            UsagePricingLineItem(
                type="llm_proxy_surcharge",
                quantity=1.0,
                unit_price=surcharge_amount,
                amount=surcharge_amount,
                description=f"{provider} - {model} - llm_proxy_surcharge",
                is_surcharge=True,
            )
        )

    return line_items


def build_charge_specs(
    *,
    provider: str,
    model: str,
    usage: dict[str, float],
    surcharge_rate: float = LLM_PROXY_SURCHARGE_RATE,
) -> list[ChargeSpec]:
    return [
        ChargeSpec(amount=line_item.amount * 100, description=line_item.description)
        for line_item in build_usage_pricing_line_items(
            provider=provider,
            model=model,
            usage=usage,
            surcharge_rate=surcharge_rate,
        )
    ]
