import pytest

from meshagent.llm_proxy.local_proxy import build_local_proxy_env
from meshagent.llm_proxy.usage import (
    UsageCollector,
    extract_anthropic_completion_usage,
    extract_openai_completion_usage,
)


def test_extract_openai_completion_usage_prefers_priced_response_model() -> None:
    usage = extract_openai_completion_usage(
        model="gpt-5.4-2026-03-05",
        request={"model": "gpt-5.4"},
        response={
            "model": "gpt-5.4-2026-03-05",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-5.4-2026-03-05"
    assert usage.tokens == {"input_tokens": 10.0, "output_tokens": 2.0}


def test_extract_openai_completion_usage_falls_back_to_request_model() -> None:
    usage = extract_openai_completion_usage(
        model="gpt-5.4-2099-01-01",
        request={"model": "gpt-5.4", "service_tier": "flex"},
        response={
            "model": "gpt-5.4-2099-01-01",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-5.4"
    assert usage.tokens == {"input_tokens_flex": 10.0, "output_tokens_flex": 2.0}


def test_extract_anthropic_completion_usage_falls_back_to_request_model() -> None:
    usage = extract_anthropic_completion_usage(
        model="claude-opus-4-6-20990101",
        request={"model": "claude-opus-4-6"},
        response={
            "model": "claude-opus-4-6-20990101",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    )

    assert usage is not None
    assert usage.provider == "anthropic"
    assert usage.model == "claude-opus-4-6"
    assert usage.tokens == {
        "input_tokens": 10.0,
        "output_tokens": 2.0,
        "cache_creation_input_tokens": 0.0,
        "cache_read_input_tokens": 0.0,
    }


@pytest.mark.asyncio
async def test_usage_collector_tracks_llm_proxy_surcharge() -> None:
    collector = UsageCollector()

    await collector.record_usage(
        provider="openai",
        model="gpt-4.1",
        tokens={"input_tokens": 1_000.0, "output_tokens": 500.0},
        request_id="req-1",
    )

    snapshot = await collector.snapshot()

    assert snapshot.total_requests == 1
    assert snapshot.subtotal == pytest.approx(0.006)
    assert snapshot.surcharge == pytest.approx(0.0003)
    assert snapshot.total == pytest.approx(0.0063)
    assert len(snapshot.recent_events) == 1
    assert snapshot.recent_events[0].request_id == "req-1"
    assert snapshot.recent_events[0].surcharge == pytest.approx(0.0003)


@pytest.mark.asyncio
async def test_usage_collector_tracks_request_activity_with_aggregated_total() -> None:
    collector = UsageCollector()

    await collector.record_usage(
        provider="openai",
        model="gpt-4.1",
        tokens={"input_tokens": 1_000.0, "output_tokens": 500.0},
        request_id="req-2",
    )
    await collector.record_request_activity(
        provider="openai",
        transport="http",
        method="POST",
        path="/openai/v1/chat/completions",
        status=200,
        result="ok",
        request_id="req-2",
    )

    snapshot = await collector.snapshot()

    assert len(snapshot.recent_requests) == 1
    assert snapshot.recent_requests[0].request_id == "req-2"
    assert snapshot.recent_requests[0].status == 200
    assert snapshot.recent_requests[0].result == "ok"
    assert snapshot.recent_requests[0].total == pytest.approx(0.0063)


def test_build_local_proxy_env_uses_expected_base_urls() -> None:
    env = build_local_proxy_env(
        base_url="http://127.0.0.1:8766",
        bearer_token="local-token",
        insecure=False,
    )

    assert env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8766/openai/v1",
        "OPENAI_API_KEY": "local-token",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8766/anthropic",
        "ANTHROPIC_API_KEY": "local-token",
    }
