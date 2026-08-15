from __future__ import annotations

import os

import pytest
from opentelemetry import trace

from llm_future_affinity.config import ExecutionConfig, InferenceConfig, ModelConfig, RoutingConfig
from llm_future_affinity.openrouter import OpenRouterClient
from llm_future_affinity.telemetry import NullTelemetry


@pytest.mark.live
async def test_live_openrouter_single_completion() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set")
    model = ModelConfig(
        model_family="smoke",
        model_id=os.environ.get("OPENROUTER_SMOKE_MODEL", "openai/gpt-5.6-luna"),
        routing=RoutingConfig(endpoint_slug=os.environ.get("OPENROUTER_SMOKE_ENDPOINT", "openai")),
        inference=InferenceConfig(
            max_tokens=64,
            temperature=0,
            top_p=1,
            top_k=None,
            min_p=None,
            seed=None,
            reasoning=None,
            thinking=None,
        ),
    )
    client = OpenRouterClient(api_key, model, ExecutionConfig(metadata_timeout_seconds=2), NullTelemetry())
    try:
        await client.preflight()
        response = await client.complete(
            [{"role": "user", "content": "Reply with exactly: SUBMIT ABCD"}],
            trace.INVALID_SPAN,
            {"test.kind": "live_smoke"},
        )
        assert response.content.strip()
        assert response.generation_id
    finally:
        await client.close()
