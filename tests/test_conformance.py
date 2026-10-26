"""Conformance tests — skipped in CI until AGENT_MEMORY_BASE_URL is set."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate

from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver

pytestmark = pytest.mark.skipif(
    not os.getenv("AGENT_MEMORY_BASE_URL"),
    reason="requires live Agent Memory Service — set AGENT_MEMORY_BASE_URL",
)


@checkpointer_test(name="HanaAgentMemorySaver")
async def hana_checkpointer() -> AsyncGenerator[HanaAgentMemorySaver, None]:
    """Fixture for HanaAgentMemorySaver with cleanup tracking.

    Tracks all thread_ids created during tests and cleans them up afterward
    to prevent data accumulation across test runs.
    """
    saver = HanaAgentMemorySaver(
        base_url=os.environ["AGENT_MEMORY_BASE_URL"],
        token_url=os.environ.get("AGENT_MEMORY_TOKEN_URL"),
        client_id=os.environ.get("AGENT_MEMORY_CLIENT_ID"),
        client_secret=os.environ.get("AGENT_MEMORY_CLIENT_SECRET"),
    )

    created_threads: set[str] = set()
    original_aput = saver.aput

    async def tracking_aput(config, checkpoint, metadata, new_versions):
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            created_threads.add(thread_id)
        return await original_aput(config, checkpoint, metadata, new_versions)

    with patch.object(saver, "aput", AsyncMock(side_effect=tracking_aput)):
        yield saver

    print(f"\n🧹 Cleaning up {len(created_threads)} test threads...")
    for thread_id in created_threads:
        try:
            await saver.adelete_thread(thread_id)
        except Exception as e:
            print(f"Warning: Failed to cleanup thread {thread_id[:8]}...: {e}")
    print("✓ Cleanup complete")


@pytest.mark.asyncio
async def test_conformance() -> None:
    report = await validate(
        hana_checkpointer,
    )
    report.print_report()
    assert report.passed_all_base(), (
        f"Base conformance tests failed: {report.to_dict()}"
    )
