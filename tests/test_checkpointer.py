"""Unit tests for HanaAgentMemorySaver — mocked HTTP, no network."""

from __future__ import annotations

import base64
import json
import time

import httpx
import pytest
import respx
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.sap.agent_memory import IAS, XSUAA, HanaAgentMemorySaver
from langgraph.checkpoint.sap.agent_memory._http import (
    CachedToken,
    TokenFetcher,
)
from langgraph.checkpoint.sap.agent_memory.checkpointer import _escape_odata_string

BASE_URL = "https://host.example/agent-memory"
THREAD_ID = "thread-abc"
SNAPSHOT_SCOPE = ""
SNAPSHOT_ID = "snap-0000-1111-2222-3333"
PARENT_SNAPSHOT_ID = "snap-prev-0000-1111-2222"
TASK_ID = "task-xyz"
TOKEN_URL = "https://tenant.authentication.example/oauth/token"
CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"

SNAPSHOTS_URL = f"{BASE_URL}/v1/agent/snapshots"
SNAPSHOT_WRITES_URL = f"{BASE_URL}/v1/agent/snapshotWrites"
DELETE_BY_THREAD_URL = f"{BASE_URL}/v1/agent/snapshots/deleteByThread"

_serde = JsonPlusSerializer()


def _saver(**kwargs) -> HanaAgentMemorySaver:
    saver = HanaAgentMemorySaver(
        base_url=BASE_URL,
        auth=XSUAA(
            token_url=TOKEN_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        ),
        **kwargs,
    )
    # Pre-seed token cache so tests never make a real token HTTP call.
    # This keeps respx.calls[0] pointing to the actual service endpoint.
    saver._token_cache._token = "test-token"
    saver._token_cache._expires_at = time.monotonic() + 3600
    return saver


def _saver_with_tenant(tenant_id: str = "t1") -> HanaAgentMemorySaver:
    """XSUAA saver with a pre-seeded per-tenant token cache."""
    saver = _saver()
    tc = saver._token_registry.get(tenant_id)
    tc._token = "test-token"
    tc._expires_at = time.monotonic() + 3600
    return saver


def _ias_saver(monkeypatch, **kwargs) -> HanaAgentMemorySaver:
    """Build an IAS-only saver (no XSUAA creds) with a pre-seeded token cache."""
    for var in (
        "AGENT_MEMORY_TOKEN_URL",
        "AGENT_MEMORY_CLIENT_ID",
        "AGENT_MEMORY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS(), **kwargs)
    saver._token_cache._token = "test-token"
    saver._token_cache._expires_at = time.monotonic() + 3600
    return saver


def _enc_state(obj: object) -> tuple[str, str]:
    type_, data = _serde.dumps_typed(obj)
    return type_, base64.b64encode(data).decode()


def _enc_metadata(metadata: dict) -> dict:
    """Return metadata as-is"""
    return metadata


def _enc_payload(payload_obj: object) -> tuple[str, str]:
    type_, data = _serde.dumps_typed(payload_obj)
    return type_, base64.b64encode(data).decode()


def _make_snapshot(
    snapshot_id: str = SNAPSHOT_ID,
    parent_snapshot_id: str | None = PARENT_SNAPSHOT_ID,
    state: dict | None = None,
    metadata: dict | None = None,
    thread_id: str = THREAD_ID,
    snapshot_scope: str = SNAPSHOT_SCOPE,
    checkpoint_id: str | None = None,
    writes: list | None = None,
) -> dict:
    # Use checkpoint_id if provided, otherwise use snapshot_id
    actual_snapshot_id = checkpoint_id if checkpoint_id is not None else snapshot_id

    state_obj = state or {
        "id": actual_snapshot_id,
        "channel_values": {"messages": []},
        "v": 1,
        "ts": "2026-01-01T00:00:00Z",
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    stype, encoded_state = _enc_state(state_obj)
    return {
        "threadId": thread_id,
        "snapshotScope": snapshot_scope,
        "snapshotId": actual_snapshot_id,
        "parentSnapshotId": parent_snapshot_id,
        "serializationType": stype,
        "state": encoded_state,
        "metadata": _enc_metadata(
            metadata or {"source": "loop", "step": 1, "writes": {}, "parents": {}}
        ),
        "createdAt": "2026-01-01T00:00:00Z",
        "writes": writes if writes is not None else [],
    }


def _make_write_row(
    snapshot_id: str = SNAPSHOT_ID,
    write_key: str = TASK_ID,
    writes: list | None = None,
) -> dict:
    payload_obj = writes if writes is not None else [(0, "messages", ["hello"])]
    stype, encoded = _enc_payload(payload_obj)
    return {
        "threadId": THREAD_ID,
        "snapshotScope": SNAPSHOT_SCOPE,
        "snapshotId": snapshot_id,
        "writeKey": write_key,
        "serializationType": stype,
        "payload": encoded,
        "createdAt": "2026-01-01T00:00:00Z",
    }


def _config(checkpoint_id: str | None = None) -> dict:
    cfg: dict = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
        }
    }
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


# ---------------------------------------------------------------------------
# get_tuple — sync
# ---------------------------------------------------------------------------


@respx.mock
def test_get_tuple_returns_none_when_no_snapshot():
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    result = _saver().get_tuple(_config())
    assert result is None


@respx.mock
def test_get_tuple_latest_uses_orderby_and_top():
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    assert result.config["configurable"]["checkpoint_id"] == SNAPSHOT_ID
    url = str(respx.calls[0].request.url)
    assert "$orderby" in url or "$top" in url


@respx.mock
def test_get_tuple_specific_checkpoint_no_orderby():
    """When checkpoint_id is specified, should use OData key syntax without $orderby."""
    row = _make_snapshot()

    # Mock the OData key-based URL (returns single object, not array)
    key_url = f"{SNAPSHOTS_URL}(threadId='{THREAD_ID}',snapshotScope='{SNAPSHOT_SCOPE}',snapshotId='{SNAPSHOT_ID}')"
    respx.get(key_url).mock(return_value=httpx.Response(200, json=row))

    # Mock writes URL

    result = _saver().get_tuple(_config(checkpoint_id=SNAPSHOT_ID))

    assert result is not None
    url = str(respx.calls[0].request.url)
    # Key-based access should not use $orderby
    assert "$orderby" not in url
    # Should use key syntax instead
    assert "threadId=" in url
    assert "snapshotId=" in url


@respx.mock
def test_get_tuple_assembles_parent_config():
    row = _make_snapshot(parent_snapshot_id=PARENT_SNAPSHOT_ID)
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    assert result.parent_config is not None
    assert result.parent_config["configurable"]["checkpoint_id"] == PARENT_SNAPSHOT_ID


@respx.mock
def test_get_tuple_no_parent_config_when_none():
    row = _make_snapshot(parent_snapshot_id=None)
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    assert result.parent_config is None


@respx.mock
def test_get_tuple_includes_pending_writes():
    write_row = _make_write_row(writes=[(0, "messages", ["hello"])])
    row = _make_snapshot(writes=[write_row])
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    assert len(result.pending_writes) == 1
    task_id, channel, value = result.pending_writes[0]
    assert task_id == TASK_ID
    assert channel == "messages"
    assert value == ["hello"]


@respx.mock
def test_get_tuple_pending_writes_sorted_by_write_index():
    write_row = _make_write_row(writes=[(1, "b", "vb"), (0, "a", "va")])
    row = _make_snapshot(writes=[write_row])
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    channels = [w[1] for w in result.pending_writes]
    assert channels == ["a", "b"]


@respx.mock
def test_get_tuple_flattens_writes_across_rows():
    write_rows = [
        _make_write_row(write_key="task-1", writes=[(0, "ch1", "v1")]),
        _make_write_row(write_key="task-2", writes=[(1, "ch2", "v2")]),
    ]
    row = _make_snapshot(writes=write_rows)
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = _saver().get_tuple(_config())

    assert result is not None
    assert len(result.pending_writes) == 2
    task_ids = {w[0] for w in result.pending_writes}
    assert task_ids == {"task-1", "task-2"}


# ---------------------------------------------------------------------------
# put — sync
# ---------------------------------------------------------------------------


@respx.mock
def test_put_posts_and_returns_updated_config():
    route = respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "2026-01-01T00:00:00Z",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "loop", "step": 3, "writes": {}, "parents": {}}

    result = _saver().put(
        _config(checkpoint_id=PARENT_SNAPSHOT_ID), checkpoint, metadata, {}
    )

    assert result["configurable"]["checkpoint_id"] == SNAPSHOT_ID
    assert result["configurable"]["thread_id"] == THREAD_ID

    body = json.loads(route.calls[0].request.content)
    assert body["threadId"] == THREAD_ID
    assert body["snapshotId"] == SNAPSHOT_ID
    assert body["parentSnapshotId"] == PARENT_SNAPSHOT_ID
    assert body["snapshotScope"] == SNAPSHOT_SCOPE
    assert body["serializationType"]
    # state decodes to original checkpoint
    decoded = _serde.loads_typed(
        (body["serializationType"], base64.b64decode(body["state"]))
    )
    assert decoded["id"] == SNAPSHOT_ID
    # metadata is now a native dict (no base64 encoding)
    assert body["metadata"] == metadata
    # no legacy fields
    assert "step" not in body
    assert "framework" not in body


@respx.mock
def test_put_sends_bearer_token_when_token_url_configured():
    """OAuth2 token is fetched and sent as Authorization header."""
    respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "fetched-token", "expires_in": 3600}
        )
    )
    route = respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    saver = HanaAgentMemorySaver(
        base_url=BASE_URL,
        auth=XSUAA(
            token_url=TOKEN_URL,
            client_id="my-client",
            client_secret="my-secret",
        ),
    )
    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(
        _config(),
        checkpoint,
        {"source": "loop", "step": 0, "writes": {}, "parents": {}},
        {},
    )

    assert route.calls[0].request.headers["Authorization"] == "Bearer fetched-token"


# ---------------------------------------------------------------------------
# put_writes — sync
# ---------------------------------------------------------------------------


@respx.mock
def test_put_writes_posts_to_writes_endpoint():
    route = respx.post(SNAPSHOT_WRITES_URL).mock(
        return_value=httpx.Response(200, json={})
    )

    _saver().put_writes(
        _config(checkpoint_id=SNAPSHOT_ID),
        [("messages", ["hello"]), ("output", "done")],
        TASK_ID,
    )

    body = json.loads(route.calls[0].request.content)
    assert body["threadId"] == THREAD_ID
    assert body["snapshotId"] == SNAPSHOT_ID
    assert body["snapshotScope"] == SNAPSHOT_SCOPE
    assert body["writeKey"] == TASK_ID
    assert body["serializationType"]
    decoded = _serde.loads_typed(
        (body["serializationType"], base64.b64decode(body["payload"]))
    )
    assert len(decoded) == 2
    assert decoded[0][1] == "messages"
    assert decoded[1][1] == "output"


def test_put_writes_raises_when_no_checkpoint_id():
    with pytest.raises(ValueError, match="checkpoint_id"):
        _saver().put_writes(_config(), [("ch", "val")], TASK_ID)


# ---------------------------------------------------------------------------
# list — sync
# ---------------------------------------------------------------------------


@respx.mock
def test_list_yields_newest_first():
    rows = [
        _make_snapshot(snapshot_id="snap-2"),
        _make_snapshot(snapshot_id="snap-1"),
    ]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    results = list(_saver().list(_config()))

    assert len(results) == 2
    assert results[0].config["configurable"]["checkpoint_id"] == "snap-2"
    assert results[1].config["configurable"]["checkpoint_id"] == "snap-1"


@respx.mock
def test_list_with_limit():
    rows = [_make_snapshot(snapshot_id="snap-1")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    results = list(_saver().list(_config(), limit=1))

    url = str(respx.calls[0].request.url)
    assert "$top" in url
    assert len(results) == 1


@respx.mock
def test_list_with_limit_does_not_follow_next_link():
    """list(limit=N) must not follow @odata.nextLink — $top caps the total."""
    rows = [_make_snapshot(snapshot_id="snap-1")]
    # First page includes a nextLink — should NOT be followed
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": rows,
                "@odata.nextLink": f"{SNAPSHOTS_URL}?$skiptoken=abc",
            },
        )
    )

    results = list(_saver().list(_config(), limit=1))

    assert len(results) == 1
    # Only 2 calls: one snapshot GET + one writes GET — no nextLink followed
    snapshot_calls = [
        c
        for c in respx.calls
        if "snapshot" in str(c.request.url)
        and "snapshotWrites" not in str(c.request.url)
        and "skiptoken" not in str(c.request.url)
    ]
    assert len(snapshot_calls) == 1


@respx.mock
def test_list_without_limit_follows_next_link():
    """list() with no limit must follow @odata.nextLink until exhausted."""
    page1_rows = [_make_snapshot(snapshot_id="snap-1")]
    page2_rows = [_make_snapshot(snapshot_id="snap-2")]
    next_url = f"{SNAPSHOTS_URL}?$skiptoken=page2"

    # Register nextLink route FIRST — respx matches in registration order,
    # and the base SNAPSHOTS_URL pattern also matches URLs with query params.
    respx.get(next_url).mock(
        return_value=httpx.Response(200, json={"value": page2_rows})
    )
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"value": page1_rows, "@odata.nextLink": next_url},
        )
    )

    results = list(_saver().list(_config()))

    assert len(results) == 2
    assert results[0].config["configurable"]["checkpoint_id"] == "snap-1"
    assert results[1].config["configurable"]["checkpoint_id"] == "snap-2"


@pytest.mark.asyncio
@respx.mock
async def test_alist_without_limit_follows_next_link():
    """alist() with no limit must follow @odata.nextLink until exhausted."""
    page1_rows = [_make_snapshot(snapshot_id="snap-1")]
    page2_rows = [_make_snapshot(snapshot_id="snap-2")]
    next_url = f"{SNAPSHOTS_URL}?$skiptoken=page2"

    respx.get(next_url).mock(
        return_value=httpx.Response(200, json={"value": page2_rows})
    )
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"value": page1_rows, "@odata.nextLink": next_url},
        )
    )

    results = [t async for t in _saver().alist(_config())]

    assert len(results) == 2


@respx.mock
def test_list_returns_empty_for_limit_zero():
    """limit=0 should return empty iterator without making requests."""
    # config=None lists all threads
    results = list(_saver().list(_config(), limit=0))
    assert results == []
    assert len(respx.calls) == 0


@respx.mock
def test_list_filter_excludes_non_matching_rows():
    """Rows whose metadata does not match the filter are excluded (server-side)."""
    # Mock returns empty because backend filtered out non-matching rows
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    results = list(_saver().list(_config(), filter={"source": "loop"}))
    assert results == []


@respx.mock
def test_list_filter_includes_matching_rows():
    """Rows whose metadata matches the filter are included."""
    row = _make_snapshot(
        metadata={"source": "loop", "step": 1, "writes": {}, "parents": {}}
    )
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    results = list(_saver().list(_config(), filter={"source": "loop"}))
    assert len(results) == 1


@respx.mock
def test_list_with_none_config_lists_all_threads():
    """config=None should list checkpoints across all threads."""
    rows = [
        _make_snapshot(thread_id="thread-1", checkpoint_id="snap-1"),
        _make_snapshot(thread_id="thread-2", checkpoint_id="snap-2"),
    ]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    results = list(_saver().list(None))

    assert len(results) == 2
    # Verify URL has no threadId filter
    url = str(respx.calls[0].request.url)
    assert "threadId" not in url
    assert "$orderby=snapshotId%20desc" in url


@respx.mock
def test_list_without_checkpoint_ns_lists_all_namespaces():
    """Missing checkpoint_ns key should list all namespaces."""
    config = {"configurable": {"thread_id": "thread-abc"}}  # No checkpoint_ns
    rows = [
        _make_snapshot(
            thread_id="thread-abc", snapshot_scope="", checkpoint_id="snap-1"
        ),
        _make_snapshot(
            thread_id="thread-abc", snapshot_scope="subgraph", checkpoint_id="snap-2"
        ),
    ]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    results = list(_saver().list(config))

    assert len(results) == 2
    url = str(respx.calls[0].request.url)
    assert "snapshotScope" not in url  # No namespace filter


@respx.mock
def test_list_with_empty_checkpoint_ns_filters_to_root():
    """Explicit checkpoint_ns='' should filter to root namespace."""
    config = {"configurable": {"thread_id": "thread-abc", "checkpoint_ns": ""}}
    row = _make_snapshot(
        thread_id="thread-abc", snapshot_scope="", checkpoint_id="snap-1"
    )
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    list(_saver().list(config))  # Execute but don't store

    url = str(respx.calls[0].request.url)
    # Check for URL-encoded version: %27 is single quote '
    assert "snapshotScope%20eq%20''" in url or "snapshotScope%20eq%20%27%27" in url


@respx.mock
def test_list_with_checkpoint_id_returns_single():
    """checkpoint_id should return single checkpoint (point query)."""
    config = {
        "configurable": {
            "thread_id": "thread-abc",
            "checkpoint_ns": "",
            "checkpoint_id": "snap-123",
        }
    }
    row = _make_snapshot(
        thread_id="thread-abc", snapshot_scope="", checkpoint_id="snap-123"
    )
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    results = list(_saver().list(config))

    assert len(results) == 1
    url = str(respx.calls[0].request.url)
    # Check for URL-encoded version: %27 is single quote '
    assert (
        "snapshotId%20eq%20'snap-123'" in url
        or "snapshotId%20eq%20%27snap-123%27" in url
    )


# ---------------------------------------------------------------------------
# Async — get_tuple
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_aget_tuple_returns_none_when_no_snapshot():
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    result = await _saver().aget_tuple(_config())
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_aget_tuple_returns_latest():
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    result = await _saver().aget_tuple(_config())

    assert result is not None
    assert result.config["configurable"]["checkpoint_id"] == SNAPSHOT_ID


# ---------------------------------------------------------------------------
# Async — aput
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_aput_posts_checkpoint():
    route = respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "loop", "step": 5, "writes": {}, "parents": {}}
    result = await _saver().aput(
        _config(checkpoint_id=PARENT_SNAPSHOT_ID), checkpoint, metadata, {}
    )

    assert result["configurable"]["checkpoint_id"] == SNAPSHOT_ID
    body = json.loads(route.calls[0].request.content)
    assert body["snapshotId"] == SNAPSHOT_ID
    assert body["parentSnapshotId"] == PARENT_SNAPSHOT_ID
    # metadata is now a native dict (no base64 encoding)
    assert body["metadata"] == metadata


# ---------------------------------------------------------------------------
# Async — aput_writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_aput_writes_posts_to_writes_endpoint():
    route = respx.post(SNAPSHOT_WRITES_URL).mock(
        return_value=httpx.Response(200, json={})
    )

    await _saver().aput_writes(
        _config(checkpoint_id=SNAPSHOT_ID),
        [("messages", "hello")],
        TASK_ID,
    )

    body = json.loads(route.calls[0].request.content)
    assert body["writeKey"] == TASK_ID
    assert body["snapshotId"] == SNAPSHOT_ID
    decoded = _serde.loads_typed(
        (body["serializationType"], base64.b64decode(body["payload"]))
    )
    assert len(decoded) == 1
    assert decoded[0][1] == "messages"


# ---------------------------------------------------------------------------
# Async — alist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_alist_yields_checkpoints():
    rows = [_make_snapshot(snapshot_id="snap-1")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    results = [t async for t in _saver().alist(_config())]

    assert len(results) == 1
    assert results[0].config["configurable"]["checkpoint_id"] == "snap-1"


# ---------------------------------------------------------------------------
# New cases (SPEC requirements)
# ---------------------------------------------------------------------------


def test_get_next_version_returns_uuid7_string():
    import re

    v = _saver().get_next_version(None, None)
    assert isinstance(v, str)
    # UUID format: 8-4-4-4-12 hex chars
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}", v
    )


def test_get_next_version_is_monotonically_increasing():
    saver = _saver()
    v1 = saver.get_next_version(None, None)
    v2 = saver.get_next_version(v1, None)
    assert v2 > v1


@respx.mock
def test_delete_thread_posts_to_delete_by_thread_action():
    route = respx.post(DELETE_BY_THREAD_URL).mock(
        return_value=httpx.Response(200, json={"deleted": 3})
    )

    _saver_with_tenant().delete_thread(THREAD_ID, tenant_id="t1")

    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["threadId"] == THREAD_ID


@respx.mock
def test_delete_thread_treats_404_as_success():
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(404))

    # Should not raise
    _saver_with_tenant().delete_thread(THREAD_ID, tenant_id="t1")


@pytest.mark.asyncio
@respx.mock
async def test_adelete_thread_posts_to_delete_by_thread_action():
    route = respx.post(DELETE_BY_THREAD_URL).mock(
        return_value=httpx.Response(200, json={"deleted": 3})
    )

    await _saver_with_tenant().adelete_thread(THREAD_ID, tenant_id="t1")

    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["threadId"] == THREAD_ID


@pytest.mark.asyncio
@respx.mock
async def test_adelete_thread_treats_404_as_success():
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(404))

    await _saver_with_tenant().adelete_thread(THREAD_ID, tenant_id="t1")


@respx.mock
def test_put_retry_fires_on_429():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429)
        return httpx.Response(201, json={})

    respx.post(SNAPSHOTS_URL).mock(side_effect=handler)

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    _saver().put(
        _config(),
        checkpoint,
        {"source": "loop", "step": 0, "writes": {}, "parents": {}},
        {},
    )

    assert call_count == 3


@respx.mock
def test_put_retry_fires_on_502():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return httpx.Response(502)
        return httpx.Response(201, json={})

    respx.post(SNAPSHOTS_URL).mock(side_effect=handler)

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    _saver().put(
        _config(),
        checkpoint,
        {"source": "loop", "step": 0, "writes": {}, "parents": {}},
        {},
    )

    assert call_count == 2


@respx.mock
def test_put_does_not_retry_on_400():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(400)

    respx.post(SNAPSHOTS_URL).mock(side_effect=handler)

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    with pytest.raises(httpx.HTTPStatusError):
        _saver().put(
            _config(),
            checkpoint,
            {"source": "loop", "step": 0, "writes": {}, "parents": {}},
            {},
        )

    assert call_count == 1


@respx.mock
def test_put_does_not_retry_on_413():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(413)

    respx.post(SNAPSHOTS_URL).mock(side_effect=handler)

    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    with pytest.raises(httpx.HTTPStatusError):
        _saver().put(
            _config(),
            checkpoint,
            {"source": "loop", "step": 0, "writes": {}, "parents": {}},
            {},
        )

    assert call_count == 1


def test_config_specs_returns_tenant_id():
    specs = _saver().config_specs
    ids = {s.id for s in specs}
    assert "tenant_id" in ids
    assert "agent_id" not in ids
    assert "invoker_id" not in ids


@respx.mock
def test_checkpoint_map_silently_discarded():
    """checkpoint_map in configurable must not be forwarded to the API."""
    route = respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    config_with_map = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "checkpoint_map": {"ns": "snap-abc"},  # must be discarded
        }
    }
    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    _saver().put(
        config_with_map,
        checkpoint,
        {"source": "loop", "step": 0, "writes": {}, "parents": {}},
        {},
    )

    body = json.loads(route.calls[0].request.content)
    assert "checkpointMap" not in body
    assert "checkpoint_map" not in body


@respx.mock
def test_checkpoint_ns_pipe_format_passes_through():
    """Pipe-separated subgraph checkpoint_ns passes through to snapshotScope unchanged."""
    route = respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    ns = "outer_node:uuid1|inner_node:uuid2"
    cfg = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": ns,
        }
    }
    checkpoint = {
        "id": SNAPSHOT_ID,
        "v": 1,
        "ts": "",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    _saver().put(
        cfg, checkpoint, {"source": "loop", "step": 0, "writes": {}, "parents": {}}, {}
    )

    body = json.loads(route.calls[0].request.content)
    assert body["snapshotScope"] == ns


@respx.mock
def test_put_writes_with_task_path_sets_write_key():
    route = respx.post(SNAPSHOT_WRITES_URL).mock(
        return_value=httpx.Response(200, json={})
    )

    _saver().put_writes(
        _config(checkpoint_id=SNAPSHOT_ID),
        [("ch", "val")],
        task_id="task-1",
        task_path="subgraph-path",
    )

    body = json.loads(route.calls[0].request.content)
    assert body["writeKey"] == "task-1:subgraph-path"


# ===========================================================================
# Security Tests - OData Injection Prevention
# ===========================================================================


def test_escape_odata_string_with_single_quote():
    """Single quote should be doubled for OData safety."""
    assert _escape_odata_string("abc'def") == "abc''def"


def test_escape_odata_string_with_sql_injection_attempt():
    """SQL injection patterns should be escaped safely."""
    malicious = "'; DROP TABLE snapshots--"
    escaped = _escape_odata_string(malicious)
    assert escaped == "''; DROP TABLE snapshots--"
    # Verify no unescaped single quotes
    assert "'" not in escaped.replace("''", "")


def test_escape_odata_string_with_odata_injection_attempt():
    """OData filter injection should be neutralized."""
    malicious = "abc' or threadId eq '"
    escaped = _escape_odata_string(malicious)
    assert escaped == "abc'' or threadId eq ''"


def test_escape_odata_string_with_multiple_quotes():
    """Multiple quotes should all be doubled."""
    assert _escape_odata_string("user's friend's data") == "user''s friend''s data"


def test_escape_odata_string_with_already_escaped():
    """Already escaped quotes should be double-escaped."""
    assert _escape_odata_string("user''s data") == "user''''s data"


def test_escape_odata_string_with_empty_string():
    """Empty string should remain empty."""
    assert _escape_odata_string("") == ""


def test_escape_odata_string_with_unicode():
    """Unicode with quotes should be handled correctly."""
    assert _escape_odata_string("用户's数据") == "用户''s数据"


def test_escape_odata_string_with_no_quotes():
    """String without quotes should be unchanged."""
    assert _escape_odata_string("normal_string_123") == "normal_string_123"


@respx.mock
def test_get_tuple_escapes_malicious_thread_id():
    """get_tuple should escape malicious thread_id in OData filter."""
    malicious_thread_id = "thread' or threadId eq '"

    # Mock empty response
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {
        "configurable": {
            "thread_id": malicious_thread_id,
            "checkpoint_ns": "",
        }
    }

    result = saver.get_tuple(config)

    assert result is None

    # Verify the request was made with escaped thread_id
    request = respx.calls[0].request
    url = str(request.url)
    # The single quotes should be doubled: thread' -> thread''
    # Then URL-encoded: ' -> %27
    # So "thread' or" becomes "thread'' or" which is URL-encoded to "thread%27%27%20or"
    assert "%27%27" in url  # Doubled single quotes (URL-encoded)
    # The OData filter should contain the escaped value
    assert "$filter" in url


@respx.mock
def test_get_tuple_escapes_malicious_checkpoint_id():
    """get_tuple should escape malicious checkpoint_id in OData key."""
    malicious_checkpoint_id = "snap' or '1'='1"
    thread_id = "safe-thread"

    # Mock the OData key URL using a pattern matcher
    respx.get(url__regex=r".*snapshots\(.*").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )
    saver = _saver()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": malicious_checkpoint_id,
        }
    }

    result = saver.get_tuple(config)

    assert result is None

    # Verify the malicious checkpoint_id was escaped in the URL
    request = respx.calls[0].request
    url = str(request.url)
    # Check that doubled single quotes appear in URL (may be URL-encoded)
    assert "''" in url or "%27%27" in url


@respx.mock
def test_odata_key_url_escapes_single_quote_before_url_encoding():
    """OData escaping must run before URL-encoding.

    For thread_id="user's", the correct sequence is:
      1. OData-escape: "user's" → "user''s"
      2. URL-encode:   "user''s" → "user%27%27s"

    The old (broken) order ran URL-encode first ("user%27s"), making the
    OData escape a no-op — the server would decode to "user's" and fail.
    """
    thread_id = "user's"
    respx.get(url__regex=r".*snapshots\(.*").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )

    saver = _saver()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": SNAPSHOT_ID,
        }
    }

    saver.get_tuple(config)

    url = str(respx.calls[0].request.url)
    # Doubled single quote URL-encoded: '' → %27%27
    assert "%27%27" in url
    # Single (un-doubled) encoded quote must NOT appear as the threadId value
    # i.e. "user%27s" without doubling would indicate wrong order
    assert "user%27%27s" in url


@respx.mock
def test_list_escapes_malicious_before_parameter():
    """list should escape malicious checkpoint_id in before parameter."""
    malicious_before_id = "snap'; DELETE FROM--"

    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    before_config = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": malicious_before_id,
        }
    }

    results = list(saver.list(config, before=before_config))

    assert results == []

    # Verify request was made
    request = respx.calls[0].request
    url = str(request.url)
    # The before parameter should be escaped using snapshotId lt 'escaped-value' in OData filter
    # The single quote in the malicious string should be doubled
    assert "snapshotId%20lt%20%27snap%27%27%3B%20DELETE%20FROM--%27" in url


@respx.mock
def test_delete_thread_with_malicious_thread_id():
    """delete_thread should safely handle malicious thread_id."""
    malicious_thread_id = "thread'; DROP TABLE--"

    respx.post(DELETE_BY_THREAD_URL).mock(
        return_value=httpx.Response(200, json={"deleted": 0})
    )

    saver = _saver_with_tenant()

    # Should not raise exception
    saver.delete_thread(malicious_thread_id, tenant_id="t1")

    # Verify the request body contains the thread_id (JSON-escaped)
    request = respx.calls[0].request
    body = request.read().decode()
    assert malicious_thread_id in body


# ===========================================================================
# Token Authentication Error Handling Tests
# ===========================================================================


@respx.mock
def test_token_fetch_handles_401_unauthorized():
    """Token fetch should raise ValueError on 401."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="Failed to obtain OAuth2 token"):
        fetcher.get_sync(CachedToken())


@respx.mock
def test_token_fetch_handles_403_forbidden():
    """Token fetch should raise ValueError on 403."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(
        return_value=httpx.Response(403, json={"error": "forbidden"})
    )

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="Failed to obtain OAuth2 token"):
        fetcher.get_sync(CachedToken())


@respx.mock
def test_token_fetch_handles_network_timeout():
    """Token fetch should raise ValueError on network timeout."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(side_effect=httpx.TimeoutException("Connection timeout"))

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="OAuth2 token service unavailable"):
        fetcher.get_sync(CachedToken())


@respx.mock
def test_token_fetch_handles_connection_error():
    """Token fetch should raise ValueError on connection error."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(side_effect=httpx.ConnectError("Connection refused"))

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="OAuth2 token service unavailable"):
        fetcher.get_sync(CachedToken())


@respx.mock
def test_token_fetch_handles_invalid_json():
    """Token fetch should raise ValueError on malformed JSON response."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(return_value=httpx.Response(200, text="not json"))

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="OAuth2 token service unavailable"):
        fetcher.get_sync(CachedToken())


@respx.mock
def test_token_fetch_success_caches_token():
    """Successful token fetch should cache the token."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(
        return_value=httpx.Response(
            200, json={"access_token": "test-token-123", "expires_in": 3600}
        )
    )

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)
    cached = CachedToken()

    token1 = fetcher.get_sync(cached)
    assert token1 == "test-token-123"

    # Second call should use cache (no new HTTP request)
    token2 = fetcher.get_sync(cached)
    assert token2 == "test-token-123"
    assert len(respx.calls) == 1  # Only one HTTP request


def test_token_cache_without_auth_returns_none():
    """Token cache without credentials should return None."""
    fetcher = TokenFetcher(None, None, None, timeout=10.0)

    token = fetcher.get_sync(CachedToken())
    assert token is None


@pytest.mark.asyncio
@respx.mock
async def test_token_fetch_async_handles_401():
    """Async token fetch should raise ValueError on 401."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="Failed to obtain OAuth2 token"):
        await fetcher.get_async(CachedToken())


@pytest.mark.asyncio
@respx.mock
async def test_token_fetch_async_handles_timeout():
    """Async token fetch should raise ValueError on timeout."""
    token_url = "https://auth.example/oauth/token"
    respx.post(token_url).mock(side_effect=httpx.TimeoutException("Timeout"))

    fetcher = TokenFetcher(token_url, "client-id", "client-secret", timeout=10.0)

    with pytest.raises(ValueError, match="OAuth2 token service unavailable"):
        await fetcher.get_async(CachedToken())


@pytest.mark.asyncio
async def test_token_cache_async_without_auth_returns_none():
    """Async token cache without credentials should return None."""
    fetcher = TokenFetcher(None, None, None, timeout=10.0)

    token = await fetcher.get_async(CachedToken())
    assert token is None


# ===========================================================================
# JSON Decode Error Handling Tests
# ===========================================================================


@respx.mock
def test_get_tuple_raises_on_malformed_checkpoint_json():
    """get_tuple should raise RuntimeError on malformed JSON in checkpoint response."""
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, text='{"incomplete json')
    )

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    with pytest.raises(RuntimeError, match="not valid JSON"):
        saver.get_tuple(config)


@respx.mock
def test_get_tuple_sends_expand_writes_query():
    """get_tuple (latest) must include $expand=writes in the snapshot request URL."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    _saver().get_tuple(_config())

    url = str(respx.calls[0].request.url)
    assert "%24expand=writes" in url or "$expand=writes" in url


@respx.mock
def test_get_tuple_specific_checkpoint_sends_expand_writes():
    """get_tuple (by id) must append ?$expand=writes to the OData key URL."""
    row = _make_snapshot()
    respx.get(url__regex=r".*snapshots\(.*").mock(
        return_value=httpx.Response(200, json=row)
    )

    _saver().get_tuple(_config(checkpoint_id=SNAPSHOT_ID))

    url = str(respx.calls[0].request.url)
    assert "expand=writes" in url


@pytest.mark.asyncio
@respx.mock
async def test_aget_tuple_sends_expand_writes_query():
    """aget_tuple (latest) must include $expand=writes in the snapshot request URL."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    await _saver().aget_tuple(_config())

    url = str(respx.calls[0].request.url)
    assert "%24expand=writes" in url or "$expand=writes" in url


@respx.mock
def test_list_sends_expand_writes_query():
    """list() must include $expand=writes in the snapshot request URL."""
    rows = [_make_snapshot(snapshot_id="snap-1")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    list(_saver().list(_config()))

    url = str(respx.calls[0].request.url)
    assert "%24expand=writes" in url or "$expand=writes" in url


@pytest.mark.asyncio
@respx.mock
async def test_alist_sends_expand_writes_query():
    """alist() must include $expand=writes in the snapshot request URL."""
    rows = [_make_snapshot(snapshot_id="snap-1")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    _ = [t async for t in _saver().alist(_config())]

    url = str(respx.calls[0].request.url)
    assert "%24expand=writes" in url or "$expand=writes" in url


# ===========================================================================
# Retry Logic and Error Handling Tests
# ===========================================================================


@respx.mock
def test_put_retries_on_503_service_unavailable():
    """put should retry on 503 service unavailable."""
    route = respx.post(SNAPSHOTS_URL).mock(
        side_effect=[
            httpx.Response(503, text="Service unavailable"),
            httpx.Response(200, json={"snapshotId": SNAPSHOT_ID}),
        ]
    )

    checkpoint = {
        "id": SNAPSHOT_ID,
        "channel_values": {},
        "v": 1,
        "ts": "2026-01-01T00:00:00Z",
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }

    result = _saver().put(_config(), checkpoint, {}, {})

    assert result["configurable"]["checkpoint_id"] == SNAPSHOT_ID
    assert route.call_count == 2


@respx.mock
def test_put_retries_on_timeout():
    """put should retry on timeout exceptions."""
    route = respx.post(SNAPSHOTS_URL).mock(
        side_effect=[
            httpx.TimeoutException("Connection timeout"),
            httpx.Response(200, json={"snapshotId": SNAPSHOT_ID}),
        ]
    )

    checkpoint = {
        "id": SNAPSHOT_ID,
        "channel_values": {},
        "v": 1,
        "ts": "2026-01-01T00:00:00Z",
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }

    result = _saver().put(_config(), checkpoint, {}, {})

    assert result["configurable"]["checkpoint_id"] == SNAPSHOT_ID
    assert route.call_count == 2


@respx.mock
def test_put_fails_after_max_retries():
    """put should fail after max retries exhausted."""
    route = respx.post(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(503, text="Service unavailable")
    )

    checkpoint = {
        "id": SNAPSHOT_ID,
        "channel_values": {},
        "v": 1,
        "ts": "2026-01-01T00:00:00Z",
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }

    with pytest.raises(httpx.HTTPStatusError):
        _saver().put(_config(), checkpoint, {}, {})

    # Should try max_retries + 1 times (default is 3 retries + initial = 4 total)
    assert route.call_count == 4


@respx.mock
def test_get_tuple_handles_connection_error():
    """get_tuple should propagate connection errors."""
    respx.get(SNAPSHOTS_URL).mock(side_effect=httpx.ConnectError("Connection refused"))

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    with pytest.raises(httpx.ConnectError):
        saver.get_tuple(config)


@pytest.mark.asyncio
@respx.mock
async def test_aget_tuple_handles_timeout():
    """aget_tuple should propagate timeout errors."""
    respx.get(SNAPSHOTS_URL).mock(side_effect=httpx.TimeoutException("Timeout"))

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    with pytest.raises(httpx.TimeoutException):
        await saver.aget_tuple(config)


# ===========================================================================
# Edge Cases Tests
# ===========================================================================


@respx.mock
def test_get_tuple_with_empty_thread_id():
    """get_tuple should handle empty thread_id."""
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": "", "checkpoint_ns": ""}}

    result = saver.get_tuple(config)

    # Should work with empty thread_id (OData escaping handles it)
    assert result is None


@respx.mock
def test_get_tuple_with_unicode_thread_id():
    """get_tuple should handle unicode thread_id."""
    unicode_thread_id = "用户-线程-123"

    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": unicode_thread_id, "checkpoint_ns": ""}}

    result = saver.get_tuple(config)

    assert result is None

    # Verify unicode was handled correctly
    request = respx.calls[0].request
    url = str(request.url)
    # Unicode should be URL-encoded or present in URL
    assert "threadId" in url


@respx.mock
def test_get_tuple_with_special_characters_in_thread_id():
    """get_tuple should handle special characters in thread_id."""
    special_thread_id = "thread@#$%^&*()"

    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": special_thread_id, "checkpoint_ns": ""}}

    result = saver.get_tuple(config)

    assert result is None


@respx.mock
def test_list_with_limit_zero():
    """list with limit=0 should return empty iterator."""
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    results = list(saver.list(config, limit=0))

    assert results == []


@respx.mock
def test_list_with_no_checkpoints():
    """list should handle no checkpoints gracefully."""
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    results = list(saver.list(config))

    assert results == []


# ---------------------------------------------------------------------------
# Credential resolution — env var fallback
# ---------------------------------------------------------------------------

_ALL_REQUIRED_ENVS = {
    "AGENT_MEMORY_BASE_URL": BASE_URL,
    "AGENT_MEMORY_TOKEN_URL": TOKEN_URL,
    "AGENT_MEMORY_CLIENT_ID": CLIENT_ID,
    "AGENT_MEMORY_CLIENT_SECRET": CLIENT_SECRET,
}


def test_init_raises_when_any_credential_missing(monkeypatch):
    """XSUAA: all three XSUAA creds required — missing any raises ValueError."""
    for missing_var in _ALL_REQUIRED_ENVS:
        if missing_var == "AGENT_MEMORY_BASE_URL":
            continue  # base_url missing is a separate error path
        for var, val in _ALL_REQUIRED_ENVS.items():
            monkeypatch.setenv(var, val)
        monkeypatch.delenv(missing_var)
        with pytest.raises(ValueError, match=missing_var.replace("AGENT_MEMORY_", "")):
            HanaAgentMemorySaver(auth=XSUAA())


def test_init_requires_auth():
    """auth is a required keyword — omitting it raises TypeError."""
    with pytest.raises(TypeError):
        HanaAgentMemorySaver(base_url=BASE_URL)  # type: ignore[call-arg]


def test_init_rejects_non_auth_object():
    """auth must be an IAS()/XSUAA() instance — a bare string is rejected."""
    with pytest.raises(ValueError, match="must be an IAS\\(\\) or XSUAA\\(\\)"):
        HanaAgentMemorySaver(base_url=BASE_URL, auth="XSUAA")  # type: ignore[arg-type]


def test_init_uses_all_env_credentials(monkeypatch):
    """XSUAA creds resolve from env vars when no cred args are passed."""
    for var, val in _ALL_REQUIRED_ENVS.items():
        monkeypatch.setenv(var, val)
    saver = HanaAgentMemorySaver(auth=XSUAA())
    assert saver._base == BASE_URL.rstrip("/")


def test_init_explicit_args_win_over_env(monkeypatch):
    """Explicit XSUAA(...) args take priority over env vars."""
    other_url = "https://from-env.example"
    monkeypatch.setenv("AGENT_MEMORY_BASE_URL", other_url)
    monkeypatch.setenv("AGENT_MEMORY_TOKEN_URL", "https://from-env.example/token")
    monkeypatch.setenv("AGENT_MEMORY_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AGENT_MEMORY_CLIENT_SECRET", "env-client-secret")
    saver = HanaAgentMemorySaver(
        base_url=BASE_URL,
        auth=XSUAA(
            token_url=TOKEN_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        ),
    )
    assert saver._base == BASE_URL.rstrip("/")


# ---------------------------------------------------------------------------
# IAS token — config["configurable"]["ias_access_token"] forwarding
# ---------------------------------------------------------------------------


def test_init_base_url_only_allowed():
    """IAS() with only base_url succeeds — IAS requires no XSUAA creds."""
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS())
    assert saver._base == BASE_URL.rstrip("/")


def test_init_partial_xsuaa_creds_raises(monkeypatch):
    """XSUAA() with only some creds resolving should raise."""
    for var in (
        "AGENT_MEMORY_TOKEN_URL",
        "AGENT_MEMORY_CLIENT_ID",
        "AGENT_MEMORY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="XSUAA requires"):
        HanaAgentMemorySaver(base_url=BASE_URL, auth=XSUAA(token_url=TOKEN_URL))


def test_init_ias_default_token(monkeypatch):
    """IAS(ias_access_token=...) sets a default token on the saver."""
    monkeypatch.delenv("AGENT_MEMORY_IAS_ACCESS_TOKEN", raising=False)
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS(ias_access_token="tok"))
    assert saver._default_ias_token == "tok"


def test_init_ias_default_token_from_env(monkeypatch):
    """IAS() falls back to AGENT_MEMORY_IAS_ACCESS_TOKEN for its default token."""
    monkeypatch.setenv("AGENT_MEMORY_IAS_ACCESS_TOKEN", "env-tok")
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS())
    assert saver._default_ias_token == "env-tok"


def test_init_ias_only_base_url(monkeypatch):
    """IAS() with only AGENT_MEMORY_BASE_URL in env → IAS mode."""
    monkeypatch.setenv("AGENT_MEMORY_BASE_URL", BASE_URL)
    saver = HanaAgentMemorySaver(auth=IAS())
    assert saver._is_ias is True


def test_init_env_all_xsuaa_creds_is_xsuaa(monkeypatch):
    """XSUAA() with all four AGENT_MEMORY_* env vars set → XSUAA mode."""
    monkeypatch.setenv("AGENT_MEMORY_BASE_URL", BASE_URL)
    monkeypatch.setenv("AGENT_MEMORY_TOKEN_URL", TOKEN_URL)
    monkeypatch.setenv("AGENT_MEMORY_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("AGENT_MEMORY_CLIENT_SECRET", CLIENT_SECRET)

    saver = HanaAgentMemorySaver(auth=XSUAA())

    assert saver._is_ias is False


@respx.mock
def test_ias_access_token_in_config_used_as_bearer(monkeypatch):
    """IAS saver: ias_access_token in config["configurable"] should be sent as Authorization: Bearer."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    result = saver.get_tuple(config)

    assert result is not None
    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer my-ias-token"


@respx.mock
def test_ias_access_token_absent_uses_cached_xsuaa_token():
    """When ias_access_token is absent, the pre-seeded XSUAA token should be used."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _saver()
    result = saver.get_tuple(_config())

    assert result is not None
    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer test-token"


@respx.mock
def test_put_propagates_ias_access_token_in_returned_config(monkeypatch):
    """IAS saver: put() should include ias_access_token in its returned configurable dict."""
    respx.post(SNAPSHOTS_URL).mock(return_value=httpx.Response(201, json={}))

    saver = _ias_saver(monkeypatch)
    from langgraph.checkpoint.base import empty_checkpoint

    checkpoint = empty_checkpoint()
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    result = saver.put(config, checkpoint, {}, {})

    assert result["configurable"].get("ias_access_token") == "my-ias-token"


@respx.mock
def test_delete_thread_with_ias_access_token_sends_bearer_header(monkeypatch):
    """IAS saver: delete_thread(ias_access_token=...) should send Authorization: Bearer."""
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(204))

    saver = _ias_saver(monkeypatch)
    saver.delete_thread(THREAD_ID, ias_access_token="my-ias-token")

    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer my-ias-token"


@pytest.mark.asyncio
@respx.mock
async def test_adelete_thread_with_ias_access_token_sends_bearer_header(monkeypatch):
    """IAS saver: adelete_thread(ias_access_token=...) should send Authorization: Bearer."""
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(204))

    saver = _ias_saver(monkeypatch)
    await saver.adelete_thread(THREAD_ID, ias_access_token="my-ias-token")

    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer my-ias-token"


# ---------------------------------------------------------------------------
# Per-request auth validation (IAS vs XSUAA)
# ---------------------------------------------------------------------------


def test_ias_only_saver_without_ias_access_token_raises(monkeypatch):
    """IAS-only saver (no XSUAA creds) invoked without ias_access_token must fail loud."""
    for var in (
        "AGENT_MEMORY_TOKEN_URL",
        "AGENT_MEMORY_CLIENT_ID",
        "AGENT_MEMORY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS())
    config = {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": SNAPSHOT_SCOPE}}

    with pytest.raises(ValueError, match="hana-agent-memory-serverless"):
        saver.get_tuple(config)


@respx.mock
def test_xsuaa_without_tenant_id_logs_debug(caplog):
    """XSUAA saver with no tenant_id logs at DEBUG about the provider tenant on every call."""
    respx.get(SNAPSHOTS_URL).mock(return_value=httpx.Response(200, json={"value": []}))

    saver = _saver()
    with caplog.at_level("DEBUG"):
        saver.get_tuple(_config())
        saver.get_tuple(_config())

    tenant_logs = [r for r in caplog.records if "provider tenant" in r.message]
    assert len(tenant_logs) == 2


@respx.mock
def test_ias_access_token_ignored_on_xsuaa_saver():
    """XSUAA saver: ias_access_token in config must be silently ignored; XSUAA token used instead."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _saver()
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "should-be-ignored",
        }
    }
    result = saver.get_tuple(config)

    assert result is not None
    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer test-token"
    assert auth_header != "Bearer should-be-ignored"


@respx.mock
def test_xsuaa_with_tenant_id_suppresses_provider_warning(caplog):
    """XSUAA saver with tenant_id present must not fire the provider-tenant warning."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _saver()
    # Pre-seed the per-tenant cache so no real token fetch is attempted.
    tenant_cache = saver._token_registry.get("t1")
    tenant_cache._token = "test-token"
    tenant_cache._expires_at = time.monotonic() + 3600

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "tenant_id": "t1",
        }
    }
    with caplog.at_level("DEBUG"):
        saver.get_tuple(config)

    assert not [r for r in caplog.records if "provider tenant" in r.message]


def test_delete_thread_ias_only_saver_without_token_raises(monkeypatch):
    """delete_thread parity: IAS-only saver with no ias_access_token must raise."""
    for var in (
        "AGENT_MEMORY_TOKEN_URL",
        "AGENT_MEMORY_CLIENT_ID",
        "AGENT_MEMORY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS())

    with pytest.raises(ValueError, match="hana-agent-memory-serverless"):
        saver.delete_thread(THREAD_ID)


@respx.mock
def test_delete_thread_xsuaa_without_tenant_id_logs_debug(caplog):
    """XSUAA saver: delete_thread without tenant_id logs debug and uses provider token."""
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(204))

    saver = _saver()
    with caplog.at_level("DEBUG"):
        saver.delete_thread(THREAD_ID)

    assert any("provider tenant" in r.message for r in caplog.records)
    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer test-token"


@pytest.mark.asyncio
async def test_adelete_thread_ias_only_saver_without_token_raises(monkeypatch):
    """adelete_thread parity: IAS-only saver with no ias_access_token must raise."""
    for var in (
        "AGENT_MEMORY_TOKEN_URL",
        "AGENT_MEMORY_CLIENT_ID",
        "AGENT_MEMORY_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    saver = HanaAgentMemorySaver(base_url=BASE_URL, auth=IAS())

    with pytest.raises(ValueError, match="hana-agent-memory-serverless"):
        await saver.adelete_thread(THREAD_ID)


@pytest.mark.asyncio
@respx.mock
async def test_adelete_thread_xsuaa_without_tenant_id_logs_debug(caplog):
    """XSUAA saver: adelete_thread without tenant_id logs debug and uses provider token."""
    respx.post(DELETE_BY_THREAD_URL).mock(return_value=httpx.Response(204))

    saver = _saver()
    with caplog.at_level("DEBUG"):
        await saver.adelete_thread(THREAD_ID)

    assert any("provider tenant" in r.message for r in caplog.records)
    auth_header = respx.calls[0].request.headers.get("authorization", "")
    assert auth_header == "Bearer test-token"


# ---------------------------------------------------------------------------
# Auth propagation in returned configs from get_tuple / list / aget_tuple / alist
# ---------------------------------------------------------------------------


@respx.mock
def test_get_tuple_propagates_ias_access_token_in_returned_config(monkeypatch):
    """IAS saver: get_tuple should include ias_access_token in its returned config."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    result = saver.get_tuple(config)

    assert result is not None
    assert result.config["configurable"].get("ias_access_token") == "my-ias-token"


@respx.mock
def test_get_tuple_propagates_ias_access_token_in_parent_config(monkeypatch):
    """IAS saver: get_tuple should include ias_access_token in parent_config too."""
    row = _make_snapshot(parent_snapshot_id=PARENT_SNAPSHOT_ID)
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    result = saver.get_tuple(config)

    assert result is not None
    assert result.parent_config is not None
    assert (
        result.parent_config["configurable"].get("ias_access_token") == "my-ias-token"
    )


@respx.mock
def test_get_tuple_propagates_tenant_id_in_returned_config():
    """XSUAA saver: get_tuple should include tenant_id in its returned config."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _saver()
    saver._token_registry.get("zone-abc")._token = "t"
    saver._token_registry.get("zone-abc")._expires_at = time.monotonic() + 3600
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "tenant_id": "zone-abc",
        }
    }

    result = saver.get_tuple(config)

    assert result is not None
    assert result.config["configurable"].get("tenant_id") == "zone-abc"


@respx.mock
def test_get_tuple_propagates_tenant_id_in_parent_config():
    """XSUAA saver: get_tuple should include tenant_id in parent_config."""
    row = _make_snapshot(parent_snapshot_id=PARENT_SNAPSHOT_ID)
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _saver()
    saver._token_registry.get("zone-abc")._token = "t"
    saver._token_registry.get("zone-abc")._expires_at = time.monotonic() + 3600
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "tenant_id": "zone-abc",
        }
    }

    result = saver.get_tuple(config)

    assert result is not None
    assert result.parent_config is not None
    assert result.parent_config["configurable"].get("tenant_id") == "zone-abc"


@respx.mock
def test_list_propagates_ias_access_token_in_returned_configs(monkeypatch):
    """IAS saver: list() should include ias_access_token in every yielded config."""
    rows = [_make_snapshot(snapshot_id="snap-1"), _make_snapshot(snapshot_id="snap-2")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    results = list(saver.list(config))

    assert len(results) == 2
    for tup in results:
        assert tup.config["configurable"].get("ias_access_token") == "my-ias-token"


@respx.mock
def test_list_propagates_tenant_id_in_returned_configs():
    """XSUAA saver: list() should include tenant_id in every yielded config."""
    rows = [_make_snapshot(snapshot_id="snap-1"), _make_snapshot(snapshot_id="snap-2")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    saver = _saver()
    saver._token_registry.get("zone-abc")._token = "t"
    saver._token_registry.get("zone-abc")._expires_at = time.monotonic() + 3600
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "tenant_id": "zone-abc",
        }
    }

    results = list(saver.list(config))

    assert len(results) == 2
    for tup in results:
        assert tup.config["configurable"].get("tenant_id") == "zone-abc"


@pytest.mark.asyncio
@respx.mock
async def test_aget_tuple_propagates_ias_access_token_in_returned_config(monkeypatch):
    """IAS saver: aget_tuple should include ias_access_token in its returned config."""
    row = _make_snapshot()
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": [row]})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    result = await saver.aget_tuple(config)

    assert result is not None
    assert result.config["configurable"].get("ias_access_token") == "my-ias-token"


@pytest.mark.asyncio
@respx.mock
async def test_alist_propagates_ias_access_token_in_returned_configs(monkeypatch):
    """IAS saver: alist() should include ias_access_token in every yielded config."""
    rows = [_make_snapshot(snapshot_id="snap-1"), _make_snapshot(snapshot_id="snap-2")]
    respx.get(SNAPSHOTS_URL).mock(
        return_value=httpx.Response(200, json={"value": rows})
    )

    saver = _ias_saver(monkeypatch)
    config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": SNAPSHOT_SCOPE,
            "ias_access_token": "my-ias-token",
        }
    }

    results = [t async for t in saver.alist(config)]

    assert len(results) == 2
    for tup in results:
        assert tup.config["configurable"].get("ias_access_token") == "my-ias-token"
