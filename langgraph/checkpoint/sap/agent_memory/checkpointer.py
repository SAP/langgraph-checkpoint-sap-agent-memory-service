"""LangGraph checkpoint saver backed by the SAP HANA Agent Memory Service."""

from __future__ import annotations

import base64
import builtins
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from uuid6 import uuid7

from langgraph.checkpoint.sap.agent_memory._http import (
    CachedToken,
    RetryClient,
    SyncRetryClient,
    TokenFetcher,
    TokenRegistry,
    collect_all_pages_async,
    collect_all_pages_sync,
)

logger = logging.getLogger(__name__)

_KNOWN_CONFIGURABLE_KEYS = {
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "tenant_id",
    "ias_access_token",
}


def _b64enc(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _b64dec(s: str) -> bytes:
    return base64.b64decode(s)


def _escape_odata_string(value: str) -> str:
    """Escape single quotes for OData string literals.

    OData v4 spec requires single quotes in string literals to be doubled.
    This prevents injection attacks when building OData filter expressions.

    Examples:
        "abc" -> "abc"
        "user's data" -> "user''s data"
        "'; DROP TABLE--" -> "''; DROP TABLE--"

    Args:
        value: The string value to escape

    Returns:
        Escaped string safe for use in OData filter expressions
    """
    return value.replace("'", "''")


class HanaAgentMemorySaver(BaseCheckpointSaver[str]):
    """LangGraph checkpoint saver backed by the SAP HANA Agent Memory Service.

    ``tenant_id`` and ``ias_access_token`` are passed per-invocation via
    ``config["configurable"]`` for multi-tenant and IAS-authenticated routing::

        saver = HanaAgentMemorySaver(
            base_url="https://<service-host>",
            token_url="https://<tenant>.authentication.<region>/oauth/token",
            client_id="<client-id>",
            client_secret="<client-secret>",
        )
        graph = builder.compile(checkpointer=saver)

        config = {
            "configurable": {
                "thread_id": context_id,
                "tenant_id": "<btp-subaccount-zone-id>",          # optional - XSUAA savers only. fallback to provider if absent
                "ias_access_token": "<ias-bearer-token>",         # IAS savers only
            }
        }
        result = await graph.ainvoke({"messages": [...]}, config=config)

    **Saver kind is fixed at construction and immutable:**

    The saver is either an **IAS saver** or a **XSUAA saver** depending on
    the credentials supplied to the constructor. This cannot change at runtime.

    - ``HanaAgentMemorySaver(base_url=...)`` — **IAS saver**: no XSUAA
      credentials. Each request must supply ``ias_access_token`` in
      ``config["configurable"]``; the token is used as a Bearer token directly.
    - ``HanaAgentMemorySaver(base_url, token_url, client_id, client_secret)``
      — **XSUAA saver**: ``client_credentials`` flow. Any ``ias_access_token`` in
      ``config["configurable"]`` is ignored. Optionally pass ``tenant_id`` for
      multi-tenant routing (omitting it uses the provider tenant with a warning).

    **Tenant routing (XSUAA savers only):**

    Pass ``tenant_id`` in ``config["configurable"]`` to route the token request
    with an ``x-zid`` header for that BTP subaccount zone ID.

    Credentials are resolved hierarchically: explicit constructor arguments take
    priority; any ``None`` parameter falls back to the corresponding environment
    variable:

    - ``AGENT_MEMORY_BASE_URL``    → ``base_url``
    - ``AGENT_MEMORY_TOKEN_URL``   → ``token_url``
    - ``AGENT_MEMORY_CLIENT_ID``   → ``client_id``
    - ``AGENT_MEMORY_CLIENT_SECRET`` → ``client_secret``

    ``base_url`` is always required. XSUAA credentials are all-or-none: omit all
    three for IAS-only deployments, or provide all three for XSUAA::

        saver = HanaAgentMemorySaver()  # reads all from env vars
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 30.0,
        serde: Any | None = None,
    ) -> None:
        """Initialise the saver.

        Credentials are resolved hierarchically: explicit arguments win;
        ``None`` parameters fall back to environment variables.

        ``base_url`` is always required. XSUAA credentials (``token_url``,
        ``client_id``, ``client_secret``) are optional — omit all three for IAS
        mode, where the per-request token is passed via
        ``config["configurable"]["ias_access_token"]``. If any XSUAA credential is
        provided, all three must be present.

        Args:
            base_url: Agent Memory Service base URL, e.g.
                ``"https://<service-host>"``. Falls back to
                ``AGENT_MEMORY_BASE_URL`` env var.
            token_url: OAuth2 token endpoint (``client_credentials`` grant).
                Falls back to ``AGENT_MEMORY_TOKEN_URL``.
            client_id: OAuth2 client ID. Falls back to
                ``AGENT_MEMORY_CLIENT_ID``.
            client_secret: OAuth2 client secret. Falls back to
                ``AGENT_MEMORY_CLIENT_SECRET``.
            timeout: HTTP request timeout in seconds. Default ``30.0``.
            serde: Custom serializer. Defaults to ``JsonPlusSerializer``.
        """
        # Source-layer resolution: explicit kwargs win as a unit; env vars are
        # consulted only when no explicit base_url is provided.  Never mix
        # sources — that would allow ambient env vars to silently flip the auth
        # mode when a caller passes only base_url (Rule 1: base_url-only = IAS).
        if base_url:
            resolved_base_url = base_url
            resolved_token_url = token_url
            resolved_client_id = client_id
            resolved_client_secret = client_secret
        else:
            resolved_base_url = os.getenv("AGENT_MEMORY_BASE_URL")
            resolved_token_url = os.getenv("AGENT_MEMORY_TOKEN_URL")
            resolved_client_id = os.getenv("AGENT_MEMORY_CLIENT_ID")
            resolved_client_secret = os.getenv("AGENT_MEMORY_CLIENT_SECRET")

        if not resolved_base_url:
            raise ValueError(
                "HanaAgentMemorySaver requires base_url / AGENT_MEMORY_BASE_URL."
            )

        xsuaa_creds = (resolved_token_url, resolved_client_id, resolved_client_secret)
        self._is_ias = not all(xsuaa_creds)  # True = IAS-only binding, False = XSUAA
        if any(xsuaa_creds) and not all(xsuaa_creds):
            missing = [
                name
                for name, value in (
                    ("token_url / AGENT_MEMORY_TOKEN_URL", resolved_token_url),
                    ("client_id / AGENT_MEMORY_CLIENT_ID", resolved_client_id),
                    (
                        "client_secret / AGENT_MEMORY_CLIENT_SECRET",
                        resolved_client_secret,
                    ),
                )
                if not value
            ]
            raise ValueError(
                "HanaAgentMemorySaver: partial XSUAA credentials provided — "
                "all three (token_url, client_id, client_secret) must be set together: "
                + ", ".join(missing)
            )

        logger.info(
            "HanaAgentMemorySaver: auth mode = %s",
            "IAS" if self._is_ias else "XSUAA",
        )
        super().__init__(serde=serde or JsonPlusSerializer())
        self._base = resolved_base_url.rstrip("/")
        self._timeout = timeout

        self._token_fetcher = TokenFetcher(
            resolved_token_url,
            resolved_client_id,
            resolved_client_secret,
            timeout,
        )
        self._token_registry = TokenRegistry()
        # Tests pre-seed _token_cache._token to skip real token fetches.
        self._token_cache = self._token_registry.get()

        self._retry = RetryClient(
            self._token_fetcher,
        )
        self._sync_retry = SyncRetryClient(
            self._token_fetcher,
        )

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _snapshots_url(self) -> str:
        return f"{self._base}/v1/agent/snapshots"

    def _snapshot_writes_url(self) -> str:
        return f"{self._base}/v1/agent/snapshotWrites"

    def _delete_by_thread_url(self) -> str:
        return f"{self._base}/v1/agent/snapshots/deleteByThread"

    @staticmethod
    def _build_url(base: str, params: dict[str, str]) -> str:
        """Build a URL keeping OData ``$``-prefixed params un-encoded.

        httpx's ``params=`` dict percent-encodes ``$`` → ``%24``, which makes
        OData system query options like ``$top`` and ``$orderby`` unrecognised
        by the service (400 Bad Request).  Plain params (``threadId``, etc.)
        are encoded normally via ``urlencode``.
        """
        plain = {k: v for k, v in params.items() if not k.startswith("$")}
        odata = {k: v for k, v in params.items() if k.startswith("$")}
        parts: list[str] = []
        if plain:
            parts.append(urlencode(plain, quote_via=quote))
        for k, v in odata.items():
            # Only preserve commas in $orderby (e.g., "field1 desc,field2 asc")
            # Fully encode everything else ($filter, $top, $skip, etc.)
            if k == "$orderby":
                parts.append(f"{k}={quote(v, safe=',')}")
            else:
                parts.append(f"{k}={quote(v)}")  # No safe chars - encode everything
        if not parts:
            return base
        return f"{base}?{'&'.join(parts)}"

    @staticmethod
    def _build_odata_key_url(base: str, keys: dict[str, str]) -> str:
        """Build OData URL with key syntax: /entity(key1='val1',key2='val2')

        Used for accessing specific entities by their composite key.
        Example: /snapshot(threadId='abc',snapshotScope='',snapshotId='xyz')
        """
        key_parts = [
            f"{k}='{quote(_escape_odata_string(v), safe='')}'" for k, v in keys.items()
        ]
        return f"{base}({','.join(key_parts)})"

    def _build_snapshot_filter(
        self,
        thread_id: str | None,
        snapshot_scope: str | None,
        before_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str | None:
        """Build OData $filter for snapshot queries with proper escaping.

        Args:
            thread_id: Optional thread filter (None = list all threads)
            snapshot_scope: Optional namespace filter (None = list all namespaces)
            before_id: Optional cursor for pagination (filters to snapshotId < before_id)
            checkpoint_id: Optional single checkpoint filter (point query)

        Returns:
            OData filter string, or None if no filters (list everything)
        """
        parts = []

        if thread_id is not None:
            parts.append(f"threadId eq '{_escape_odata_string(thread_id)}'")

        if snapshot_scope is not None:
            parts.append(f"snapshotScope eq '{_escape_odata_string(snapshot_scope)}'")

        if checkpoint_id is not None:
            parts.append(f"snapshotId eq '{_escape_odata_string(checkpoint_id)}'")

        if before_id is not None:
            parts.append(f"snapshotId lt '{_escape_odata_string(before_id)}'")

        return " and ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @property
    def config_specs(self) -> builtins.list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="tenant_id",
                annotation=str,
                name="Tenant ID",
                description="BTP subaccount zone ID (x_zid) for multi-tenant routing. "
                "Omit for single-tenant (no x-zid header).",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="ias_access_token",
                annotation=str,
                name="IAS Token",
                description="Per-request IAS Bearer token for IAS-authenticated tenants. "
                "When present, used instead of XSUAA client_credentials.",
                default="",
                is_shared=True,
            ),
        ]

    def _extract_configurable(self, config: RunnableConfig) -> dict[str, str]:
        """Return only the known configurable keys; discard checkpoint_map etc."""
        cfg = config.get("configurable", {})
        return {k: v for k, v in cfg.items() if k in _KNOWN_CONFIGURABLE_KEYS}

    def _resolve_tenant_cache(self, cfg: dict[str, str]) -> CachedToken:
        """Return the CachedToken for this call.

        Uses config["configurable"]["tenant_id"] when provided; falls back to no x-zid (single-tenant).
        """
        tenant_id = cfg.get("tenant_id") or None
        if tenant_id is not None:
            return self._token_registry.get(tenant_id)
        return self._token_cache

    def _resolve_auth(self, cfg: dict[str, str]) -> tuple[str | None, CachedToken]:
        """Validate per-request auth and return ``(ias_access_token, token_cache)``.

        Branches on the saver's construction-time kind (``_is_ias``), which is
        immutable for the saver's lifetime:

        - **IAS saver** (``_is_ias=True``, no XSUAA credentials): ``ias_access_token``
          in ``cfg`` is mandatory. Raises if absent or whitespace-only.
        - **XSUAA saver** (``_is_ias=False``, full XSUAA credentials): ``ias_access_token``
          in ``cfg`` is ignored. Warns on every call if ``tenant_id`` is absent
          (provider tenant used instead of multi-tenant routing).

        Raises:
            ValueError: IAS saver invoked without an ``ias_access_token`` — would
                otherwise send an unauthenticated request.
        """
        if self._is_ias:
            raw = cfg.get("ias_access_token")
            ias_access_token = raw.strip() if raw else None
            if not ias_access_token:
                raise ValueError(
                    "No auth available: this saver was constructed with the "
                    "hana-agent-memory-serverless plan, so an IAS token is "
                    "required for this request."
                )
            return ias_access_token, self._token_cache  # cache unused for IAS

        # XSUAA saver: ignore any ias_access_token in cfg, validate tenant
        if not cfg.get("tenant_id"):
            logger.debug(
                "No tenant_id provided; using provider tenant. "
                "Set tenant_id for multi-tenant routing."
            )
        return None, self._resolve_tenant_cache(cfg)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _encode_checkpoint(self, checkpoint: Checkpoint) -> tuple[str, str]:
        """Returns (serialization_type, base64_state)."""
        type_, data = self.serde.dumps_typed(checkpoint)
        return type_, _b64enc(data)

    def _decode_checkpoint(self, serialization_type: str, encoded: str) -> Checkpoint:
        return self.serde.loads_typed((serialization_type, _b64dec(encoded)))

    def _encode_writes(self, writes: Sequence[tuple[str, Any]]) -> tuple[str, str]:
        """Pack writes into a payload blob; returns (serialization_type, base64_payload)."""
        payload_obj = [
            (WRITES_IDX_MAP.get(channel, idx), channel, value)
            for idx, (channel, value) in enumerate(writes)
        ]
        type_, data = self.serde.dumps_typed(payload_obj)
        return type_, _b64enc(data)

    def _write_key(self, task_id: str, task_path: str = "") -> str:
        return f"{task_id}:{task_path}" if task_path else task_id

    def _build_metadata_filter(self, filter: dict[str, Any]) -> str | None:
        """Build OData $filter expression for server-side metadata filtering.

        Converts dict filters like {"source": "input", "step": 2} into OData
        contains() clauses that match exact JSON key-value patterns.

        Example:
            {"source": "input"} → contains(metadata, '"source":"input"')
            {"source": "input", "step": 2} → contains(metadata, '"source":"input"') and contains(metadata, '"step":2')
        """
        if not filter:
            return None

        clauses = []
        for key, value in filter.items():
            try:
                json_pattern = f'"{key}":{json.dumps(value)}'
                # Escape single quotes for OData string literal
                escaped_pattern = json_pattern.replace("'", "''")
                clauses.append(f"contains(metadata, '{escaped_pattern}')")
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to serialize filter value for key '{key}': {e}. "
                    f"Value type: {type(value).__name__}, value: {value!r}"
                ) from e

        return " and ".join(clauses)

    # ------------------------------------------------------------------
    # Row → CheckpointTuple
    # ------------------------------------------------------------------

    def _row_to_tuple(
        self,
        row: dict[str, Any],
        *,
        ias_access_token: str | None = None,
        tenant_id: str | None = None,
    ) -> CheckpointTuple:
        checkpoint = self._decode_checkpoint(row["serializationType"], row["state"])
        metadata: CheckpointMetadata = row.get("metadata") or {}

        flattened: list[tuple[int, str, str, Any]] = []
        for w in row.get("writes", []):
            decoded = self.serde.loads_typed(
                (w["serializationType"], _b64dec(w["payload"]))
            )
            for write_index, channel, value in decoded:
                flattened.append((write_index, w["writeKey"], channel, value))

        flattened.sort(key=lambda t: t[0])
        pending_writes: list[tuple[str, str, Any]] = [
            (task_id, channel, value) for _idx, task_id, channel, value in flattened
        ]

        thread_id: str = row["threadId"]
        snapshot_scope: str = row.get("snapshotScope", "")

        parent_config: RunnableConfig | None = None
        parent_snapshot_id = row.get("parentSnapshotId")

        if parent_snapshot_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": snapshot_scope,
                    "checkpoint_id": parent_snapshot_id,
                    **({"tenant_id": tenant_id} if tenant_id else {}),
                    **(
                        {"ias_access_token": ias_access_token}
                        if ias_access_token
                        else {}
                    ),
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": snapshot_scope,
                    "checkpoint_id": row["snapshotId"],
                    **({"tenant_id": tenant_id} if tenant_id else {}),
                    **(
                        {"ias_access_token": ias_access_token}
                        if ias_access_token
                        else {}
                    ),
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    # ------------------------------------------------------------------
    # get_next_version
    # ------------------------------------------------------------------

    def get_next_version(self, current: str | None, channel: None) -> str:  # type: ignore[override]
        return str(uuid7())

    # ------------------------------------------------------------------
    # Internal helpers for reducing sync/async duplication
    # ------------------------------------------------------------------

    def _parse_snapshot_response(
        self,
        response: httpx.Response,
        checkpoint_id: str | None,
        thread_id: str,
        log_prefix: str = "GET checkpoint",
    ) -> dict[str, Any] | None:
        """Parse and validate snapshot response. Returns None if not found, raises on bad JSON."""
        # For key-based access (specific checkpoint_id), response is single object
        if checkpoint_id:
            try:
                row = response.json()
            except ValueError as e:
                raise RuntimeError(
                    f"Agent Memory Service snapshot response is not valid JSON — "
                    f"cannot restore checkpoint for thread '{thread_id}'. "
                ) from e
            if not row or not row.get("snapshotId"):
                logger.debug(
                    "%s: NOT FOUND thread_id=%s checkpoint_id=%s",
                    log_prefix,
                    thread_id,
                    checkpoint_id,
                )
                return None
            return row

        # For query-based access (latest), response has 'value' array
        try:
            rows = response.json().get("value", [])
        except ValueError as e:
            raise RuntimeError(
                f"Agent Memory Service snapshot response is not valid JSON — "
                f"cannot restore checkpoint for thread '{thread_id}'. "
            ) from e
        if not rows:
            logger.debug("%s: NOT FOUND thread_id=%s", log_prefix, thread_id)
            return None
        return rows[0]

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        thread_id = cfg["thread_id"]
        snapshot_scope = cfg.get("checkpoint_ns", "")
        checkpoint_id = cfg.get("checkpoint_id")

        logger.debug(
            "GET checkpoint: thread_id=%s checkpoint_id=%s ns=%s",
            thread_id,
            checkpoint_id or "(latest)",
            snapshot_scope or "(root)",
        )

        # If we have a specific checkpoint_id, use OData key syntax for direct access
        if checkpoint_id:
            # Use OData key syntax: /snapshot(threadId='x',snapshotScope='',snapshotId='y')
            url = self._build_odata_key_url(
                self._snapshots_url(),
                {
                    "threadId": thread_id,
                    "snapshotScope": snapshot_scope,
                    "snapshotId": checkpoint_id,
                },
            )
            url += "?$expand=writes"
        else:
            # Query for latest checkpoint using OData $filter
            # Must use $filter for proper query-based filtering
            params: dict[str, str] = {
                "$top": "1",
                "$orderby": "snapshotId desc",
                "$expand": "writes",
            }
            snapshot_filter = self._build_snapshot_filter(thread_id, snapshot_scope)
            if snapshot_filter:
                params["$filter"] = snapshot_filter
            url = self._build_url(self._snapshots_url(), params)

        try:
            r = self._sync_retry.get(
                url, token_cache=cache, ias_access_token=ias_access_token
            )
        except httpx.HTTPStatusError as e:
            # For key-based access, 404 means checkpoint doesn't exist
            if checkpoint_id and e.response.status_code == 404:
                logger.debug(
                    "GET checkpoint: NOT FOUND thread_id=%s checkpoint_id=%s",
                    thread_id,
                    checkpoint_id,
                )
                return None
            # For other errors or query-based access, re-raise
            raise

        # Parse snapshot response
        row = self._parse_snapshot_response(
            r, checkpoint_id, thread_id, "GET checkpoint"
        )
        if row is None:
            return None

        result = self._row_to_tuple(
            row,
            ias_access_token=ias_access_token,
            tenant_id=cfg.get("tenant_id"),
        )
        logger.debug(
            "GET checkpoint SUCCESS: thread_id=%s returned_id=%s has_parent=%s writes_count=%d",
            thread_id,
            result.checkpoint["id"],
            result.parent_config is not None,
            len(result.pending_writes or []),
        )
        return result

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints, optionally filtered.

        Args:
            config: Optional config with thread_id, checkpoint_ns, checkpoint_id.
                    - If None: lists across ALL threads (used by admin tooling)
                    - If checkpoint_ns key is missing: lists all namespaces for the thread
                    - If checkpoint_ns is explicitly "": lists only root namespace
                    - If checkpoint_id is provided: returns only that checkpoint (point query)
            filter: Optional metadata filter (exact equality, AND logic, server-side)
            before: Optional cursor for pagination (exclusive, keyset-based)
            limit: Optional maximum number of results to return

        Note:
            ``config=None`` (list all threads) is **not supported for IAS savers**.
            There is no mechanism to supply an ``ias_access_token`` without a config
            object, so the call will raise ``ValueError``.  XSUAA savers fall back
            to the provider tenant when ``config`` is ``None``.

        Yields:
            CheckpointTuple objects in DESC order by checkpoint_id (newest first).
            Ordering is GUARANTEED and required for pagination semantics.
        """
        if limit == 0:
            return

        # Support config=None for listing all threads (used by admin tooling)
        if config is None:
            cfg: dict[str, str] = {}
            thread_id = None
            snapshot_scope = None
            checkpoint_id = None
        else:
            cfg = self._extract_configurable(config)
            thread_id = cfg.get("thread_id")  # May be None for cross-thread queries

            # Distinguish missing checkpoint_ns from explicit ""
            # Missing key → list all namespaces, explicit "" → root only
            snapshot_scope = (
                cfg.get("checkpoint_ns") if "checkpoint_ns" in cfg else None
            )

            # Support point queries with checkpoint_id
            checkpoint_id = cfg.get("checkpoint_id")

        ias_access_token, cache = self._resolve_auth(cfg)

        before_id = None
        if before is not None:
            before_cfg = self._extract_configurable(before)
            before_id = before_cfg.get("checkpoint_id")

        logger.debug(
            "LIST checkpoints: thread_id=%s ns=%s limit=%s before=%s filter=%s checkpoint_id=%s",
            thread_id,
            snapshot_scope,
            limit,
            before_id,
            filter,
            checkpoint_id,
        )

        # Build snapshot filter
        base_filter = self._build_snapshot_filter(
            thread_id, snapshot_scope, before_id, checkpoint_id
        )
        filter_parts = [base_filter] if base_filter else []

        # Add metadata filter if provided
        if filter:
            metadata_filter = self._build_metadata_filter(filter)
            if metadata_filter:
                filter_parts.append(metadata_filter)

        # Always order DESC; request writes inline to avoid N+1 round-trips
        params: dict[str, str] = {"$orderby": "snapshotId desc", "$expand": "writes"}
        if filter_parts:
            params["$filter"] = " and ".join(filter_parts)

        # Server-side limit always applied
        if limit is not None:
            params["$top"] = str(limit)

        # When limit is set, one page is enough — $top enforces the cap server-side.
        # When unbounded, follow @odata.nextLink to collect all pages.
        first_url = self._build_url(self._snapshots_url(), params)
        if limit is not None:
            r = self._sync_retry.get(
                first_url, token_cache=cache, ias_access_token=ias_access_token
            )
            r.raise_for_status()
            rows = r.json().get("value", [])
        else:
            rows = collect_all_pages_sync(
                self._sync_retry,
                first_url,
                token_cache=cache,
                ias_access_token=ias_access_token,
            )

        for row in rows:
            yield self._row_to_tuple(
                row,
                ias_access_token=ias_access_token,
                tenant_id=cfg.get("tenant_id") if config is not None else None,
            )

        logger.debug(
            "LIST checkpoints SUCCESS: thread_id=%s fetched=%d",
            thread_id,
            len(rows),
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        thread_id = cfg["thread_id"]
        snapshot_scope = cfg.get("checkpoint_ns", "")
        serialization_type, encoded_state = self._encode_checkpoint(checkpoint)

        logger.debug(
            "PUT checkpoint: thread_id=%s checkpoint_id=%s ns=%s parent=%s",
            thread_id,
            checkpoint["id"],
            snapshot_scope or "(root)",
            cfg.get("checkpoint_id") or "(none)",
        )
        logger.debug("PUT checkpoint data: %d bytes serialized", len(encoded_state))

        body = {
            "threadId": thread_id,
            "snapshotScope": snapshot_scope,
            "snapshotId": checkpoint["id"],
            "parentSnapshotId": cfg.get("checkpoint_id"),
            "serializationType": serialization_type,
            "state": encoded_state,
            "metadata": metadata or {},
        }

        response = self._sync_retry.post(
            self._snapshots_url(),
            json=body,
            token_cache=cache,
            ias_access_token=ias_access_token,
        )
        logger.debug(
            "PUT checkpoint SUCCESS: thread_id=%s checkpoint_id=%s status=%d",
            thread_id,
            checkpoint["id"],
            response.status_code,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": snapshot_scope,
                "checkpoint_id": checkpoint["id"],
                **({"tenant_id": cfg["tenant_id"]} if cfg.get("tenant_id") else {}),
                **({"ias_access_token": ias_access_token} if ias_access_token else {}),
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return

        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        snapshot_id = cfg.get("checkpoint_id")
        if not snapshot_id:
            raise ValueError(
                "put_writes requires checkpoint_id in config but none was provided"
            )

        logger.debug(
            "PUT writes: thread_id=%s snapshot_id=%s task_id=%s task_path=%s writes_count=%d",
            cfg["thread_id"],
            snapshot_id,
            task_id,
            task_path or "(root)",
            len(writes),
        )

        serialization_type, encoded_payload = self._encode_writes(writes)
        logger.debug("PUT writes data: %d bytes serialized", len(encoded_payload))

        body = {
            "threadId": cfg["thread_id"],
            "snapshotScope": cfg.get("checkpoint_ns", ""),
            "snapshotId": snapshot_id,
            "writeKey": self._write_key(task_id, task_path),
            "serializationType": serialization_type,
            "payload": encoded_payload,
        }

        response = self._sync_retry.post(
            self._snapshot_writes_url(),
            json=body,
            token_cache=cache,
            ias_access_token=ias_access_token,
        )
        logger.debug(
            "PUT writes SUCCESS: thread_id=%s snapshot_id=%s task_id=%s status=%d",
            cfg["thread_id"],
            snapshot_id,
            task_id,
            response.status_code,
        )

    def delete_thread(
        self,
        thread_id: str,
        *,
        tenant_id: str | None = None,
        ias_access_token: str | None = None,
    ) -> None:
        """Delete all checkpoints and writes for a thread.

        Unlike ``put``, ``list``, and ``get_tuple`` — which LangGraph invokes
        automatically and supplies with a ``RunnableConfig`` — this method is
        called directly by application code (e.g.
        ``graph.checkpointer.delete_thread(thread_id)``).  LangGraph never
        calls it, so there is no ``config`` object to carry auth context.

        For that reason, ``tenant_id`` and ``ias_access_token`` are accepted as
        explicit keyword arguments instead of being read from
        ``config["configurable"]``.  The required argument depends on the
        saver kind set at construction time:

        - **IAS saver** (``base_url`` only): pass ``ias_access_token``.
        - **XSUAA saver** (full credentials): pass ``tenant_id``.

        ::

            # IAS saver
            saver.delete_thread(thread_id, ias_access_token="<bearer-token>")

            # XSUAA saver
            saver.delete_thread(thread_id, tenant_id="<zone-id>")

        The call is idempotent: deleting a thread that does not exist returns
        without error.

        Args:
            thread_id: ID of the thread whose checkpoints and writes to delete.
            tenant_id: BTP subaccount zone ID. Required for XSUAA savers.
            ias_access_token: IAS Bearer token. Required for IAS savers.
        """
        # POST to the deleteByThread unbound action on AgentSnapshotService.
        # OData v4 only permits DELETE by full key, so bulk-delete-by-thread is
        # exposed as an action on the backend (see agent-snapshot-service.cds).
        # 404 is in _NO_RETRY_STATUSES so the retry loop raises HTTPStatusError
        # for it, which we catch below as an idempotent no-op.
        logger.debug(
            "DELETE thread: thread_id=%s action=deleteByThread",
            thread_id,
        )
        if self._is_ias:
            if not (ias_access_token and ias_access_token.strip()):
                raise ValueError(
                    "No auth available: this saver was constructed with the "
                    "hana-agent-memory-serverless plan, so an IAS token is "
                    "required for this request."
                )
            resolved_ias_access_token, cache = ias_access_token, self._token_cache
        else:
            if not tenant_id:
                logger.debug(
                    "No tenant_id provided; using provider tenant. "
                    "Set tenant_id for multi-tenant routing."
                )
            resolved_ias_access_token, cache = (
                None,
                self._resolve_tenant_cache({"tenant_id": tenant_id or ""}),
            )

        try:
            r = self._sync_retry.post(
                self._delete_by_thread_url(),
                json={"threadId": thread_id},
                token_cache=cache,
                ias_access_token=resolved_ias_access_token,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(
                    "DELETE thread: NOT FOUND (already deleted) thread_id=%s status=404",
                    thread_id,
                )
                return
            raise
        logger.debug(
            "DELETE thread SUCCESS: thread_id=%s status=%d",
            thread_id,
            r.status_code,
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        thread_id = cfg["thread_id"]
        snapshot_scope = cfg.get("checkpoint_ns", "")
        checkpoint_id = cfg.get("checkpoint_id")

        logger.debug(
            "GET checkpoint (async): thread_id=%s checkpoint_id=%s ns='%s'",
            thread_id,
            checkpoint_id if checkpoint_id else "latest",
            snapshot_scope,
        )

        # If we have a specific checkpoint_id, use OData key syntax for direct access
        if checkpoint_id:
            # Use OData key syntax: /snapshot(threadId='x',snapshotScope='',snapshotId='y')
            url = self._build_odata_key_url(
                self._snapshots_url(),
                {
                    "threadId": thread_id,
                    "snapshotScope": snapshot_scope,
                    "snapshotId": checkpoint_id,
                },
            )
            url += "?$expand=writes"
        else:
            # Query for latest checkpoint using OData $filter
            # Must use $filter for proper query-based filtering
            params: dict[str, str] = {
                "$top": "1",
                "$orderby": "snapshotId desc",
                "$expand": "writes",
            }
            snapshot_filter = self._build_snapshot_filter(thread_id, snapshot_scope)
            if snapshot_filter:
                params["$filter"] = snapshot_filter
            url = self._build_url(self._snapshots_url(), params)

        try:
            r = await self._retry.get(
                url, token_cache=cache, ias_access_token=ias_access_token
            )
        except httpx.HTTPStatusError as e:
            # For key-based access, 404 means checkpoint doesn't exist
            if checkpoint_id and e.response.status_code == 404:
                logger.debug(
                    "GET checkpoint NOT FOUND (async): thread_id=%s checkpoint_id=%s",
                    thread_id,
                    checkpoint_id,
                )
                return None
            # For other errors or query-based access, re-raise
            raise

        # Parse snapshot response
        row = self._parse_snapshot_response(
            r, checkpoint_id, thread_id, "GET checkpoint (async)"
        )
        if row is None:
            return None

        result = self._row_to_tuple(
            row,
            ias_access_token=ias_access_token,
            tenant_id=cfg.get("tenant_id"),
        )
        logger.debug(
            "GET checkpoint SUCCESS (async): returned_id=%s has_parent=%s writes=%d",
            result.config["configurable"]["checkpoint_id"],
            result.parent_config is not None,
            len(result.pending_writes) if result.pending_writes else 0,
        )
        return result

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints, optionally filtered (async version).

        Args:
            config: Optional config with thread_id, checkpoint_ns, checkpoint_id.
                    - If None: lists across ALL threads (used by admin tooling)
                    - If checkpoint_ns key is missing: lists all namespaces for the thread
                    - If checkpoint_ns is explicitly "": lists only root namespace
                    - If checkpoint_id is provided: returns only that checkpoint (point query)
            filter: Optional metadata filter (exact equality, AND logic, server-side)
            before: Optional cursor for pagination (exclusive, keyset-based)
            limit: Optional maximum number of results to return

        Note:
            ``config=None`` (list all threads) is **not supported for IAS savers**.
            There is no mechanism to supply an ``ias_access_token`` without a config
            object, so the call will raise ``ValueError``.  XSUAA savers fall back
            to the provider tenant when ``config`` is ``None``.

        Yields:
            CheckpointTuple objects in DESC order by checkpoint_id (newest first).
            Ordering is GUARANTEED and required for pagination semantics.
        """
        if limit == 0:
            return

        # Support config=None for listing all threads (used by admin tooling)
        if config is None:
            cfg: dict[str, str] = {}
            thread_id = None
            snapshot_scope = None
            checkpoint_id = None
        else:
            cfg = self._extract_configurable(config)
            thread_id = cfg.get("thread_id")

            # Distinguish missing checkpoint_ns from explicit ""
            # Missing key → list all namespaces, explicit "" → root only
            snapshot_scope = (
                cfg.get("checkpoint_ns") if "checkpoint_ns" in cfg else None
            )

            # Support point queries with checkpoint_id
            checkpoint_id = cfg.get("checkpoint_id")

        ias_access_token, cache = self._resolve_auth(cfg)

        before_id = None
        if before is not None:
            before_cfg = self._extract_configurable(before)
            before_id = before_cfg.get("checkpoint_id")

        logger.debug(
            "LIST checkpoints (async): thread_id=%s ns=%s limit=%s before=%s filter=%s checkpoint_id=%s",
            thread_id,
            snapshot_scope,
            limit,
            before_id,
            filter,
            checkpoint_id,
        )

        # Build snapshot filter (may be None for unbounded queries)
        base_filter = self._build_snapshot_filter(
            thread_id, snapshot_scope, before_id, checkpoint_id
        )
        filter_parts = [base_filter] if base_filter else []

        # Add metadata filter if provided
        if filter:
            metadata_filter = self._build_metadata_filter(filter)
            if metadata_filter:
                filter_parts.append(metadata_filter)

        # Always order DESC; request writes inline to avoid N+1 round-trips
        params: dict[str, str] = {"$orderby": "snapshotId desc", "$expand": "writes"}
        if filter_parts:
            params["$filter"] = " and ".join(filter_parts)

        # Server-side limit always applied
        if limit is not None:
            params["$top"] = str(limit)

        # When limit is set, one page is enough — $top enforces the cap server-side.
        # When unbounded, follow @odata.nextLink to collect all pages.
        first_url = self._build_url(self._snapshots_url(), params)
        if limit is not None:
            r = await self._retry.get(
                first_url, token_cache=cache, ias_access_token=ias_access_token
            )
            r.raise_for_status()
            rows = r.json().get("value", [])
        else:
            rows = await collect_all_pages_async(
                self._retry,
                first_url,
                token_cache=cache,
                ias_access_token=ias_access_token,
            )

        for row in rows:
            yield self._row_to_tuple(
                row,
                ias_access_token=ias_access_token,
                tenant_id=cfg.get("tenant_id") if config is not None else None,
            )

        logger.debug(
            "LIST checkpoints SUCCESS (async): fetched=%d",
            len(rows),
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        thread_id = cfg["thread_id"]
        snapshot_scope = cfg.get("checkpoint_ns", "")
        serialization_type, encoded_state = self._encode_checkpoint(checkpoint)

        parent_id = cfg.get("checkpoint_id")
        logger.debug(
            "PUT checkpoint (async): thread_id=%s checkpoint_id=%s ns='%s' parent=%s",
            thread_id,
            checkpoint["id"],
            snapshot_scope,
            parent_id if parent_id else "none",
        )
        logger.debug(
            "PUT checkpoint data (async): %d bytes serialized", len(encoded_state)
        )

        body = {
            "threadId": thread_id,
            "snapshotScope": snapshot_scope,
            "snapshotId": checkpoint["id"],
            "parentSnapshotId": parent_id,
            "serializationType": serialization_type,
            "state": encoded_state,
            "metadata": metadata or {},
        }

        response = await self._retry.post(
            self._snapshots_url(),
            json=body,
            token_cache=cache,
            ias_access_token=ias_access_token,
        )
        logger.debug(
            "PUT checkpoint SUCCESS (async): thread_id=%s checkpoint_id=%s status=%d",
            thread_id,
            checkpoint["id"],
            response.status_code,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": snapshot_scope,
                "checkpoint_id": checkpoint["id"],
                **({"tenant_id": cfg["tenant_id"]} if cfg.get("tenant_id") else {}),
                **({"ias_access_token": ias_access_token} if ias_access_token else {}),
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return

        cfg = self._extract_configurable(config)
        ias_access_token, cache = self._resolve_auth(cfg)
        snapshot_id = cfg.get("checkpoint_id")
        if not snapshot_id:
            raise ValueError(
                "aput_writes requires checkpoint_id in config but none was provided"
            )

        thread_id = cfg["thread_id"]
        logger.debug(
            "PUT writes (async): thread_id=%s checkpoint_id=%s writes=%d",
            thread_id,
            snapshot_id,
            len(writes),
        )

        serialization_type, encoded_payload = self._encode_writes(writes)
        logger.debug(
            "PUT writes data (async): %d bytes serialized", len(encoded_payload)
        )

        body = {
            "threadId": thread_id,
            "snapshotScope": cfg.get("checkpoint_ns", ""),
            "snapshotId": snapshot_id,
            "writeKey": self._write_key(task_id, task_path),
            "serializationType": serialization_type,
            "payload": encoded_payload,
        }

        response = await self._retry.post(
            self._snapshot_writes_url(),
            json=body,
            token_cache=cache,
            ias_access_token=ias_access_token,
        )
        logger.debug(
            "PUT writes SUCCESS (async): thread_id=%s checkpoint_id=%s status=%d",
            thread_id,
            snapshot_id,
            response.status_code,
        )

    async def adelete_thread(
        self,
        thread_id: str,
        *,
        tenant_id: str | None = None,
        ias_access_token: str | None = None,
    ) -> None:
        """Async variant of :meth:`delete_thread`.

        Unlike ``aput``, ``alist``, and ``aget_tuple`` — which LangGraph
        invokes automatically with a ``RunnableConfig`` — this method is called
        directly by application code.  LangGraph never calls it, so there is no
        ``config`` object to carry auth context.

        Pass ``ias_access_token`` and ``tenant_id`` explicitly; the required argument
        depends on the saver kind set at construction time:

        - **IAS saver** (``base_url`` only): pass ``ias_access_token``.
        - **XSUAA saver** (full credentials): pass ``tenant_id``.

        The call is idempotent: deleting a thread that does not exist returns
        without error.

        Args:
            thread_id: ID of the thread whose checkpoints and writes to delete.
            tenant_id: BTP subaccount zone ID. Required for XSUAA savers.
            ias_access_token: IAS Bearer token. Required for IAS savers.
        """
        # POST to the deleteByThread unbound action on AgentSnapshotService.
        # OData v4 only permits DELETE by full key, so bulk-delete-by-thread is
        # exposed as an action on the backend (see agent-snapshot-service.cds).
        # See delete_thread() for the 404-as-idempotent-success pattern.
        logger.debug("DELETE thread (async): thread_id=%s", thread_id)
        if self._is_ias:
            if not (ias_access_token and ias_access_token.strip()):
                raise ValueError(
                    "No auth available: this saver was constructed with the "
                    "hana-agent-memory-serverless plan, so an IAS token is "
                    "required for this request."
                )
            resolved_ias_access_token, cache = ias_access_token, self._token_cache
        else:
            if not tenant_id:
                logger.debug(
                    "No tenant_id provided; using provider tenant. "
                    "Set tenant_id for multi-tenant routing."
                )
            resolved_ias_access_token, cache = (
                None,
                self._resolve_tenant_cache({"tenant_id": tenant_id or ""}),
            )

        try:
            r = await self._retry.post(
                self._delete_by_thread_url(),
                json={"threadId": thread_id},
                token_cache=cache,
                ias_access_token=resolved_ias_access_token,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(
                    "DELETE thread NOT FOUND (async): thread_id=%s (idempotent)",
                    thread_id,
                )
                return
            raise
        logger.debug(
            "DELETE thread SUCCESS (async): thread_id=%s status=%d",
            thread_id,
            r.status_code,
        )
