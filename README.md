[![REUSE status](https://api.reuse.software/badge/github.com/SAP/langgraph-checkpoint-sap-agent-memory-service)](https://api.reuse.software/info/github.com/SAP/langgraph-checkpoint-sap-agent-memory-service)
[![CI](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/actions/workflows/ci.yml/badge.svg)](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/actions/workflows/codeql.yml/badge.svg)](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/actions/workflows/codeql.yml)

# LangGraph checkpointers for SAP Agent Memory Service

## About this project
This library provides an implementation of LangGraph's checkpoint saver backed by the [SAP HANA Agent Memory Service](https://help.sap.com/docs/hana-cloud/sap-hana-cloud-agent-memory-service-guide/sap-hana-cloud-agent-memory-service-guide?locale=en-US). Use it when you want LangGraph state persistence backed by SAP Agent Memory Service for durable, long-running workflows and agents.

Supports conversation continuity, crash recovery, human-in-the-loop (HITL) pause/resume, and full state history — with no database to manage.

## Requirements and Setup

- Python ≥ 3.10
- A running instance of the SAP HANA Agent Memory Service
- Understanding of [LangGraph's checkpointer](https://reference.langchain.com/python/langgraph.checkpoint) concepts

### Installation

```bash
pip install langgraph-checkpoint-sap-agent-memory-service        # Using pip
uv add langgraph-checkpoint-sap-agent-memory-service             # Using uv

```

### Quick start

#### Basic setup

Declare the auth flow explicitly with `auth=`. For XSUAA, credentials fall back to
`AGENT_MEMORY_*` env vars when omitted:

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, XsuaaAuth

saver = HanaAgentMemorySaver(auth=XsuaaAuth())
graph = builder.compile(checkpointer=saver)

# First run — checkpoints are saved automatically at each node boundary
result = await graph.ainvoke(
    {"messages": [HumanMessage("Plan my trip to Paris")]},
    config={"configurable": {"thread_id": "trip-001"}},
)

# Resume after a crash — same thread_id restores the latest checkpoint
result = await graph.ainvoke(
    None,
    config={"configurable": {"thread_id": "trip-001"}},
)
```

`XsuaaAuth()` reads `AGENT_MEMORY_TOKEN_URL`, `AGENT_MEMORY_CLIENT_ID`, and
`AGENT_MEMORY_CLIENT_SECRET`; `base_url` reads `AGENT_MEMORY_BASE_URL`.
See [Environment variables](#environment-variables) for details.

#### Custom configuration

Pass credentials directly inside the auth object if you prefer explicit configuration:

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, XsuaaAuth

saver = HanaAgentMemorySaver(
    base_url="https://<service-host>",
    auth=XsuaaAuth(
        token_url="https://<tenant>.authentication.<region>/oauth/token",
        client_id="<client-id>",
        client_secret="<client-secret>",
    ),
)
graph = builder.compile(checkpointer=saver)
```

#### IAS (serverless) plan

For the `hana-agent-memory-serverless` plan, declare `auth=IASAuth()`. Pass a Bearer token
per request via `config["configurable"]["ias_access_token"]`, or set a default token
(string or callable) on `IASAuth(ias_access_token=...)`:

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, IASAuth

# Option 1: per-request string token
saver = HanaAgentMemorySaver(base_url="https://<service-host>", auth=IASAuth())
graph = builder.compile(checkpointer=saver)

result = await graph.ainvoke(
    {"messages": [HumanMessage("Plan my trip to Paris")]},
    config={"configurable": {"thread_id": "trip-001", "ias_access_token": "<token>"}},
)
```

```python
# Option 2: callable token — fetched once and re-fetched automatically after a 401
def get_ias_token() -> str:
    return fetch_token_from_credential_store()  # your token logic here

saver = HanaAgentMemorySaver(
    base_url="https://<service-host>",
    auth=IASAuth(ias_access_token=get_ias_token),
)
graph = builder.compile(checkpointer=saver)

# No ias_access_token in config — the callable is invoked automatically
result = await graph.ainvoke(
    {"messages": [HumanMessage("Plan my trip to Paris")]},
    config={"configurable": {"thread_id": "trip-001"}},
)
```

Async functions and callable objects are supported by async saver methods; see [IAS token refresh](#ias-token-refresh) for execution and identity scope.

## Common usage patterns

### Crash recovery

Every node transition is saved. If the process crashes mid-run, restart with the same `thread_id` and LangGraph resumes from the last committed checkpoint — no special code required.

```python
# Run 1 — crashes after node_2
graph.invoke(initial_state, config)  # raises

# Run 2 — resumes from node_3, skips node_1 and node_2
graph.invoke(None, config)
```

### Human-in-the-loop (HITL)

Use LangGraph's `interrupt()` to pause at a node and resume with human input. The checkpoint preserves the interrupted state across process boundaries.

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    decision = interrupt({"question": "Approve this action?"})
    return {"approved": decision["approved"]}

# Pause — graph.invoke raises NodeInterrupt
graph.invoke(initial_state, config)

# Resume with human input
graph.invoke(Command(resume={"approved": True}), config)
```

### IAS token refresh

- **String token**: used directly. A 401 is returned without a refresh attempt.
- **Callable**: fetched on first use and refreshed after a 401. Concurrent calls
  share one refresh; a delayed 401 cannot invalidate a newer token. Each request
  gets at most one authentication retry. Failed acquisition can be retried.
- **Per-request override** via `config["configurable"]["ias_access_token"]`
  uses its own cache. Missing or space-only overrides select the default.
  Tokens must be nonempty strings without internal whitespace or control
  characters; surrounding spaces are removed.

A default IAS source must represent **one stable authenticated principal**.
For different users, pass authenticated tokens per request or use separate savers.
`tenant_id` routes XSUAA requests only; it does not partition the IAS default cache.

Async saver methods support async functions, objects with async `__call__`,
and callbacks returning awaitables. Synchronous callbacks run in a worker with
context variables preserved. Callbacks needing an event loop during invocation
should use `async def`. Use async saver methods for async-only providers.

`timeout` applies to HTTP requests. IAS callbacks must set their own I/O
timeouts. Cancelling an async caller does not cancel a shared refresh; keep its
event loop alive until that refresh finishes.

```python
# Sync callable
auth=IASAuth(ias_access_token=lambda: my_token_service.get_token())

# Async callable
async def get_token() -> str:
    return await my_async_token_service.get_token()

auth=IASAuth(ias_access_token=get_token)
```

## State history

```python
# All checkpoints for a thread, newest first
for tup in graph.get_state_history(config):
    print(tup.config["configurable"]["checkpoint_id"])
    print(tup.metadata)
```

## Subgraph support

Nested subgraphs work without any extra configuration. LangGraph sets `checkpoint_ns` automatically using a pipe-separated format (`"outer_node:uuid|inner_node:uuid"`), which maps directly to the `snapshotScope` field in the Agent Memory Service.

## Configuration reference

### `HanaAgentMemorySaver`

The auth flow is declared explicitly via the required `auth=` argument, which takes
either an `XsuaaAuth(...)` or `IASAuth(...)` object. Credentials live inside that object and
each can be supplied as an argument, an environment variable, or both (argument wins).

| Parameter  | Type                         | Environment variable    | Default              | Description                                          |
| ---------- | ---------------------------- | ----------------------- | -------------------- | ---------------------------------------------------- |
| `base_url` | `str \| None`                | `AGENT_MEMORY_BASE_URL` | —                    | Agent Memory Service base URL, e.g. `https://<host>` |
| `auth`     | `XsuaaAuth \| IASAuth`       | —                       | — (required)         | Auth flow declaration (see below)                    |
| `timeout`  | `float`                      | —                       | `30.0`               | HTTP request timeout in seconds                      |
| `serde`    | `SerializerProtocol \| None` | —                       | `JsonPlusSerializer` | Custom serializer for checkpoint state               |

**`XsuaaAuth(token_url=..., client_id=..., client_secret=...)`** — OAuth2 `client_credentials` flow:

| Parameter       | Type          | Environment variable         | Description               |
| --------------- | ------------- | ---------------------------- | ------------------------- |
| `token_url`     | `str \| None` | `AGENT_MEMORY_TOKEN_URL`     | OAuth2 token endpoint     |
| `client_id`     | `str \| None` | `AGENT_MEMORY_CLIENT_ID`     | OAuth2 client ID          |
| `client_secret` | `str \| None` | `AGENT_MEMORY_CLIENT_SECRET` | OAuth2 client secret      |

**`IASAuth(ias_access_token=...)`** — per-request Bearer token (`hana-agent-memory-serverless`):

| Parameter          | Type          | Environment variable              | Description                                                         |
| ------------------ | ------------- | --------------------------------- | ------------------------------------------------------------------- |
| `ias_access_token` | `str \| Callable \| None` | `AGENT_MEMORY_IAS_ACCESS_TOKEN`   | Token string, sync callable, or async callable. Callable is invoked lazily and re-invoked after a 401. `config["configurable"]["ias_access_token"]` overrides per request. Env var only accepts a string. |

### Environment variables

```bash
AGENT_MEMORY_BASE_URL          # Agent Memory Service base URL
AGENT_MEMORY_TOKEN_URL         # XSUAA: OAuth2 token endpoint
AGENT_MEMORY_CLIENT_ID         # XSUAA: OAuth2 client ID
AGENT_MEMORY_CLIENT_SECRET     # XSUAA: OAuth2 client secret
AGENT_MEMORY_IAS_ACCESS_TOKEN  # IAS: default Bearer token (optional)
```

Environment variables are consulted only for arguments left as None. Explicit empty credentials are rejected; they do not select environment credentials.

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2026 SAP SE or an SAP affiliate company and langgraph-checkpoint-sap-agent-memory-service contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/langgraph-checkpoint-sap-agent-memory-service).
