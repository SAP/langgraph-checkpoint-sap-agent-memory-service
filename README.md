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
pip install langgraph-checkpoint-sap-agent-memory        # Using pip
uv add langgraph-checkpoint-sap-agent-memory             # Using uv

```

### Quick start

#### Basic setup

Declare the auth flow explicitly with `auth=`. For XSUAA, credentials fall back to
`AGENT_MEMORY_*` env vars when omitted:

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, XSUAA

saver = HanaAgentMemorySaver(auth=XSUAA())
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

`XSUAA()` reads `AGENT_MEMORY_TOKEN_URL`, `AGENT_MEMORY_CLIENT_ID`, and
`AGENT_MEMORY_CLIENT_SECRET`; `base_url` reads `AGENT_MEMORY_BASE_URL`.
See [Environment variables](#environment-variables) for details.

#### Custom configuration

Pass credentials directly inside the auth object if you prefer explicit configuration:

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, XSUAA

saver = HanaAgentMemorySaver(
    base_url="https://<service-host>",
    auth=XSUAA(
        token_url="https://<tenant>.authentication.<region>/oauth/token",
        client_id="<client-id>",
        client_secret="<client-secret>",
    ),
)
graph = builder.compile(checkpointer=saver)
```

#### IAS (serverless) plan

For the `hana-agent-memory-serverless` plan, declare `auth=IAS()` and pass a Bearer
token per request via `config["configurable"]["ias_access_token"]` (an optional
default token may be set on `IAS(ias_access_token=...)`, falling back to
`AGENT_MEMORY_IAS_ACCESS_TOKEN`):

```python
from langgraph.checkpoint.sap.agent_memory import HanaAgentMemorySaver, IAS

saver = HanaAgentMemorySaver(base_url="https://<service-host>", auth=IAS())
graph = builder.compile(checkpointer=saver)

result = await graph.ainvoke(
    {"messages": [HumanMessage("Plan my trip to Paris")]},
    config={"configurable": {"thread_id": "trip-001", "ias_access_token": "<token>"}},
)
```

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
either an `XSUAA(...)` or `IAS(...)` object. Credentials live inside that object and
each can be supplied as an argument, an environment variable, or both (argument wins).

| Parameter  | Type                         | Environment variable    | Default              | Description                                          |
| ---------- | ---------------------------- | ----------------------- | -------------------- | ---------------------------------------------------- |
| `base_url` | `str \| None`                | `AGENT_MEMORY_BASE_URL` | —                    | Agent Memory Service base URL, e.g. `https://<host>` |
| `auth`     | `XSUAA \| IAS`               | —                       | — (required)         | Auth flow declaration (see below)                    |
| `timeout`  | `float`                      | —                       | `30.0`               | HTTP request timeout in seconds                      |
| `serde`    | `SerializerProtocol \| None` | —                       | `JsonPlusSerializer` | Custom serializer for checkpoint state               |

**`XSUAA(token_url=..., client_id=..., client_secret=...)`** — OAuth2 `client_credentials` flow:

| Parameter       | Type          | Environment variable         | Description               |
| --------------- | ------------- | ---------------------------- | ------------------------- |
| `token_url`     | `str \| None` | `AGENT_MEMORY_TOKEN_URL`     | OAuth2 token endpoint     |
| `client_id`     | `str \| None` | `AGENT_MEMORY_CLIENT_ID`     | OAuth2 client ID          |
| `client_secret` | `str \| None` | `AGENT_MEMORY_CLIENT_SECRET` | OAuth2 client secret      |

**`IAS(ias_access_token=...)`** — per-request Bearer token (`hana-agent-memory-serverless`):

| Parameter          | Type          | Environment variable              | Description                                                         |
| ------------------ | ------------- | --------------------------------- | ------------------------------------------------------------------- |
| `ias_access_token` | `str \| None` | `AGENT_MEMORY_IAS_ACCESS_TOKEN`   | Optional default token; `config["configurable"]["ias_access_token"]` overrides it per request |

### Environment variables

```bash
AGENT_MEMORY_BASE_URL          # Agent Memory Service base URL
AGENT_MEMORY_TOKEN_URL         # XSUAA: OAuth2 token endpoint
AGENT_MEMORY_CLIENT_ID         # XSUAA: OAuth2 client ID
AGENT_MEMORY_CLIENT_SECRET     # XSUAA: OAuth2 client secret
AGENT_MEMORY_IAS_ACCESS_TOKEN  # IAS: default Bearer token (optional)
```

Constructor/auth-object arguments take priority over environment variables when both are supplied.

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/langgraph-checkpoint-sap-agent-memory-service/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2026 SAP SE or an SAP affiliate company and langgraph-checkpoint-sap-agent-memory-service contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/langgraph-checkpoint-sap-agent-memory-service).
