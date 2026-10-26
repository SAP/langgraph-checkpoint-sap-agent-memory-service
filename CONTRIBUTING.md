# Contributing

This guide covers how to build, test, and contribute to `langgraph-checkpoint-sap-agent-memory`.

## Code of Conduct

All members of the project community must abide by the [SAP Open Source Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md).
Only by respecting each other we can develop a productive, collaborative community.
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting [a project maintainer](REUSE.toml).

## Engaging in Our Project

We use GitHub to manage reviews of pull requests.

* If you are a new contributor, see: [Steps to Contribute](#steps-to-contribute)

* Before implementing your change, create an issue that describes the problem you would like to solve or the code that should be enhanced. Please note that you are willing to work on that issue.

* The team will review the issue and decide whether it should be implemented as a pull request. In that case, they will assign the issue to you. If the team decides against picking up the issue, the team will post a comment with an explanation.

## Steps to Contribute

Should you wish to work on an issue, please claim it first by commenting on the GitHub issue that you want to work on. This is to prevent duplicated efforts from other contributors on the same issue.

If you have questions about one of the issues, please comment on them, and one of the maintainers will clarify.

## Contributing Code or Documentation

You are welcome to contribute code in order to fix a bug or to implement a new feature that is logged as an issue.

The following rule governs code contributions:

* Contributions must be licensed under the [Apache 2.0 License](./LICENSE).
* Due to legal reasons, contributors will be asked to accept a Developer Certificate of Origin (DCO) when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).
* Contributions must follow our [guidelines on AI-generated code](https://github.com/SAP/.github/blob/main/CONTRIBUTING_USING_GENAI.md) in case you are using such tools.

## Issues and Planning

* We use GitHub issues to track bugs and enhancement requests.

* Please provide as much context as possible when you open an issue. The information you provide must be comprehensive enough to reproduce that issue for the assignee.

---

## Building and Testing

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [`uv`](https://docs.astral.sh/uv/) — used for environment and dependency management
- [SAP Agent Memory Service](https://help.sap.com/docs/hana-cloud/sap-hana-cloud-agent-memory-service-guide/sap-hana-cloud-agent-memory-service-guide?locale=en-US) — for conformance tests

---

### Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd langgraph-checkpoint-sap-agent-memory

# Create a virtual environment and install all dependencies (including dev)
uv venv --python 3.12
uv pip install -e ".[dev]"
```

---

### Running checks

All checks mirror what CI runs. Run them in this order before raising a PR:

```bash
# Lint
uv run ruff check langgraph/ tests/
uv run ruff format --check langgraph/ tests/

# Type check
uv run ty check langgraph/

# Unit + serde tests (no network required)
uv run pytest tests/ -k "not conformance" -v

# Or run everything at once via make
make check
```

#### Auto-fix formatting

```bash
uv run ruff format langgraph/ tests/
```

---

### Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
The format is enforced by CI on every PR title and every push to `main`.

```
<type>(<scope>): <short description>

[optional body]

[optional footer — e.g. BREAKING CHANGE: ...]
```

**Allowed types:**

| Type | When to use |
|---|---|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `test` | Adding or updating tests |
| `refactor` | Code change that is neither a fix nor a feature |
| `chore` | Dependency updates, build config, tooling |
| `ci` | Changes to GitHub Actions workflows |
| `perf` | Performance improvements |

**Examples:**

```bash
feat(checkpointer): add async list() support
fix(http): retry on 503 responses
docs: add conformance test setup to CONTRIBUTING
ci: add concurrency controls to all workflows
chore(deps): bump httpx to 0.28
feat!: drop Python 3.9 support  # the ! signals a breaking change
```

**Scope** is optional but encouraged when the change is localised (e.g. `checkpointer`, `http`, `serde`, `deps`).

**PR title** is what matters most — when you squash-merge, the PR title becomes the commit message on `main`.

---

### Test layout

| File                         | What it covers                                                           |
| ---------------------------- | ------------------------------------------------------------------------ |
| `tests/test_checkpointer.py` | All `HanaAgentMemorySaver` methods — mocked HTTP via `respx`, no network |
| `tests/test_serde.py`        | Base64 round-trip for every serialization type tag                       |
| `tests/test_conformance.py`  | LangGraph conformance suite — requires a live Agent Memory Service       |

#### Running conformance tests

The conformance suite validates the full `BaseCheckpointSaver` contract. See the official docs:

- [Testing with the conformance suite](https://docs.langchain.com/oss/python/langgraph/checkpointers#testing-with-the-conformance-suite) — LangGraph custom checkpointer guide

To run them locally you need SAP Agent Memory Service. Set the following environment variables:

```bash
export AGENT_MEMORY_BASE_URL=https://<service-host>
export AGENT_MEMORY_TOKEN_URL=https://<tenant>.authentication.<region>/oauth/token
export AGENT_MEMORY_CLIENT_ID=<client-id>
export AGENT_MEMORY_CLIENT_SECRET=<client-secret>

uv run pytest tests/test_conformance.py -v -s
```

---

### Project layout

```
langgraph/
└── checkpoint/
    └── sap/
        └── agent_memory/
            ├── __init__.py
            ├── checkpointer.py
            └── _http.py
tests/
    ├── test_checkpointer.py
    ├── test_serde.py
    └── test_conformance.py
```

---

### Ruff rules

Configured in `pyproject.toml`. Run `ruff check --help` for rule documentation.

---

### Adding a new dependency

Stop and ask first. The dependency list is intentionally minimal:

| Dependency                  | Reason                                                             |
| --------------------------- | ------------------------------------------------------------------ |
| `langgraph-checkpoint>=2.0` | `BaseCheckpointSaver`, `JsonPlusSerializer`                        |
| `httpx>=0.27`               | Async HTTP client                                                  |
| `uuid6>=2024.1`             | UUID v7 for `get_next_version()` — not in stdlib until Python 3.14 |
