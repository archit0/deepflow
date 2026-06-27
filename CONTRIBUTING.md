# Contributing to deepflow

Thanks for your interest! deepflow is a small, focused package — a streaming
workflow layer on top of [Deep Agents](https://github.com/langchain-ai/deepagents).
Contributions that keep it small, well-tested, and decoupled from Deep Agents
internals are very welcome.

## Development setup

deepflow uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/archit0/deepflow.git
cd deepflow
uv sync --group dev
```

## The workflow (pun intended)

```bash
make test     # run unit tests (no network/model needed)
make lint     # ruff check
make format   # ruff format + autofix
```

Or directly:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Please run `make format` and `make lint` and make sure `make test` passes before
opening a PR.

## Guidelines

- **Stay decoupled.** The engine and middleware only depend on LangChain /
  LangGraph public APIs and on Deep Agents' public surface (`create_deep_agent`,
  `middleware=`). Don't import Deep Agents private (`_`-prefixed) internals — if
  you need a helper, inline a small version (see `src/deepflow/_subagent.py`).
- **Keep the event schema honest.** Anything the engine emits lives in
  `src/deepflow/events.py`, and the way results are read should match the way
  the run streams. If you add an event, add it there and cover it with a test.
- **Tests don't need models.** Sub-agents in tests are tiny fakes that expose
  `stream` / `astream`; keep new tests network-free.
- **Conventional-ish commits** are appreciated (`feat:`, `fix:`, `docs:`,
  `test:`) but not enforced.

## Reporting issues

Open an issue with a minimal repro. For questions about Deep Agents itself,
the [Deep Agents repo](https://github.com/langchain-ai/deepagents) is the right
place.
