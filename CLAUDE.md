# AGENTS.md — deepflow

Guidance for AI agents and human contributors working in this repo. This file
and `CLAUDE.md` are kept identical.

## What this is

**deepflow** is a small, focused package that adds **streaming, declarative
multi-agent orchestration** to [Deep Agents](https://github.com/langchain-ai/deepagents).
Two modes, one idea — keep the orchestrator's context tiny while sub-agents do
the work, and stream it live:

- **Workflow mode** (`workflow` tool): the agent authors a phase/step plan in a
  single call; steps fan out in parallel, later steps consume earlier results via
  `{{step_id}}` templating, each step runs in its own isolated sub-agent.
- **Task-list mode** (`process_todos` tool): a job that explodes into hundreds or
  thousands of to-dos is dispatched to workers in **disjoint batches** — each
  worker sees only its slice, the store never enters a prompt, the orchestrator
  sees only a status rollup.

- **PyPI name:** `deepflow-agents` (bare `deepflow` is owned by an unrelated
  project). **Import name:** `deepflow`. So `pip install deepflow-agents` →
  `from deepflow import create_workflow_agent`.
- **Built on top of Deep Agents, not a fork.** It uses only the public
  `deepagents` surface (`create_deep_agent`, the `middleware=` extension point)
  and inlines the few small helpers it needs — never import `deepagents` private
  (`_`-prefixed) internals.

## Layout

```
src/deepflow/
  __init__.py     # public API + __version__
  spec.py         # Pydantic models, validation, {{}} templating, plan payload
  events.py       # the event-name schema emitted on the custom stream
  _subagent.py    # decoupled helpers: state prep, result extraction, delta merge
  engine.py       # streaming executor (sync + async) — workflow mode
  middleware.py   # WorkflowMiddleware + the `workflow` tool + prompts
  tasklist.py     # task-list mode: store, paginated tools, dispatcher, workers, TaskListMiddleware
  agent.py        # create_workflow_agent / create_tasklist_agent (batteries-included wrappers)
  py.typed
tests/test_workflow.py   # network-free unit tests for workflow mode
tests/test_tasklist.py   # network-free unit tests for task-list mode
examples/                # runnable demos (need OPENAI_* env)
```

## Public API (`from deepflow import ...`)

- `create_workflow_agent(model, *, subagents=None, tools=None, backend=None, system_prompt=None, workflow_model=None, max_concurrency=None, max_steps=25, enable_todos=False, todo_batch_size=50, todo_max_workers=8, **kwargs)` — workflow mode (with optional `enable_todos` to also mount task-list mode).
- `create_tasklist_agent(model, *, tools=None, backend=None, system_prompt=None, worker_model=None, worker_system_prompt=…, batch_size=50, max_workers=8, **kwargs)` — task-list mode. Seed `todos` at invoke via `make_todos([...])`, or let the agent build them with `add_todos`.
- `WorkflowMiddleware` — injects the `workflow` tool; `TaskListMiddleware` — injects `count_todos`/`add_todos`/`process_todos`. Both drop into any `create_deep_agent(middleware=[...])`.
- `make_todos`, `dispatch`, `verify`, `aggregate`, `make_worker`, `worker_tool_names`, `Todo`, `TodoSpec`, `TaskListState`, `TodoStoreMiddleware`.
- `WorkflowSpec`, `WorkflowPhase`, `WorkflowStep`, `WorkflowToolArgs`, `validate_workflow`, `render_prompt`, `plan_payload`, `CompiledSubAgent`, `WorkflowSubAgent`, `events`.

## Architecture & key behaviours

- **The workflow the model authors:** `phases: [{title, steps: [{id, subagent_type, description, prompt, depends_on}]}]`. Phases run **sequentially**; steps within a phase run **concurrently** (fan-out); a later step consumes an earlier one via `{{step_id}}` (fan-in). Validation (`validate_workflow`) rejects unknown sub-agents, forward/same-phase deps, undeclared `{{}}` refs, duplicate ids, and over-`max_steps` — returning an **actionable message to the model**, not an opaque tool error (the tool advertises a loose `list[Any]` arg schema so the engine, not the schema boundary, reports the error).
- **Sub-agent tools:** a workflow worker gets **everything the orchestrator has except the `workflow` tool** — the full Deep Agents suite (`write_todos`, filesystem tools, `execute`), the same `backend`, any `tools=` you pass, and its own `task` tool (it's a full Deep Agent). The `WorkflowMiddleware` is attached only to the orchestrator, which prevents nested workflows.
- **Streaming-first (`deepflow.events`):** the engine emits `plan` → `phase_start` → `step_start` → `step_event` → `step_done` → `phase_done` → `workflow_done` on the LangGraph **custom** stream as `{"deepflow": {...}}`. Consume with `agent.stream(..., stream_mode=["updates", "custom"])`. `step_done` fires the moment a step settles (not batched at phase end). The **async** path forwards live per-message `step_event`s from inside each running sub-agent; the **sync** path emits `step_start`/`step_done` from the tool's own thread and skips the per-message firehose.

### Task-list mode (`tasklist.py`)

- **The store:** a dedicated **`tasks`** state channel (`{id: {content, status, result, check?, group?}}`) — NOT `todos` — with a **merge reducer** (`_merge`, union by id) so concurrent worker writes never clobber. Using its own channel means it never aliases the planning `todos` channel that `deepagents` mounts (that collision crashed `count_todos`/`process_todos` and wiped the store — see history). Seed it at invoke with `{"tasks": make_todos([...])}`. `make_todos` accepts plain strings or `{content, check?, group?}` dicts. It is **never injected into a prompt** — agents touch it only via paginated/filtered tools. `_merge` keeps the existing dict on any non-dict write (belt-and-braces). `tasks` is in `_subagent.EXCLUDED_STATE_KEYS`, so it never leaks into workflow steps.
- **Tools:** workers get `read_todos` (paginated page, never the whole store) / `write_todos` / `add_todos` / `count_todos`; the orchestrator (`TaskListMiddleware`) gets `count_todos` / `add_todos` / `process_todos` / `verify_todos` only — **not** `write_todos`. Tool names keep the `todos` wording; the state channel is `tasks`.
- **Verified "done" (3 layers, all opt-in):** (1) **`check`** — a per-to-do shell command the *engine* runs in `dispatch` after a worker marks `done` (`_run_check`/`_verify_checks`, injectable `check_runner` for tests); a non-zero exit flips `done`→`failed` and emits `check_failed`. Deterministic, model-proof. (2) **Evidence** — `WORKER_PROMPT` requires a concrete `result`, not a task restatement. (3) **`verify`/`verify_todos`** — an independent agent (`agent_verifier_fn`) re-checks a deterministic **stride sample** (`done[::step]`, no RNG) of completed to-dos and flips wrong ones to `failed`; emits `verify_plan`/`verify_done`. Convergence loop: `count → process → verify → process …` until none pending/failed.
- **Group co-location:** `_batches` clusters same-`group` to-dos (first-seen order) into the same batch so related work lands on one worker; ungrouped to-dos keep insertion order; a group larger than `batch_size` still splits (size is a hard cap).
- **Workers drain their WHOLE batch** (`read_todos()` with no status filter), not just `pending` — otherwise a retry (which re-dispatches `failed` items) would be a no-op because the worker would see no `pending` items and stop.
- **The dispatcher (`dispatch`)** partitions pending to-dos into disjoint batches and runs them in a `ThreadPoolExecutor`, capturing `copy_context()` in the tool thread (same lesson as the workflow engine) so tracing/streaming propagate. Returns `(delta, rollup, batch_count)`. It emits `tasklist_plan` → `batch_start` → `worker_read` → `batch_done` → `tasklist_done`.
- **Workers** are built with langchain `create_agent` + `FilesystemMiddleware` + `SummarizationMiddleware` + `TodoStoreMiddleware` — **deliberately not `create_deep_agent`**, so they get filesystem/`execute`/compaction but **no `task` and no `workflow`** (verify with `worker_tool_names`).
- **`enable_todos` on `create_workflow_agent`** appends `TaskListMiddleware` to a `create_deep_agent`; because the store is on its own `tasks` channel, it coexists cleanly with deepagents' planning `todos` (both work independently).

## Gotchas (these have bitten us — keep them in mind)

1. **Thread context for callbacks/streaming.** A LangGraph stream writer and the
   parent run's callbacks/tracing (Langfuse, LangSmith) only propagate from the
   **tool's own thread**. The sync engine runs steps in a `ThreadPoolExecutor`,
   so it must capture `contextvars.copy_context()` **in the tool thread** and
   hand it to each worker via `ctx.run` — capturing it inside the worker grabs
   the worker's empty context and silently drops callbacks (a workflow then
   traces as a black-box tool span with nothing inside). See `engine.run_workflow`.
2. **No nested same-quote f-strings** in `examples/` — this repo targets Python
   3.11, where reusing the outer quote inside an f-string is a syntax error.
   Precompute the substring instead.
3. **Result-reader matches the emitter.** Event names live only in
   `deepflow.events`; results are extracted from the same final state the stream
   surfaces. If you add an event, add it there and cover it with a test.

## Dev workflow

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev      # install
make test                # uv run pytest -q  (no network/models needed)
make lint                # uv run ruff check .
make format              # uv run ruff format . && ruff check --fix .
```

Always run `make format` + `make lint` and ensure `make test` passes before
committing. Tests use tiny fake sub-agents exposing `invoke`/`stream`/`astream`
that yield `{"messages": [...]}` — keep new tests network-free.

## Conventions

- **Ruff** line length 150; config + per-file ignores in `pyproject.toml`.
- **Commits:** conventional-ish (`feat:`, `fix:`, `docs:`, `test:`, `build:`).
- **No Claude / AI attribution** in commits, PRs, or docs — author as the repo owner only.
- Keep the package **decoupled from `deepagents` internals** (inline small helpers).

## Release process (always tag + release)

Automated via `.github/workflows/release.yml` (PyPI **Trusted Publishing** — no
tokens/secrets). Pushing a `vX.Y.Z` tag builds, `twine check`s, publishes to
PyPI, and creates the GitHub release.

1. Bump `__version__` in `src/deepflow/__init__.py` to match the tag.
2. `make format && make lint && make test`.
3. `git commit -am "release: X.Y.Z" && git tag vX.Y.Z && git push origin main vX.Y.Z`.
4. The workflow does the rest. **Every published version gets a git tag + GitHub release** — this is required, never PyPI-only.

One-time PyPI config: project `deepflow-agents` → Manage → Publishing → add a
GitHub trusted publisher (owner `archit0`, repo `deepflow`, workflow
`release.yml`, environment `pypi`).

Manual fallback (if needed): `uv build && uvx twine upload dist/*` (token in
`~/.pypirc`), then `git tag` + `gh release create`.

## Links

- Repo: https://github.com/archit0/deepflow
- PyPI: https://pypi.org/project/deepflow-agents/
- Built on: https://github.com/langchain-ai/deepagents
