# deepflow

**Streaming, declarative multi-agent orchestration for [Deep Agents](https://github.com/langchain-ai/deepagents).**

One idea, two modes: keep the orchestrator's context **tiny** while many sub-agents do the work — and stream the whole thing live.

- 🔀 **Workflow mode** — the agent authors a phase/step plan in a single call. Independent steps **fan out** in parallel, later steps **fan in** by consuming earlier results, and each step runs in its own **isolated sub-agent**.
- 🗂️ **Task-list mode** — when a job explodes into hundreds or thousands of to-dos, a deterministic dispatcher fans them out to workers in **disjoint batches**. Each worker sees **only its slice**, so the store never has to fit in any prompt.

```text
🔀 workflow                         🗂️ task-list   (500 to-dos)
   Phase 1 · Build  (3 in ∥)           plan  500 pending · batch 50 → 10 workers (not 500 agents)
     b1   b2   b3                       ┌ w0  sees ONLY its slice (50 of 500)
   Phase 2 · Verify                     │  read_todos → 50   the other 450 are invisible to it
     v1  ⇐ b1,b2,b3                      │  ✓✓✓ … ✓
   ✓ done                               └ … w9
                                        done=500   ← all the orchestrator gets back
```

---

## Why

A normal agent runs every step in one conversation and re-reads the whole history each turn — tokens climb and the model drifts on long jobs. `deepflow` lets the agent **commit to a structure** and run the work in fresh, focused sub-agents:

- the **orchestrator** stays `O(small)` — it sees a plan or a status rollup, never the full work;
- each **worker** sees only its step or its batch;
- everything **streams** as it happens.

It's built **on top of** Deep Agents (not a fork): every sub-agent is a full Deep Agent (filesystem, shell, tools), and each mode is just an `AgentMiddleware` you can drop into any agent.

## Install

```bash
pip install deepflow-agents   # pulls in deepagents
# or
uv add deepflow-agents
```

Requires Python 3.11+. The install name is `deepflow-agents` (the bare `deepflow` is taken on PyPI); you still `import deepflow`.

---

## Quickstart

Give the agent one objective; it authors a **DAG of sub-agents** and runs it. Stream the run to watch the plan and every step as they happen:

```python
from deepflow import create_workflow_agent

agent = create_workflow_agent("openai:gpt-5.5")

for mode, chunk in agent.stream(
    {"messages": "Research PostgreSQL, SQLite and DuckDB in parallel, then recommend one for a local analytics CLI."},
    stream_mode=["updates", "custom"],
):
    if mode == "custom" and "deepflow" in chunk:
        ev = chunk["deepflow"]
        print(ev["event"], {k: v for k, v in ev.items() if k != "event"})
```

```text
plan           {'phase_count': 2, 'step_count': 4}
phase_start    {'index': 0, 'title': 'Research'}
step_start     {'id': 'postgres', 'subagent': 'general-purpose'}   ┐
step_start     {'id': 'sqlite',   'subagent': 'general-purpose'}   │ run in parallel
step_start     {'id': 'duckdb',   'subagent': 'general-purpose'}   ┘
step_done      {'id': 'duckdb'}  …
phase_start    {'index': 1, 'title': 'Recommend'}
step_done      {'id': 'recommendation'}      # ⇐ depends on all three (fan-in)
workflow_done  {}
```

The two ways to use it follow — **workflows**, and a **workflow that also drives a large task list**. Runnable versions live in [`examples/`](examples/).

---

## Use cases

### 1 · Creating workflows

The agent authors a phase/step plan in a single call. It decides *when* — it stays a normal agent and only reaches for a workflow when work fans out into parallel parts that then combine.

```python
from deepflow import create_workflow_agent

agent = create_workflow_agent(model="openai:gpt-5.5")

result = agent.invoke({"messages":
    "Compare PostgreSQL, SQLite and DuckDB for a local analytics CLI — research each in parallel, then recommend one."
})
print(result["messages"][-1].content)
```

A workflow is an ordered list of **phases**; each phase has **steps**. Phases run sequentially; steps within a phase run in parallel; a later step consumes an earlier one's output via `{{step_id}}`:

```json
{
  "phases": [
    {"title": "Research", "steps": [
      {"id": "a", "subagent_type": "general-purpose", "description": "Research A", "prompt": "Research topic A."},
      {"id": "b", "subagent_type": "general-purpose", "description": "Research B", "prompt": "Research topic B."}
    ]},
    {"title": "Synthesize", "steps": [
      {"id": "s", "subagent_type": "general-purpose", "description": "Synthesize", "depends_on": ["a", "b"],
       "prompt": "Compare and synthesize:\n\nA: {{a}}\n\nB: {{b}}"}
    ]}
  ]
}
```

Invalid plans come back to the model as actionable messages (*"Step 's' references {{a}} but does not list it in depends_on"*), not opaque tool errors.

### 2 · A workflow with a task list

A workflow can also drive a **large, long-running task list**. One flag — `enable_todos=True` — gives the agent `count_todos` / `add_todos` / `process_todos` alongside `workflow`. Hand it a **specified** list of tasks, or let it **create its own internal tasks** with `add_todos`, and it works through them at scale:

```python
from deepflow import create_workflow_agent, make_todos

agent = create_workflow_agent(model="openai:gpt-5.5", enable_todos=True, todo_batch_size=50)

# a specified task list (could be 5,000) — or let the agent build it with add_todos
result = agent.invoke({
    "messages": "Work through every pending to-do; each worker writes a one-line result.",
    "tasks": make_todos(tasks),
})
# the orchestrator only ever sees: "done=5000" — never the 5,000 items
```

**Task management** — the to-dos live in a store with status; `process_todos` partitions the pending ones into **disjoint batches**, runs them in parallel, and you re-run it to retry anything still pending/failed.

**Context management** — the orchestrator plans from **counts, never contents** (`O(log N)` — 10× the to-dos adds one digit, not 10× the tokens); each worker is handed only its batch (`O(batch)`); the store lives in state and never enters a prompt. That's what lets a single workflow drive thousands of tasks across a long-running job without its context exploding.

> **Workers are Deep Agents minus orchestration.** Every worker has the full Deep Agent toolset — filesystem, `execute`, summarization/compaction — **but no `task` and no `workflow`**: it drains its assigned slice and nothing more. Disjoint batches ⇒ no race, no double-processing.

*Prefer a dedicated task-list agent without the `workflow` tool? `create_tasklist_agent(model, batch_size=...)` is the same store and dispatch on its own.*

---

## Streaming (the point)

Stream the run and render events as they happen — same for `agent.stream(...)` and `agent.astream(...)`:

```python
for mode, chunk in agent.stream(
    {"messages": "...", "tasks": tasks},
    stream_mode=["updates", "custom"],
):
    if mode == "custom" and "deepflow" in chunk:
        ev = chunk["deepflow"]
        print(ev["event"], ev)
```

**Workflow events**

| event | when | fields |
|---|---|---|
| `plan` | before anything runs | `phase_count`, `step_count`, `phases[…]` |
| `phase_start` / `phase_done` | a phase starts / finishes | `index`, `title` |
| `step_start` | a step begins | `id`, `subagent` |
| `step_event` | live activity inside a running step | `id`, `kind` |
| `step_done` | a step settles (fires **immediately**, not batched) | `id` |
| `step_error` | a step failed (isolated) | `id`, `error` |
| `workflow_done` | the whole run finished | — |

**Task-list events**

| event | when | fields |
|---|---|---|
| `tasklist_plan` | before dispatch | `total`, `pending`, `batch_size`, `worker_count` |
| `batch_start` | a worker gets its slice | `worker`, `size`, `todos[]` |
| `worker_read` | a worker calls `read_todos` | `worker`, `returned`, `ids` |
| `batch_done` | a worker finished its slice | `worker`, `results[]` |
| `tasklist_done` | dispatch finished | `done`, `failed`, `pending`, `in_progress` |

The names live in `deepflow.events`, so the code that emits them and any reader never drift.

---

## Recipes

**Run real commands** — pass a sandbox/shell backend; workers share it with the orchestrator:

```python
import tempfile
from deepagents.backends import LocalShellBackend

agent = create_workflow_agent(
    model="openai:gpt-5.5",
    backend=LocalShellBackend(root_dir=tempfile.mkdtemp(), inherit_env=True),
)
```

**Cheaper workers** — strong orchestrator, cheap/fast workers:

```python
create_workflow_agent(model="openai:gpt-5.5", workflow_model="openai:gpt-5-mini")
create_tasklist_agent(model="openai:gpt-5.5", worker_model="openai:gpt-5-mini")
```

**Custom workflow workers** (a `general-purpose` worker is added automatically if you don't define one):

```python
create_workflow_agent(
    model="openai:gpt-5.5",
    subagents=[
        {"name": "researcher", "description": "Researches one topic.", "system_prompt": "Return 3 concise bullets."},
        {"name": "writer", "description": "Writes a synthesis.", "system_prompt": "Combine inputs into a tight brief."},
    ],
)
```

**Use the middleware directly** — each mode is an `AgentMiddleware`:

```python
from deepagents import create_deep_agent
from deepflow import WorkflowMiddleware, TaskListMiddleware

create_deep_agent(model="openai:gpt-5.5", middleware=[TaskListMiddleware(model="openai:gpt-5.5")])
```

---

## Examples

| file | shows |
|---|---|
| [`examples/workflow_demo.py`](examples/workflow_demo.py) | **1 · creating workflows** — three databases researched in parallel, then a synthesis step fans in; the plan + every step streamed live |
| [`examples/workflow_with_tasks_demo.py`](examples/workflow_with_tasks_demo.py) | **2 · a workflow with a task list** — `enable_todos=True`, a specified list dispatched in disjoint batches with a per-worker visual of each slice, its `read_todos`, and its results |
| [`examples/build_library_demo.py`](examples/build_library_demo.py) | a workflow builds a small Python library in a real shell — plain Deep Agent vs. workflow, tokens/turns compared |
| [`examples/explore_then_workflow_demo.py`](examples/explore_then_workflow_demo.py) | the agent does ordinary tool calls first, *then* authors a workflow once it knows what to fan out over |

```bash
OPENAI_API_KEY=… uv run python examples/workflow_demo.py
OPENAI_API_KEY=… uv run python examples/workflow_with_tasks_demo.py
```

## How it relates to Deep Agents

`deepflow` depends on `deepagents` and only uses its public surface (`create_deep_agent`, the `middleware=` extension point). It doesn't fork or patch internals, so it rides along with upstream Deep Agents releases.

## License

MIT — see [LICENSE](LICENSE). Built on top of [Deep Agents](https://github.com/langchain-ai/deepagents) (MIT).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
