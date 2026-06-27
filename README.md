# deepflow

**Streaming, declarative multi-agent workflows for [Deep Agents](https://github.com/langchain-ai/deepagents).**

Give an agent a single `workflow` tool and it can plan a whole multi-step job in one call instead of improvising step-by-step in one long, growing conversation. Independent steps **fan out** in parallel, later steps **fan in** by consuming earlier results, and each step runs in its own **isolated sub-agent** — so the orchestrator's context stays small and the run **streams live** from start to finish.

```text
📋 plan   2 phases / 4 steps
   Phase 1 · Build      (3 in parallel)
      b1  build add        b2  build is_even      b3  build reverse_string
   Phase 2 · Verify
      v1  ⇐ b1,b2,b3   run all tests
⚙ Build    ✓ b1   ✓ b2   ✓ b3
⚙ Verify   ✓ v1
```

---

## Why

A normal agent runs every step in one conversation and re-reads the whole history on each turn — tokens climb and the model has more room to drift on long jobs. `deepflow` lets the agent commit to a plan up front and run each step in a fresh, focused sub-agent. Same result, a fraction of the orchestrator's context, and a clean live stream of what's happening.

It's built **on top of** Deep Agents (not a fork): each sub-agent is a full Deep Agent with the filesystem, shell, and tools, and the `workflow` tool is just an `AgentMiddleware` you can drop into any agent.

## Install

```bash
pip install deepflow        # pulls in deepagents
# or
uv add deepflow
```

Requires Python 3.11+.

## Quickstart

```python
from deepflow import create_workflow_agent

agent = create_workflow_agent(model="openai:gpt-5.5")

result = agent.invoke({"messages": "Research Postgres and SQLite, then recommend one for a CLI app."})
print(result["messages"][-1].content)
```

The agent decides *when* to use a workflow — it stays a normal agent and only authors one for multi-stage or fan-out/fan-in work. For everything else it just works directly.

### Streaming (the point)

Stream the run and render events as they happen:

```python
for mode, chunk in agent.stream(
    {"messages": "Build a small utility library and run its tests."},
    stream_mode=["updates", "custom"],
):
    if mode == "custom" and "deepflow" in chunk:
        ev = chunk["deepflow"]
        print(ev["event"], ev)        # plan / phase_start / step_start / step_event / step_done / ...
```

Events (see `deepflow.events`):

| event | when | fields |
|---|---|---|
| `plan` | before anything runs | `phase_count`, `step_count`, `phases[…]` |
| `phase_start` / `phase_done` | a phase starts / finishes | `index`, `title` |
| `step_start` | a step begins | `id`, `subagent` |
| `step_event` | live activity inside a running step | `id`, `kind` (`message`/`tool_call`/`tool_result`) |
| `step_done` | a step settles (fires **immediately**, not batched) | `id` |
| `step_error` | a step failed (isolated) | `id`, `error` |
| `workflow_done` | the whole run finished | — |

Async works the same with `agent.astream(...)`, and per-step events fire in real time in both.

## The workflow the model authors

A workflow is an ordered list of **phases**; each phase has **steps**:

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

- **Phases run sequentially**; **steps within a phase run in parallel** (fan-out).
- A later step consumes an earlier one's output via `{{step_id}}` (fan-in); referenced ids must be in `depends_on`.
- Each `step.description` shows up in the `plan` preview before the run.

Invalid plans come back to the model as actionable messages (e.g. *"Step 's' references {{a}} but does not list it in depends_on"*), not opaque tool errors.

## Running real commands

Pass a sandbox/shell backend so workflow steps can write files and run commands — the workers share it with the orchestrator:

```python
import tempfile
from deepagents.backends import LocalShellBackend
from deepflow import create_workflow_agent

agent = create_workflow_agent(
    model="openai:gpt-5.5",
    backend=LocalShellBackend(root_dir=tempfile.mkdtemp(), inherit_env=True),
)
```

## Cheaper workers

Run the orchestrator on a strong model and the workflow's workers on a cheaper/faster one:

```python
agent = create_workflow_agent(
    model="openai:gpt-5.5",            # orchestrator (authors the plan)
    workflow_model="openai:gpt-5-mini",  # the step workers
)
```

## Custom workers

```python
agent = create_workflow_agent(
    model="openai:gpt-5.5",
    subagents=[
        {"name": "researcher", "description": "Researches one topic.", "system_prompt": "Return 3 concise bullets."},
        {"name": "writer", "description": "Writes a synthesis.", "system_prompt": "Combine inputs into a tight brief."},
    ],
)
```

A `general-purpose` worker is added automatically if you don't define one.

## Use the middleware directly

`create_workflow_agent` is a convenience wrapper. The core is an `AgentMiddleware` you can add to any Deep Agent:

```python
from deepagents import create_deep_agent
from deepflow import WorkflowMiddleware

workers = [{"name": "general-purpose", "description": "...", "runnable": create_deep_agent(model="openai:gpt-5.5")}]
agent = create_deep_agent(model="openai:gpt-5.5", middleware=[WorkflowMiddleware(subagents=workers)])
```

## Examples

- [`examples/build_library_demo.py`](examples/build_library_demo.py) — one agent builds a small Python library (writes files, runs the tests in a real shell) two ways: a plain Deep Agent vs. a workflow. Compares tokens, turns, and streams every event live.

## How it relates to Deep Agents

`deepflow` depends on `deepagents` and only uses its public surface (`create_deep_agent`, the `middleware=` extension point). It doesn't fork or patch internals, so it rides along with upstream Deep Agents releases. If workflow mode ever lands in Deep Agents itself, migrating off `deepflow` is a one-line change.

## License

MIT — see [LICENSE](LICENSE). Built on top of [Deep Agents](https://github.com/langchain-ai/deepagents) (MIT).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
