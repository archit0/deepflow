#!/usr/bin/env python
"""Example 2 — A workflow with a specified task list.

`create_workflow_agent(enable_todos=True)` gives one agent BOTH the `workflow`
tool AND a scalable to-do list (`count_todos` / `add_todos` / `process_todos`).
Hand it a *specified* list of tasks (below) — or let it create its own internal
tasks with `add_todos` — and it manages them at scale:

  * task management — a store-backed to-do list with status, batches and retries;
  * context management — the orchestrator plans from COUNTS (never the contents)
    and dispatches the work to workers in disjoint batches, so each worker only
    ever sees its own slice.

That's how a workflow stays small while still driving a long-running job with a
large number of tasks. Every worker is a full deep agent (filesystem, shell,
compaction) minus the ability to spawn more agents.

Reads OPENAI_API_KEY / OPENAI_BASE_URL from the env. Model via DEMO_MODEL
(default gpt-5.5).

    uv run python examples/workflow_with_tasks_demo.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasklist_view import col, make_renderer, print_board

from deepflow import create_workflow_agent, make_todos

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

# The "specified task" — a list the agent must work through (imagine 5,000 of these).
FEATURES = [
    "passwordless login", "dark mode", "CSV export", "webhook retries", "audit log",
    "SSO (SAML)", "rate limiting", "in-app search", "bulk delete", "API keys v2",
    "usage dashboard", "email digests",
]
TASKS = [f"Write a one-line release note for: {f}." for f in FEATURES]

ASK = (
    "You have a specified list of pending to-dos — a long-running job. Do NOT read them all. "
    "Check `count_todos` first, then call `process_todos` once with batch_size=4 so the work fans out "
    "across workers; each worker reads its own to-dos and writes a one-line result. Report the final tally."
)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    print(col("deepflow — example 2 · a workflow with a specified task list", "bold"))
    print(f"model: {MODEL}  ·  {len(TASKS)} specified tasks · batch_size=4 → 3 workers\n")

    agent = create_workflow_agent(model=MODEL, enable_todos=True, todo_batch_size=4, todo_max_workers=3)

    start = time.perf_counter()
    render = make_renderer(start)
    final = None
    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": ASK}], "tasks": make_todos(TASKS)},
        stream_mode=["updates", "custom", "values"],
    ):
        if mode == "values":
            final = chunk
            continue
        render(mode, chunk)

    if final and isinstance(final.get("tasks"), dict):
        print_board(final["tasks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
