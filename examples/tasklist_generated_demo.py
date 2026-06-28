#!/usr/bin/env python
"""Task-list mode WITHOUT a defined to-do list (the agent generates it).

No to-dos are seeded. The orchestrator is given an OBJECTIVE, breaks it into
concrete to-dos itself via ``add_todos``, then calls ``process_todos`` to fan
them out to worker deep-agents in disjoint batches. Same per-worker visual as the
seeded demo — so you can see that whether the list is user-given or AI-built, the
dispatch + isolation is identical.

Reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` from the env. Model via
``DEMO_MODEL`` (default ``gpt-5.5``).

    uv run python examples/tasklist_generated_demo.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasklist_view import col, make_renderer, print_board

from deepflow import create_tasklist_agent

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

# No to-do list — just an objective. The agent invents the to-dos.
ASK = (
    "I want to harden a typical web application's security before launch. "
    "First, use `add_todos` to create a list of about 8 concrete, distinct security checks to perform "
    "(e.g. auth, input validation, secrets, dependencies, transport, headers, rate-limiting, logging). "
    "Then call `process_todos` exactly once with batch_size=3 (so the work fans out across several workers) — "
    "instruct each worker to read its to-dos, perform its check (describe what to verify), write a one-line "
    "result, and mark each to-do 'done' once assessed (use 'failed' ONLY if a check genuinely cannot be performed). "
    "Finally tell me how many checks are done vs failed."
)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    print(col("deepflow — task-list mode · AI-GENERATED to-do list", "bold"))
    print(f"model: {MODEL}  ·  no seeded to-dos (the agent builds them)  ·  batch_size=3\n")

    agent = create_tasklist_agent(model=MODEL, batch_size=3, max_workers=4)

    start = time.perf_counter()
    render = make_renderer(start)
    final_state = None
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": ASK}]}, stream_mode=["updates", "custom", "values"]):
        if mode == "values":
            final_state = chunk
            continue
        render(mode, chunk)

    if final_state:
        print_board(final_state.get("todos") or {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
