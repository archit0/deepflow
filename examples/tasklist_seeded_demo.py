#!/usr/bin/env python
"""Task-list mode WITH a defined to-do list (the user seeds the to-dos).

The store is seeded up front via ``make_todos([...])``. The orchestrator checks
the COUNT (not the contents), then calls ``process_todos`` once — the dispatcher
fans the to-dos out to worker deep-agents in disjoint batches. The visual shows,
per worker: its slice (its entire visible context), what it saw when it called
``read_todos``, and the status it wrote — so you can see each worker is isolated.

Reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` from the env. Model via
``DEMO_MODEL`` (default ``gpt-5.5``).

    uv run python examples/tasklist_seeded_demo.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasklist_view import col, make_renderer, print_board

from deepflow import create_tasklist_agent, make_todos

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

# A DEFINED task list — 12 to-dos the user already has.
TASKS = [
    "Summarize PostgreSQL in one line.",
    "Summarize Redis in one line.",
    "Summarize Kafka in one line.",
    "Summarize SQLite in one line.",
    "Summarize MongoDB in one line.",
    "Summarize Elasticsearch in one line.",
    "Summarize RabbitMQ in one line.",
    "Summarize Cassandra in one line.",
    "Summarize DynamoDB in one line.",
    "Summarize Neo4j in one line.",
    "Summarize ClickHouse in one line.",
    "Summarize CockroachDB in one line.",
]

ASK = (
    "There are pending to-dos in the store. Check `count_todos` first (do NOT try to read them all), "
    "then call `process_todos` exactly once with batch_size=4 (so the work fans out across several workers) "
    "to handle them all — instruct each worker to read its to-dos and write a one-line summary as the result. "
    "Finally tell me how many are done vs failed."
)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    print(col("deepflow — task-list mode · DEFINED to-do list", "bold"))
    print(f"model: {MODEL}  ·  {len(TASKS)} seeded to-dos  ·  batch_size=4 → 3 workers\n")

    agent = create_tasklist_agent(model=MODEL, batch_size=4, max_workers=3)
    seed = make_todos(TASKS)

    start = time.perf_counter()
    render = make_renderer(start)
    final_state = None
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": ASK}], "todos": seed}, stream_mode=["updates", "custom", "values"]):
        if mode == "values":
            final_state = chunk
            continue
        render(mode, chunk)

    if final_state:
        print_board(final_state.get("todos") or {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
