#!/usr/bin/env python
"""Demo: the agent does normal tool calls first, THEN starts a workflow.

The agent isn't forced into a workflow up front. It first explores with ordinary
tool calls (list the notes, read one to learn the format), and only *then*
authors a workflow to summarize every note in parallel — because by that point
it knows what to fan out over.

The stream shows both: the orchestrator's plain tool calls (🧠 → / 🔧) and then
the workflow events (plan / phase / step).

Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the env. Model via `DEMO_MODEL`
(default `gpt-5.5`).

    uv run python examples/explore_then_workflow_demo.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage, ToolMessage

from deepflow import create_workflow_agent, events

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

NOTES = {
    "note_postgres.md": "PostgreSQL: relational, SQL-standard, strong transactions, JSONB, great general-purpose default.",
    "note_redis.md": "Redis: in-memory key-value store, sub-millisecond, used for caching, queues, rate limiting.",
    "note_kafka.md": "Kafka: distributed event log, high-throughput pub/sub, backbone for streaming pipelines.",
    "note_sqlite.md": "SQLite: embedded, serverless, single-file SQL database; perfect for local apps and tests.",
}

SYSTEM_PROMPT = """You are an assistant with filesystem + shell tools AND a `workflow` tool.

There are several note files in the current directory. Do this in order:

1. EXPLORE FIRST with ordinary tool calls — list the directory, and read ONE note to see the format. Do NOT use the `workflow` tool yet.
2. THEN summarize every note in parallel by authoring ONE workflow: phase 1 has one step per note file (each step reads its note and writes a one-line summary to summaries/<name>.md); phase 2 has a single step that depends on all of them and writes index.md listing every summary.

Use the workflow only for the bulk parallel summarization, after you've explored.
"""

TASK = "Summarize all the notes in this directory. Explore first, then do the bulk work as a workflow."

_C = {"dim": "2", "bold": "1", "cyan": "36", "green": "32", "yellow": "33", "magenta": "35", "red": "31"}


def col(text, name):
    return text if not sys.stdout.isatty() else f"\033[{_C[name]}m{text}\033[0m"


def trunc(text, n=70):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def render(mode, chunk, start, names):
    t = col(f"[{time.perf_counter() - start:5.1f}s]", "dim")
    if mode == "updates" and isinstance(chunk, dict):
        # The orchestrator's own, ordinary tool calls — before and after the workflow.
        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        names[call.get("id", "")] = call["name"]
                        arg = next((str(v) for k, v in (call.get("args") or {}).items() if k in ("file_path", "command", "path")), "")
                        tag = col("workflow!", "magenta") if call["name"] == "workflow" else col(call["name"], "yellow")
                        print(f"{t} {col('agent →', 'bold')} {tag} {trunc(arg, 40)}")
                    if not msg.tool_calls and msg.text and msg.text.strip():
                        print(f"{t} {col('agent', 'bold')} {col('final answer', 'green')}")
                elif isinstance(msg, ToolMessage):
                    name = names.get(msg.tool_call_id, "tool")
                    if name != "workflow":  # workflow internals are shown via custom events below
                        print(f"{t}   {col('🔧 ' + name, 'cyan')}: {trunc(msg.content)}")
    elif mode == "custom" and isinstance(chunk, dict) and events.NAMESPACE in chunk:
        ev = chunk[events.NAMESPACE]
        kind = ev["event"]
        if kind == events.PLAN:
            print(f"{t}   {col('└─ workflow plan', 'magenta')} {ev['phase_count']} phases / {ev['step_count']} steps")
        elif kind == events.PHASE_START:
            print(f"{t}   {col('   phase', 'magenta')} #{ev['index']} {ev['title']}")
        elif kind == events.STEP_DONE:
            print(f"{t}   {col('   ✓', 'green')} {ev['id']}")
        elif kind == events.STEP_ERROR:
            print(f"{t}   {col('   ✗', 'red')} {ev['id']}: {trunc(ev.get('error', ''))}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    workdir = tempfile.mkdtemp(prefix="deepflow_explore_")
    for name, body in NOTES.items():
        Path(workdir, name).write_text(body + "\n")

    print(col("deepflow — explore first, then workflow", "bold"))
    print(f"model: {MODEL}  ·  workdir: {workdir}  ·  {len(NOTES)} notes seeded\n")

    agent = create_workflow_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=workdir, inherit_env=True),
    )

    start = time.perf_counter()
    names: dict[str, str] = {}
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": TASK}]}, stream_mode=["updates", "custom"]):
        render(mode, chunk, start, names)

    summaries = len(list(Path(workdir, "summaries").glob("*.md"))) if Path(workdir, "summaries").exists() else 0
    print(f"\n{col('done', 'green')} · {summaries} summaries · index.md={'yes' if Path(workdir, 'index.md').exists() else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
