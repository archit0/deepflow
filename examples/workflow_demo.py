#!/usr/bin/env python
r"""Example 1 — Creating workflows (a DAG of sub-agents).

The agent authors a phase/step plan in a SINGLE call. The plan is a DAG: a step
depends on earlier steps (`depends_on`) and consumes their outputs via
`{{step_id}}`. Steps with no unmet dependency run in parallel; a later step fans
in once its dependencies finish. Each step runs in its own isolated sub-agent.

This demo nudges the model toward a classic **diamond**:

        research
        /      \\
  announcement  migration       (run in parallel — both depend on research)
        \\      /
        checklist                (fans in — depends on both)

Reads OPENAI_API_KEY / OPENAI_BASE_URL from the env. Model via DEMO_MODEL
(default gpt-5.5).

    uv run python examples/workflow_demo.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage
from tasklist_view import col, trunc

from deepflow import create_workflow_agent, events

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

SYSTEM_PROMPT = (
    "Use the `workflow` tool and build a DAG. Phase 1: a single `research` step. Phase 2: TWO steps that both "
    "`depends_on` research and run in parallel. Phase 3: a single step that `depends_on` BOTH phase-2 steps and "
    "combines them. Always consume an earlier step's output with `{{step_id}}` and list every id you reference in "
    "that step's `depends_on`."
)

TASK = (
    "Produce a launch plan for a new 'passwordless login' feature. Research it once; then, in parallel, draft "
    "(a) a short user-facing announcement and (b) developer migration notes; then assemble a one-page launch "
    "checklist that combines both. Use a workflow."
)


def render(mode, chunk, start) -> None:
    t = col(f"[{time.perf_counter() - start:5.1f}s]", "dim")
    if mode == "updates" and isinstance(chunk, dict):
        for upd in chunk.values():
            if not isinstance(upd, dict):
                continue
            for msg in upd.get("messages", []):
                if isinstance(msg, AIMessage):
                    if any(c["name"] == "workflow" for c in msg.tool_calls):
                        print(f"{t} {col('agent →', 'bold')} {col('workflow!', 'magenta')}  authors the DAG")
                    elif not msg.tool_calls and msg.text and msg.text.strip():
                        print(f"{t} {col('agent', 'bold')} {col('final answer', 'green')}")
    elif mode == "custom" and isinstance(chunk, dict) and events.NAMESPACE in chunk:
        ev = chunk[events.NAMESPACE]
        kind = ev["event"]
        if kind == events.PLAN:
            print(f"\n{t}   {col('└─ DAG plan', 'magenta')}  ·  {ev['phase_count']} phases / {ev['step_count']} steps")
            for phase in ev["phases"]:
                par = col("  (parallel)", "dim") if len(phase["steps"]) > 1 else ""
                print(f"{t}      {col('phase ' + str(phase['index']), 'blue')} · {phase['title']}{par}")
                for s in phase["steps"]:
                    dep = col(" ⇐ " + ", ".join(s["depends_on"]), "yellow") if s.get("depends_on") else col(" (root)", "dim")
                    print(f"{t}        {col('▸ ' + s['id'], 'cyan')}{dep}  {col(trunc(s.get('description') or '', 40), 'dim')}")
            print()
        elif kind == events.PHASE_START:
            print(f"{t}   {col('▶ phase ' + str(ev['index']), 'magenta')} · {ev['title']}")
        elif kind == events.STEP_DONE:
            print(f"{t}       {col('✓ ' + ev['id'], 'green')}")
        elif kind == events.STEP_ERROR:
            print(f"{t}       {col('✗ ' + ev['id'], 'red')}: {trunc(ev.get('error', ''))}")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    print(col("deepflow — example 1 · creating workflows (a DAG of sub-agents)", "bold"))
    print(f"model: {MODEL}\n")

    agent = create_workflow_agent(model=MODEL, system_prompt=SYSTEM_PROMPT)

    start = time.perf_counter()
    final = None
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": TASK}]}, stream_mode=["updates", "custom", "values"]):
        if mode == "values":
            final = chunk
            continue
        render(mode, chunk, start)

    if final:
        print("\n" + col("launch checklist:", "bold"), " ".join((final["messages"][-1].content or "").split())[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
