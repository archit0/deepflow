#!/usr/bin/env python
"""Live streaming demo: a deepflow agent builds & tests a small Python library.

The agent authors ONE workflow — a build step per function (in parallel), then a
verify step — and we render every event as it streams: the plan up front, then
phases and steps starting/finishing in real time, with live activity from inside
each step's sub-agent.

Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the env. Model via `DEMO_MODEL`
(default `gpt-5.5`). Runs real `python3` in a throwaway temp directory.

    uv run python examples/build_library_demo.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from deepagents.backends import LocalShellBackend

from deepflow import create_workflow_agent, events

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")

FUNCS = [
    ("add", "add(a, b) returns the sum of two numbers"),
    ("is_even", "is_even(n) returns True if n is even, else False"),
    ("reverse_string", "reverse_string(s) returns the string reversed"),
    ("factorial", "factorial(n) returns n!; factorial(0) == 1"),
]
SPEC = "\n".join(f"- {name}.py: {desc}" for name, desc in FUNCS)

SYSTEM_PROMPT = f"""You are a coding agent with a shell (`execute`) and filesystem tools, working in the current directory.

Build a small Python library — implement each function in its own file:
{SPEC}

You MUST do this as ONE workflow, two phases:
- Phase 1 "Build": one step PER function (ids b1, b2, ...), each with a short `description` like "Build add".
  Each step writes <name>.py and test_<name>.py (assert-based, prints OK), then runs `python3 test_<name>.py`.
- Phase 2 "Verify": one step (description "Run all tests") that depends_on every build step and runs all the tests.
"""

TASK = "Build the library and run all the tests. Begin."

_C = {"dim": "2", "bold": "1", "cyan": "36", "green": "32", "yellow": "33", "magenta": "35", "red": "31"}


def col(text, name):
    return text if not sys.stdout.isatty() else f"\033[{_C[name]}m{text}\033[0m"


def render(ev, start):
    """Render one deepflow event — the schema the engine actually emits."""
    t = col(f"[{time.perf_counter() - start:5.1f}s]", "dim")
    kind = ev["event"]
    if kind == events.PLAN:
        print(f"{t} {col('plan', 'magenta')} {ev['phase_count']} phases / {ev['step_count']} steps")
        for ph in ev["phases"]:
            steps = ph["steps"]
            shape = "parallel" if len(steps) > 1 else "single"
            header = f"Phase {ph['index']} · {ph['title']}"
            print(f"        {col(header, 'bold')} ({len(steps)} {shape})")
            for s in steps:
                dep = col(f" <= {','.join(s['depends_on'])}", "dim") if s["depends_on"] else ""
                print(f"          {col('-', 'green')} {s['id']}{dep}  {s.get('description') or ''}")
    elif kind == events.PHASE_START:
        print(f"{t} {col('phase', 'magenta')} #{ev['index']} {col(ev['title'], 'bold')}")
    elif kind == events.STEP_START:
        print(f"{t}   {col('start', 'yellow')} {ev['id']}")
    elif kind == events.STEP_EVENT:
        detail = ev.get("tools") or ev.get("preview") or ev.get("name") or ""
        print(f"{t}     {col(ev['id'], 'cyan')} {ev.get('kind')}: {str(detail)[:60]}")
    elif kind == events.STEP_DONE:
        print(f"{t}   {col('done ', 'green')} {ev['id']}")
    elif kind == events.STEP_ERROR:
        print(f"{t}   {col('error', 'red')} {ev['id']}: {ev.get('error', '')[:60]}")
    elif kind == events.WORKFLOW_DONE:
        print(f"{t} {col('workflow done', 'green')}")


def verify(workdir):
    tests = sorted(Path(workdir).glob("test_*.py"))

    def passes(test):
        try:
            r = subprocess.run([sys.executable, test.name], cwd=workdir, capture_output=True, timeout=30, check=False)
        except (subprocess.SubprocessError, OSError):
            return False
        return r.returncode == 0

    return len(list(Path(workdir).glob("*.py"))), sum(passes(t) for t in tests), len(tests)


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    workdir = tempfile.mkdtemp(prefix="deepflow_demo_")
    print(col("deepflow — live streaming workflow demo", "bold"))
    print(f"model: {MODEL}  ·  workdir: {workdir}\n")

    agent = create_workflow_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=workdir, inherit_env=True),
    )

    start = time.perf_counter()
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": TASK}]}, stream_mode=["updates", "custom"]):
        if mode == "custom" and isinstance(chunk, dict) and events.NAMESPACE in chunk:
            render(chunk[events.NAMESPACE], start)

    files, passed, total = verify(workdir)
    print()
    color = "green" if total and passed == total else "red"
    print(col(f"built {files} files · tests {passed}/{total} passing", color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
