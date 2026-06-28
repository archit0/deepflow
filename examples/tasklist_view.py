"""Shared visual renderer for the task-list examples.

Turns an ``agent.stream(..., stream_mode=["updates", "custom"])`` into a
per-agent view: the orchestrator's own tool calls, then for each worker a boxed
panel showing exactly what it was handed, what it saw when it read, and what it
wrote — so you can SEE that every worker is isolated to its own slice.
"""

import sys
import time

from langchain_core.messages import AIMessage, ToolMessage

from deepflow import events

_C = {"dim": "2", "bold": "1", "cyan": "36", "green": "32", "yellow": "33", "magenta": "35", "red": "31", "blue": "34"}


def col(text: str, name: str) -> str:
    return text if not sys.stdout.isatty() else f"\033[{_C[name]}m{text}\033[0m"


def trunc(text: str, n: int = 58) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def make_renderer(start: float):
    """Return a ``render(mode, chunk)`` closure that prints the run as it streams."""
    names: dict[str, str] = {}  # tool_call_id -> tool name (for the orchestrator's own calls)
    plan: dict = {}
    slices: dict[str, list] = {}  # worker -> its assigned to-dos
    reads: dict[str, list] = {}  # worker -> its read_todos calls

    def stamp() -> str:
        return col(f"[{time.perf_counter() - start:5.1f}s]", "dim")

    def _orchestrator(chunk: dict) -> None:
        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        names[call.get("id", "")] = call["name"]
                        hot = call["name"] in ("process_todos", "add_todos")
                        tag = col(call["name"] + ("!" if call["name"] == "process_todos" else ""), "magenta" if hot else "yellow")
                        arg = call.get("args", {}).get("instruction") or (f"{len(call['args'].get('items', []))} items" if call["name"] == "add_todos" else "")
                        print(f"{stamp()} {col('orchestrator →', 'bold')} {tag}  {col(trunc(arg, 50), 'dim')}")
                    if not msg.tool_calls and msg.text and msg.text.strip():
                        print(f"{stamp()} {col('orchestrator', 'bold')} {col('final answer', 'green')}")
                elif isinstance(msg, ToolMessage):
                    name = names.get(msg.tool_call_id, "tool")
                    if name == "process_todos":
                        print(f"{stamp()}   {col('orchestrator ← rollup', 'blue')}: {trunc(msg.content, 80)}")
                    elif name in ("count_todos", "add_todos"):
                        print(f"{stamp()}   {col('🔧 ' + name, 'cyan')}: {trunc(msg.content, 80)}")

    def _worker_box(worker: str) -> None:
        total = plan.get("total", "?")
        my = slices.get(worker, [])
        print(f"\n  {col('┌─ ' + worker, 'magenta')}  {col('deepagent · no task/workflow · sees ONLY its slice', 'dim')}")
        print(f"  {col('│', 'magenta')}  context = its entire visible to-do list ({len(my)} of {total} global):")
        for todo in my:
            print(f"  {col('│', 'magenta')}    • {col(todo['id'], 'dim')}  {trunc(todo['content'])}")
        for read in reads.get(worker, []):
            ret, tot = read.get("returned"), plan.get("total", "?")
            hidden = (tot - ret) if isinstance(tot, int) and isinstance(ret, int) else "?"
            print(f"  {col('│', 'magenta')}  {col('read_todos(pending)', 'cyan')} → returned {ret}   {col(f'(global store has {tot}; the other {hidden} are invisible to it)', 'dim')}")

    def _worker_results(worker: str, results: list) -> None:
        for res in results:
            mark = col("✓", "green") if res["status"] == "done" else col("✗", "red")
            print(f"  {col('│', 'magenta')}  {mark} {col(res['id'], 'dim')}  {res['status']}  {col(trunc(res.get('result', ''), 50), 'dim')}")
        print(f"  {col('└─ ' + worker + ' done', 'magenta')}")

    def _events(ev: dict) -> None:
        kind = ev["event"]
        if kind == events.TASKLIST_PLAN:
            plan.update(ev)
            print(
                f"\n{stamp()}   {col('└─ dispatch plan', 'magenta')}: {ev['pending']} pending · "
                f"batch_size {ev['batch_size']} → {col(str(ev['worker_count']) + ' workers', 'bold')} "
                f"(not {ev['pending']} agents)"
            )
        elif kind == events.BATCH_START:
            slices[ev["worker"]] = ev["todos"]
        elif kind == events.WORKER_READ:
            reads.setdefault(ev["worker"], []).append(ev)
        elif kind == events.BATCH_DONE:
            _worker_box(ev["worker"])
            _worker_results(ev["worker"], ev.get("results", []))
        elif kind == events.TASKLIST_DONE:
            roll = " · ".join(f"{k}={ev[k]}" for k in ("done", "failed", "pending", "in_progress") if ev.get(k))
            print(f"\n{stamp()}   {col('tasklist done', 'green')}: {roll}  {col('← all the orchestrator gets back', 'dim')}")

    def render(mode: str, chunk) -> None:
        if mode == "updates" and isinstance(chunk, dict):
            _orchestrator(chunk)
        elif mode == "custom" and isinstance(chunk, dict) and events.NAMESPACE in chunk:
            _events(chunk[events.NAMESPACE])

    return render


def print_board(final_todos: dict) -> None:
    """Print the final to-do board (status + result per to-do)."""
    done = sum(1 for t in final_todos.values() if t["status"] == "done")
    failed = sum(1 for t in final_todos.values() if t["status"] == "failed")
    pending = sum(1 for t in final_todos.values() if t["status"] == "pending")
    print(f"\n{col('final board', 'bold')}: {done} done · {failed} failed · {pending} pending\n")
    for todo in final_todos.values():
        mark = col("✓", "green") if todo["status"] == "done" else (col("✗", "red") if todo["status"] == "failed" else "·")
        print(f"  {mark} {trunc(todo['content'], 50)}")
        if todo.get("result"):
            print(f"      {col('→ ' + trunc(todo['result'], 80), 'dim')}")
