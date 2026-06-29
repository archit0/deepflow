"""Unit tests for task-list mode: the dispatcher, the store tools, and events.

The dispatcher only needs a ``worker_fn`` (no model/network), and the store
tools only need a fake runtime — so these are fast and deterministic.
"""

import json
from types import SimpleNamespace

from langgraph.types import Command

from deepflow import events
from deepflow.tasklist import (
    _batches,
    _merge,
    aggregate,
    dispatch,
    make_todos,
    store_tools,
    verify,
)
from deepflow.tasklist import (
    _Emitter as Emitter,
)


def _todos(n: int) -> dict:
    return {f"td{i:04d}": {"id": f"td{i:04d}", "content": f"task {i}", "status": "pending"} for i in range(n)}


def _drain_worker(seen: list[int]):
    """A worker that completes its batch and records how many to-dos it saw."""

    def fn(worker_id: str, batch: dict):  # noqa: ARG001
        seen.append(len(batch))
        updated = {tid: {**t, "status": "done", "result": "ok"} for tid, t in batch.items()}
        return updated, [{"returned": len(batch), "ids": list(batch)}]

    return fn


def _collector():
    seen: list[dict] = []

    def writer(payload: dict) -> None:
        if isinstance(payload, dict) and events.NAMESPACE in payload:
            seen.append(payload[events.NAMESPACE])

    return seen, Emitter(writer)


# --- dispatcher -------------------------------------------------------------
def test_dispatch_slices_into_disjoint_batches() -> None:
    seen: list[int] = []
    delta, rollup, batches = dispatch(_todos(500), _drain_worker(seen), batch_size=50, concurrency=8)
    assert batches == 10
    assert rollup == {"done": 500}
    assert len(delta) == 500
    assert max(seen) <= 50  # no worker ever saw more than its slice
    assert sum(seen) == 500  # batches are disjoint and cover everything


def test_dispatch_handles_uneven_final_batch() -> None:
    seen: list[int] = []
    _, rollup, batches = dispatch(_todos(205), _drain_worker(seen), batch_size=50, concurrency=8)
    assert batches == 5
    assert sorted(seen) == [5, 50, 50, 50, 50]
    assert rollup == {"done": 205}


def test_dispatch_retry_targets_failed_only() -> None:
    todos = _todos(100)
    # First pass: fail every 10th to-do.
    def flaky(worker_id: str, batch: dict):  # noqa: ARG001
        out = {}
        for tid, t in batch.items():
            failed = int(tid[2:]) % 10 == 0
            out[tid] = {**t, "status": "failed" if failed else "done"}
        return out, []

    delta1, agg1, _ = dispatch(todos, flaky, batch_size=25)
    assert agg1.get("failed") == 10
    merged = {**todos, **delta1}

    seen: list[int] = []
    # Retry pass: only status='failed' is re-dispatched.
    _, agg2, _ = dispatch(merged, _drain_worker(seen), batch_size=25, statuses=("failed",))
    assert sum(seen) == 10  # touched ONLY the 10 failures
    assert agg2 == {"done": 100}


def test_dispatch_empty_is_noop() -> None:
    delta, rollup, batches = dispatch({}, _drain_worker([]), batch_size=10)
    assert (delta, rollup, batches) == ({}, {}, 0)


# --- events -----------------------------------------------------------------
def test_dispatch_emits_events_in_order() -> None:
    seen, emit = _collector()
    dispatch(_todos(6), _drain_worker([]), batch_size=2, concurrency=4, emit=emit)
    names = [e["event"] for e in seen]

    assert names[0] == events.TASKLIST_PLAN
    assert names[-1] == events.TASKLIST_DONE
    assert names.count(events.BATCH_START) == 3
    assert names.count(events.BATCH_DONE) == 3
    assert events.WORKER_READ in names
    # every batch_start is emitted before any worker runs (so before any batch_done)
    assert names.index(events.BATCH_DONE) > names.index(events.BATCH_START)

    plan = seen[0]
    assert plan["total"] == 6
    assert plan["pending"] == 6
    assert plan["worker_count"] == 3

    start = next(e for e in seen if e["event"] == events.BATCH_START)
    assert start["size"] == 2
    assert len(start["todos"]) == 2  # a worker's whole visible context is just its slice
    assert all("id" in t and "content" in t for t in start["todos"])

    done = next(e for e in seen if e["event"] == events.TASKLIST_DONE)
    assert done["done"] == 6


def test_batch_start_shows_only_the_slice() -> None:
    seen, emit = _collector()
    dispatch(_todos(9), _drain_worker([]), batch_size=3, concurrency=3, emit=emit)
    starts = [e for e in seen if e["event"] == events.BATCH_START]
    # 3 workers, each sees exactly 3 to-dos, and the slices are disjoint.
    assert [s["size"] for s in starts] == [3, 3, 3]
    all_ids = [t["id"] for s in starts for t in s["todos"]]
    assert len(all_ids) == len(set(all_ids)) == 9


# --- store tools ------------------------------------------------------------
def _rt(state: dict):
    return SimpleNamespace(state=state, tool_call_id="call-1", stream_writer=None)


def _tools() -> dict:
    return {t.name: t for t in store_tools()}


def test_read_todos_paginates_and_filters() -> None:
    read = _tools()["read_todos"]
    state = {"tasks": _todos(5)}
    page = json.loads(read.func(runtime=_rt(state), status="pending", limit=2, offset=0))
    assert page["total"] == 5
    assert page["returned"] == 2
    assert len(page["todos"]) == 2  # never the whole store

    none = json.loads(read.func(runtime=_rt(state), status="done"))
    assert none["total"] == 0


def test_write_todos_updates_status_and_result() -> None:
    write = _tools()["write_todos"]
    state = {"tasks": _todos(3)}
    cmd = write.func(id="td0001", status="done", runtime=_rt(state), result="ok")
    assert isinstance(cmd, Command)
    assert cmd.update["tasks"]["td0001"]["status"] == "done"
    assert cmd.update["tasks"]["td0001"]["result"] == "ok"

    miss = write.func(id="nope", status="done", runtime=_rt(state))
    assert isinstance(miss, str)
    assert "Unknown" in miss


def test_add_and_count_todos() -> None:
    tools = _tools()
    cmd = tools["add_todos"].func(items=["a", "b", "c"], runtime=_rt({}))
    assert isinstance(cmd, Command)
    assert len(cmd.update["tasks"]) == 3
    assert all(t["status"] == "pending" for t in cmd.update["tasks"].values())

    counts = json.loads(tools["count_todos"].func(runtime=_rt({"tasks": _todos(4)})))
    assert counts["total"] == 4
    assert counts["by_status"] == {"pending": 4}


# --- helpers ----------------------------------------------------------------
def test_make_todos_and_merge_reducer() -> None:
    store = make_todos(["x", "y"])
    assert len(store) == 2
    assert all(t["status"] == "pending" for t in store.values())

    a = {"1": {"id": "1", "content": "a", "status": "pending"}}
    b = {"1": {"id": "1", "content": "a", "status": "done"}, "2": {"id": "2", "content": "b", "status": "pending"}}
    merged = _merge(a, b)
    assert merged["1"]["status"] == "done"  # update wins
    assert set(merged) == {"1", "2"}  # union, nothing deleted


def test_merge_is_defensive_against_non_dict() -> None:
    # A non-dict write must NEVER clobber the store — the existing dict is kept.
    store = {"1": {"id": "1", "content": "", "status": "done"}}
    assert _merge(store, ["a", "b"]) == store  # list write ignored, store preserved
    assert _merge(["x"], {"1": {"id": "1", "content": "", "status": "pending"}}) == {"1": {"id": "1", "content": "", "status": "pending"}}
    assert _merge(None, None) == {}


def _build_model():
    import os  # noqa: PLC0415

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(model="gpt-5.5")


def _tool_names(agent) -> set:
    names: set = set()
    for node in getattr(agent, "nodes", {}).values():
        by_name = getattr(getattr(node, "bound", None), "tools_by_name", None)
        if by_name:
            names |= set(by_name)
    return names


def test_workflow_agent_can_enable_todos() -> None:
    # The flag mounts task-list mode onto a workflow agent without a channel conflict.
    from deepflow import create_workflow_agent  # noqa: PLC0415

    agent = create_workflow_agent(model=_build_model(), enable_todos=True)
    names = _tool_names(agent)
    assert {"workflow", "process_todos", "count_todos", "add_todos"} <= names
    # the task store lives on its OWN `tasks` channel (merge reducer), separate from
    # deepagents' planning `todos` channel — so they never collide.
    assert type(agent.channels["tasks"]).__name__ == "BinaryOperatorAggregate"
    assert "todos" in agent.channels  # deepagents' planning todos still present, independently


# --- acceptance checks (deterministic verification) -------------------------
def test_make_todos_accepts_specs() -> None:
    store = make_todos(["plain", {"content": "x", "check": "pytest -q", "group": "g"}])
    todos = list(store.values())
    plain = next(t for t in todos if t["content"] == "plain")
    rich = next(t for t in todos if t["content"] == "x")
    assert "check" not in plain
    assert "group" not in plain
    assert rich["check"] == "pytest -q"
    assert rich["group"] == "g"


def test_dispatch_verifies_checks_and_flips_failures() -> None:
    todos = make_todos([{"content": "a", "check": "c1"}, {"content": "b", "check": "c2"}, "c"])

    def runner(cmd: str) -> tuple[bool, str]:
        # c1 passes, c2 fails; the engine — not the model — decides "done".
        return (cmd == "c1", "" if cmd == "c1" else "boom")

    delta, agg, _ = dispatch(todos, _drain_worker([]), batch_size=10, check_runner=runner)
    status = {t["content"]: t["status"] for t in delta.values()}
    assert status["a"] == "done"  # check passed
    assert status["b"] == "failed"  # check failed -> flipped despite the worker saying done
    assert status["c"] == "done"  # no check -> trusted
    assert agg.get("failed") == 1
    flipped = next(t for t in delta.values() if t["content"] == "b")
    assert flipped["result"].startswith("check failed")


def test_check_failure_emits_event() -> None:
    todos = make_todos([{"content": "a", "check": "x"}])
    seen, emit = _collector()
    dispatch(todos, _drain_worker([]), batch_size=5, check_runner=lambda _c: (False, "nope"), emit=emit)
    names = [e["event"] for e in seen]
    assert events.CHECK_FAILED in names
    ev = next(e for e in seen if e["event"] == events.CHECK_FAILED)
    assert ev["output"] == "nope"


# --- group co-location ------------------------------------------------------
def test_batches_colocate_same_group() -> None:
    todos = make_todos([
        {"content": "a", "group": "g1"},
        {"content": "b", "group": "g2"},
        {"content": "c", "group": "g1"},
        {"content": "d", "group": "g2"},
    ])
    batches = _batches(list(todos.items()), 2)
    groups_per_batch = [{t.get("group") for t in b.values()} for b in batches]
    assert all(len(g) == 1 for g in groups_per_batch)  # each batch is a single group, not a mix


def test_ungrouped_batches_preserve_order() -> None:
    todos = _todos(6)
    batches = _batches(list(todos.items()), 2)
    assert [t["id"] for b in batches for t in b.values()] == [f"td{i:04d}" for i in range(6)]


# --- sampled verification ---------------------------------------------------
def test_verify_samples_and_reverts() -> None:
    todos = {f"td{i:04d}": {"id": f"td{i:04d}", "content": f"t{i}", "status": "done", "result": "r"} for i in range(20)}

    def verifier(worker_id: str, batch: dict) -> dict:  # noqa: ARG001
        # Reject any to-do whose id ends in 0 (a stand-in for "confidently wrong").
        return {tid: {**t, "status": "failed" if tid.endswith("0") else "done"} for tid, t in batch.items()}

    delta, sampled, reverted = verify(todos, verifier, sample_rate=1.0, batch_size=5, concurrency=4)
    assert sampled == 20  # sample_rate 1.0 -> all done to-dos re-checked
    assert reverted == 2  # td0000, td0010
    assert set(delta) == {"td0000", "td0010"}
    assert all(t["status"] == "failed" for t in delta.values())


def test_verify_is_sublinear_sample() -> None:
    todos = {f"td{i:04d}": {"id": f"td{i:04d}", "content": "", "status": "done"} for i in range(100)}
    sampled_seen: list[int] = []

    def verifier(worker_id: str, batch: dict) -> dict:  # noqa: ARG001
        sampled_seen.append(len(batch))
        return batch  # confirm everything

    _, sampled, reverted = verify(todos, verifier, sample_rate=0.1, batch_size=50)
    assert sampled == 10  # ~1 in 10, not all 100
    assert reverted == 0
    assert sum(sampled_seen) == 10


def test_verify_noop_when_no_done() -> None:
    todos = {"1": {"id": "1", "content": "", "status": "pending"}}
    delta, sampled, reverted = verify(todos, lambda _w, b: b, sample_rate=0.5)
    assert (delta, sampled, reverted) == ({}, 0, 0)


def test_tasklist_agent_has_verify_tool() -> None:
    from deepflow import create_tasklist_agent  # noqa: PLC0415

    agent = create_tasklist_agent(model=_build_model())
    names = _tool_names(agent)
    assert {"process_todos", "verify_todos", "count_todos", "add_todos"} <= names


def test_aggregate_counts_by_status() -> None:
    store = {
        "1": {"id": "1", "content": "", "status": "done"},
        "2": {"id": "2", "content": "", "status": "done"},
        "3": {"id": "3", "content": "", "status": "failed"},
    }
    assert aggregate(store) == {"done": 2, "failed": 1}
