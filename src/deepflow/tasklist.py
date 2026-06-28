"""Task-list mode — scalable, store-backed dispatch over many to-dos.

Where ``workflow`` mode is for a small, fixed plan the model authors up front,
task-list mode is for the opposite: a task that explodes into hundreds or
thousands of homogeneous to-dos. Keeping them all in shared state would blow up
every agent's context; authoring a workflow with one step per to-do would blow up
the orchestrator's. So instead:

- A **to-do store** lives in state but is never injected wholesale into a prompt.
  Agents touch it only through paginated/filtered tools.
- The **orchestrator** plans from *counts* (``count_todos``), can populate the
  store itself (``add_todos``), and calls ``process_todos`` — it never holds the
  to-dos.
- ``process_todos`` runs a deterministic **dispatcher** that partitions the
  pending to-dos into **disjoint batches** and runs a handful of **workers** in
  parallel. Each worker is handed only its batch (so it never sees more than
  ``batch_size`` to-dos) and writes status per to-do. Disjoint batches ⇒ no race,
  no double-processing; the engine assigns, workers never self-claim.
- **Workers are full Deep Agents minus sub-agent/workflow creation**: filesystem +
  ``execute`` + summarization/compaction (so they survive large tool results),
  plus the to-do tools — but no ``task`` and no ``workflow``.

Context budget: orchestrator ``O(log N)`` (a status rollup), worker ``O(batch)``,
store ``O(N)`` but only ever in state. That is what scales to 10k+ to-dos.

Every stage streams (see :mod:`deepflow.events`): ``tasklist_plan`` →
``batch_start`` → ``worker_read`` → ``batch_done`` → ``tasklist_done``.
"""

import contextvars
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepflow import events

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
"""Default number of to-dos handed to a single worker."""

DEFAULT_MAX_WORKERS = 8
"""Default number of workers running at once."""

Status = Literal["pending", "in_progress", "done", "failed"]


class Todo(TypedDict):
    """One unit of work in the store."""

    id: str
    content: str
    status: Status
    result: NotRequired[str]


def _merge(old: dict[str, Todo] | None, new: Any) -> dict[str, Todo]:
    """Reducer: union by id (append + update, never delete). Concurrency-safe.

    Defensive: a non-dict write never clobbers the store — the existing dict is
    kept. (The store lives on its own ``tasks`` channel, so this is belt-and-braces.)
    """
    if not isinstance(new, dict):
        return old if isinstance(old, dict) else {}
    base = old if isinstance(old, dict) else {}
    return {**base, **new}


class TaskListState(AgentState):
    """Agent state extended with the shared task store.

    The store lives on its own ``tasks`` channel (NOT ``todos``) so it never
    aliases the planning ``todos`` channel that ``deepagents`` mounts — both can
    coexist on one agent (e.g. ``create_workflow_agent(enable_todos=True)``).
    """

    tasks: Annotated[NotRequired[dict[str, Todo]], _merge]


def aggregate(todos: Any) -> dict[str, int]:
    """Status rollup: counts by status. This is all the orchestrator ever sees."""
    counts: dict[str, int] = {}
    if not isinstance(todos, dict):
        return counts
    for todo in todos.values():
        counts[todo["status"]] = counts.get(todo["status"], 0) + 1
    return counts


def make_todos(items: Sequence[str]) -> dict[str, Todo]:
    """Build an initial task store from a list of contents (each starts pending).

    Convenience for seeding a run: ``agent.invoke({"messages": ..., "tasks":
    make_todos([...])})``.
    """
    store: dict[str, Todo] = {}
    for content in items:
        tid = "td_" + uuid.uuid4().hex[:8]
        store[tid] = {"id": tid, "content": content, "status": "pending"}
    return store


# --------------------------------------------------------------------------- #
# Dispatcher (deterministic; the heart of the mode)
# --------------------------------------------------------------------------- #
# A worker takes (worker_id, {id: todo}) and returns (updated {id: todo}, reads),
# where ``reads`` is a list of the read_todos calls it made (for the event stream).
ReadEvent = dict[str, Any]
WorkerResult = tuple[dict[str, Todo], list[ReadEvent]]
WorkerFn = Callable[[str, dict[str, Todo]], WorkerResult]


class _Emitter:
    """Best-effort writer to the LangGraph custom stream; never raises."""

    def __init__(self, writer: Callable[[dict], Any] | None) -> None:
        self._writer = writer

    def __call__(self, event: str, **data: Any) -> None:
        if self._writer is None:
            return
        try:
            self._writer({events.NAMESPACE: {"event": event, **data}})
        except Exception:  # noqa: BLE001 - progress emission must never break a run
            logger.debug("deepflow stream writer raised; ignoring", exc_info=True)


def _batches(items: list[tuple[str, Todo]], size: int) -> list[dict[str, Todo]]:
    return [dict(items[i : i + size]) for i in range(0, len(items), size)]


def _status_counts(todos: dict[str, Todo]) -> dict[str, int]:
    counts = aggregate(todos)
    return {status: counts.get(status, 0) for status in ("done", "failed", "pending", "in_progress")}


def dispatch(
    todos: dict[str, Todo],
    worker_fn: WorkerFn,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_MAX_WORKERS,
    statuses: tuple[str, ...] = ("pending",),
    emit: _Emitter | None = None,
) -> tuple[dict[str, Todo], dict[str, int], int]:
    """Partition the matching to-dos into disjoint batches and drain them.

    Each batch is handed to one ``worker_fn`` call, so a worker only ever sees
    ``batch_size`` to-dos. Batches run concurrently up to ``concurrency``.

    Context is captured **here** (the caller's / tool's thread) and replayed in
    each worker thread via ``ctx.run``, so callbacks/tracing propagate correctly.

    Returns ``(delta, rollup, batch_count)`` where ``delta`` is just the changed
    to-dos (safe to merge back into the store).
    """
    emit = emit or _Emitter(None)
    # Reset matched to-dos to `pending` in the worker's batch view, so a retry
    # (statuses includes "failed") is drained even if a worker filters on pending.
    pending = [(tid, {**todo, "status": "pending"}) for tid, todo in todos.items() if todo["status"] in statuses]
    batches = _batches(pending, batch_size)
    emit(events.TASKLIST_PLAN, total=len(todos), pending=len(pending), batch_size=batch_size, worker_count=len(batches))

    delta: dict[str, Todo] = {}
    if not batches:
        emit(events.TASKLIST_DONE, **_status_counts(todos))
        return delta, aggregate(todos), 0

    for i, batch in enumerate(batches):
        emit(events.BATCH_START, worker=f"w{i}", size=len(batch), todos=[{"id": tid, "content": t["content"]} for tid, t in batch.items()])

    ctxs = [contextvars.copy_context() for _ in batches]  # one snapshot per batch, taken in THIS thread
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(ctxs[i].run, worker_fn, f"w{i}", batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            wid = f"w{futures[future]}"
            try:
                updated, reads = future.result()
            except Exception as exc:
                logger.exception("deepflow task-list worker '%s' failed", wid)
                emit(events.BATCH_DONE, worker=wid, results=[], error=str(exc))
                continue
            for read in reads:
                emit(events.WORKER_READ, worker=wid, **read)
            results = [{"id": tid, "status": u["status"], "result": u.get("result", "")} for tid, u in updated.items()]
            emit(events.BATCH_DONE, worker=wid, results=results)
            delta.update(updated)

    merged = {**todos, **delta}
    emit(events.TASKLIST_DONE, **_status_counts(merged))
    return delta, aggregate(merged), len(batches)


# --------------------------------------------------------------------------- #
# To-do store tools (shared by workers; orchestrator gets a subset + dispatch)
# --------------------------------------------------------------------------- #
class _AddArgs(BaseModel):
    items: list[str] = Field(description="To-do contents to append; each becomes a pending to-do with a fresh id.")


class _ReadArgs(BaseModel):
    status: str | None = Field(default=None, description="Filter by status (pending/in_progress/done/failed).")
    limit: int = Field(default=20, description="Max to-dos to return (a PAGE — never the whole store).")
    offset: int = Field(default=0, description="Page offset.")


class _WriteArgs(BaseModel):
    id: str = Field(description="The to-do to update.")
    status: Status = Field(description="New status: in_progress / done / failed.")
    result: str | None = Field(default=None, description="Short result or failure reason.")


def store_tools() -> list[BaseTool]:
    """The four store tools: ``add_todos`` / ``read_todos`` / ``write_todos`` / ``count_todos``."""

    def add_todos(items: list[str], runtime: ToolRuntime) -> Command:
        new = make_todos(items)
        return Command(update={"tasks": new, "messages": [ToolMessage(f"Added {len(new)} to-dos.", tool_call_id=runtime.tool_call_id)]})

    def read_todos(runtime: ToolRuntime, status: str | None = None, limit: int = 20, offset: int = 0) -> str:
        todos = runtime.state.get("tasks") or {}
        items = [t for t in todos.values() if status is None or t["status"] == status]
        page = items[offset : offset + limit]
        return json.dumps({"total": len(items), "offset": offset, "returned": len(page), "todos": page})

    def write_todos(id: str, status: Status, runtime: ToolRuntime, result: str | None = None) -> Command | str:  # noqa: A002 - tool arg name is model-facing
        todos = runtime.state.get("tasks") or {}
        if id not in todos:
            return f"Unknown to-do '{id}'."
        updated: Todo = {**todos[id], "status": status}
        if result is not None:
            updated["result"] = result
        return Command(update={"tasks": {id: updated}, "messages": [ToolMessage(f"{id} -> {status}", tool_call_id=runtime.tool_call_id)]})

    def count_todos(runtime: ToolRuntime) -> str:
        todos = runtime.state.get("tasks") or {}
        return json.dumps({"total": len(todos), "by_status": aggregate(todos)})

    add_desc = "Append new to-dos (each starts pending)."
    read_desc = "Read a PAGE of to-dos (filter by status, paginate). Never returns the whole store."
    write_desc = "Set a to-do's status (in_progress/done/failed) and an optional short result."
    count_desc = "Status rollup: total + counts by status (cheap — read this, not all to-dos)."
    return [
        StructuredTool.from_function(func=add_todos, name="add_todos", description=add_desc, args_schema=_AddArgs, infer_schema=False),
        StructuredTool.from_function(func=read_todos, name="read_todos", description=read_desc, args_schema=_ReadArgs, infer_schema=False),
        StructuredTool.from_function(func=write_todos, name="write_todos", description=write_desc, args_schema=_WriteArgs, infer_schema=False),
        StructuredTool.from_function(func=count_todos, name="count_todos", description=count_desc),
    ]


WORKER_PROMPT = (
    "You are a focused worker. You have been handed a small batch of to-dos — ONLY yours, nobody else's.\n"
    "Call `read_todos()` (no filter) to see EVERY to-do in your batch, do each one with your tools, then call "
    "`write_todos(id, 'done', <short result>)` for it (or `'failed'` with the reason). Cover every to-do you were "
    "given — including any that come back as `failed` on a retry — then stop. You cannot create sub-agents or "
    "workflows; just drain your batch."
)

ORCHESTRATOR_PROMPT = """## Task-list mode

You manage a potentially huge to-do store that you must NOT read in full. Plan from counts, not contents.

- `count_todos()` — the cheap status rollup. Start here.
- `add_todos([...])` — create to-dos (use this if the user gave you an objective instead of a ready-made list).
- `process_todos(instruction, batch_size?, max_workers?)` — THE dispatcher. It partitions the pending to-dos into
  disjoint batches and runs workers in parallel; each worker drains only its batch and writes status. You get back a
  status rollup, not the contents. Re-run it to retry anything still pending/failed.

Choose `batch_size` so each worker gets a sane slice (e.g. ~25-200 to-dos), never one worker per to-do."""


class _ProcessArgs(BaseModel):
    instruction: str = Field(description="What each worker should do with every to-do in its batch.")
    batch_size: int | None = Field(default=None, description="To-dos per worker (defaults to the agent's configured value).")
    max_workers: int | None = Field(default=None, description="Max workers running at once (defaults to the agent's configured value).")


# --------------------------------------------------------------------------- #
# Workers: full Deep Agent goodness MINUS sub-agent/workflow creation
# --------------------------------------------------------------------------- #
class TodoStoreMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Adds the shared to-do store + its four tools to an agent (used for workers)."""

    state_schema = TaskListState

    def __init__(self, *, system_prompt: str = WORKER_PROMPT) -> None:
        """Bind the four store tools and the worker guidance."""
        super().__init__()
        self.tools = store_tools()
        self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Append the worker guidance to the system prompt, then defer."""
        return handler(_augment(request, self.system_prompt))


def _base_middleware(model: Any, *, backend: Any = None) -> list[AgentMiddleware]:
    """Deep Agent goodness with NO sub-agent/workflow creation: filesystem + compaction.

    Shared by both workers and the orchestrator. Built from explicit deepagents
    middleware (rather than ``create_deep_agent``) precisely so the agent gets the
    filesystem + ``execute`` + summarization tools but neither ``task`` nor
    ``workflow`` — and so the built-in ``write_todos`` never collides with our store.
    """
    from deepagents.backends import StateBackend  # noqa: PLC0415
    from deepagents.middleware.filesystem import FilesystemMiddleware  # noqa: PLC0415
    from langchain.agents.middleware import SummarizationMiddleware  # noqa: PLC0415

    return [
        FilesystemMiddleware(backend=backend or StateBackend()),
        SummarizationMiddleware(model=model, max_tokens_before_summary=60000),
    ]


def worker_middleware(model: Any, *, backend: Any = None, system_prompt: str = WORKER_PROMPT) -> list[AgentMiddleware]:
    """The worker's middleware stack: :func:`_base_middleware` + the to-do store tools."""
    return [*_base_middleware(model, backend=backend), TodoStoreMiddleware(system_prompt=system_prompt)]


def orchestrator_middleware(
    model: Any,
    *,
    worker_model: Any = None,
    tools: Sequence[Any] | None = None,
    backend: Any = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    worker_system_prompt: str = WORKER_PROMPT,
    system_prompt: str | None = ORCHESTRATOR_PROMPT,
) -> list[AgentMiddleware]:
    """The orchestrator's middleware stack: :func:`_base_middleware` + :class:`TaskListMiddleware`."""
    return [
        *_base_middleware(model, backend=backend),
        TaskListMiddleware(
            model=worker_model or model,
            tools=tools,
            backend=backend,
            batch_size=batch_size,
            max_workers=max_workers,
            worker_system_prompt=worker_system_prompt,
            system_prompt=system_prompt,
        ),
    ]


def make_worker(model: Any, *, tools: Sequence[Any] | None = None, backend: Any = None, system_prompt: str = WORKER_PROMPT) -> Any:
    """Build a worker agent: a Deep Agent with NO ``task``/``workflow`` tool.

    Uses langchain ``create_agent`` (not ``create_deep_agent``) on purpose — that
    is precisely what omits sub-agent/workflow creation while keeping the
    filesystem + compaction goodness. It drains its assigned slice and nothing else.
    """
    from langchain.agents import create_agent  # noqa: PLC0415

    return create_agent(model, tools=list(tools or []), middleware=worker_middleware(model, backend=backend, system_prompt=system_prompt))


def worker_tool_names(model: Any, *, backend: Any = None) -> list[str]:
    """The exact tool names a worker is bound to (for verifying no ``task``/``workflow``)."""
    names: list[str] = []
    for mw in worker_middleware(model, backend=backend):
        names.extend(getattr(t, "name", str(t)) for t in (getattr(mw, "tools", None) or []))
    return sorted(names)


def agent_worker_fn(worker: Any, instruction: str) -> WorkerFn:
    """Adapt a worker agent into a :data:`WorkerFn`: seed it with ONLY its batch.

    The worker's state holds only its batch, so its ``read_todos`` provably cannot
    see anything else. We surface each ``read_todos`` call as a ``reads`` entry so
    the dispatcher can stream ``worker_read`` events.
    """

    def fn(worker_id: str, batch: dict[str, Todo]) -> WorkerResult:  # noqa: ARG001 - id is used by the caller for events
        prompt = (
            f"{instruction}\n\nYou have {len(batch)} assigned to-dos. Read ALL of them with `read_todos()` (no filter), "
            "complete each, and `write_todos(id, 'done'|'failed', result)` for every one. Cover every to-do — including "
            "any already marked `failed` (a retry) — then stop."
        )
        result = worker.invoke({"messages": [HumanMessage(content=prompt)], "tasks": dict(batch)})
        updated = result.get("tasks") or dict(batch)
        reads: list[ReadEvent] = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "read_todos":
                try:
                    data = json.loads(msg.content)
                except (ValueError, TypeError):
                    continue
                reads.append({"returned": data.get("returned"), "ids": [t["id"] for t in data.get("todos", [])]})
        return updated, reads

    return fn


# --------------------------------------------------------------------------- #
# Orchestrator middleware: count_todos + add_todos + process_todos
# --------------------------------------------------------------------------- #
def _augment(request: ModelRequest[ContextT], system_prompt: str | None) -> ModelRequest[ContextT]:
    if system_prompt is None:
        return request
    current = request.system_message
    text = current.text if isinstance(current, SystemMessage) else (str(current) if current else "")
    joined = f"{text}\n\n{system_prompt}" if text else system_prompt
    return request.override(system_message=SystemMessage(content=joined))


class TaskListMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Gives the orchestrator ``count_todos`` + ``add_todos`` + ``process_todos``.

    ``process_todos`` is the only place that spawns workers — workers never get
    this tool, so they cannot start their own dispatch.

    Args:
        model: Model used to build the worker sub-agents.
        tools: Tools shared with the workers (e.g. domain tools each task needs).
        backend: Backend for the workers' filesystem/shell (defaults to an
            in-state filesystem). Pass a sandbox/shell backend for real file/shell work.
        batch_size: Default to-dos per worker.
        max_workers: Default workers running at once.
        worker_system_prompt: System prompt for the workers.
        worker_factory: Optional override returning a fresh worker runnable.
        system_prompt: Orchestrator guidance appended to its system prompt.
    """

    state_schema = TaskListState

    def __init__(
        self,
        *,
        model: Any,
        tools: Sequence[Any] | None = None,
        backend: Any = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        worker_system_prompt: str = WORKER_PROMPT,
        worker_factory: Callable[[], Any] | None = None,
        system_prompt: str | None = ORCHESTRATOR_PROMPT,
    ) -> None:
        """Build the orchestrator tool set from the given worker config."""
        super().__init__()
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._worker_factory = worker_factory or (lambda: make_worker(model, tools=tools, backend=backend, system_prompt=worker_system_prompt))
        self.system_prompt = system_prompt
        orchestrator_tools = [t for t in store_tools() if t.name in {"count_todos", "add_todos"}]
        self.tools = [*orchestrator_tools, self._process_tool()]

    def _process_tool(self) -> BaseTool:
        def process_todos(instruction: str, runtime: ToolRuntime, batch_size: int | None = None, max_workers: int | None = None) -> Command:
            emit = _Emitter(getattr(runtime, "stream_writer", None))
            todos = runtime.state.get("tasks") or {}
            size = batch_size or self._batch_size
            workers = max_workers or self._max_workers
            pending = sum(1 for t in todos.values() if t["status"] in ("pending", "failed"))
            worker_fn = agent_worker_fn(self._worker_factory(), instruction)
            delta, rollup, batches = dispatch(todos, worker_fn, batch_size=size, concurrency=workers, statuses=("pending", "failed"), emit=emit)
            summary = f"Dispatched {pending} to-dos across {batches} workers (batch<={size}). Status now: {json.dumps(rollup)}"
            return Command(update={"tasks": delta, "messages": [ToolMessage(summary, tool_call_id=runtime.tool_call_id)]})

        return StructuredTool.from_function(
            func=process_todos,
            name="process_todos",
            description="Drain all pending/failed to-dos: partition into disjoint batches, run workers in parallel, return a status rollup.",
            args_schema=_ProcessArgs,
            infer_schema=False,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Append the orchestrator guidance to the system prompt, then defer."""
        return handler(_augment(request, self.system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Append the orchestrator guidance to the system prompt, then defer."""
        return await handler(_augment(request, self.system_prompt))
