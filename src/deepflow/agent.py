"""`create_workflow_agent` — the batteries-included entry point.

Builds a Deep Agent orchestrator with a streaming `workflow` tool, plus the
worker sub-agents the workflow delegates to. Each worker is a full Deep Agent
(filesystem, shell, tools), so workflow steps can read/write files and run
commands just like the orchestrator.
"""

from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict

from deepagents import create_deep_agent

from deepflow.middleware import CompiledSubAgent, WorkflowMiddleware
from deepflow.spec import DEFAULT_MAX_STEPS
from deepflow.tasklist import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    TaskListMiddleware,
    orchestrator_middleware,
)
from deepflow.tasklist import (
    WORKER_PROMPT as TASKLIST_WORKER_PROMPT,
)

DEFAULT_WORKER_PROMPT = (
    "You are a focused worker sub-agent inside a workflow. Complete the single task you are "
    "given, using your tools. The workflow only sees your final message — make it the complete result."
)
_GENERAL_PURPOSE_DESCRIPTION = "General-purpose worker with the full tool set; handles any focused step."


class WorkflowSubAgent(TypedDict):
    """Declarative spec for a workflow worker (compiled into a Deep Agent)."""

    name: str
    description: str
    system_prompt: NotRequired[str]
    model: NotRequired[Any]
    tools: NotRequired[Sequence[Any]]


def create_workflow_agent(
    model: Any,
    *,
    subagents: Sequence[WorkflowSubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[Any] | None = None,
    backend: Any = None,
    system_prompt: str | None = None,
    workflow_model: Any = None,
    max_concurrency: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    enable_todos: bool = False,
    todo_batch_size: int = DEFAULT_BATCH_SIZE,
    todo_max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs: Any,
) -> Any:
    """Create a Deep Agent with a streaming, declarative `workflow` tool.

    Args:
        model: Orchestrator model (a ``provider:model`` string or a chat model).
        subagents: Workflow workers. Either declarative specs
            (``{name, description, system_prompt?, model?, tools?}``) that are
            compiled into Deep Agents, or pre-compiled
            ``{name, description, runnable}``. A ``general-purpose`` worker is
            added automatically if absent.
        tools: Tools shared by the orchestrator and the default workers.
        backend: Backend shared by the orchestrator and workers (use a
            sandbox/shell backend to give workflow steps a real filesystem + shell).
        system_prompt: Extra system prompt for the orchestrator.
        workflow_model: Model for the workers (defaults to ``model``) — handy
            for running workers on a cheaper/faster model.
        max_concurrency: Max steps running at once within a phase.
        max_steps: Max total steps in a single workflow.
        enable_todos: Also mount **task-list mode** — add ``count_todos`` /
            ``add_todos`` / ``process_todos`` so this agent can dispatch a large
            to-do store in batches as well as run workflows.
        todo_batch_size: Default to-dos per worker when ``enable_todos`` is set.
        todo_max_workers: Default to-do workers running at once when ``enable_todos`` is set.
        **kwargs: Forwarded to ``deepagents.create_deep_agent`` for the orchestrator.

    Returns:
        A compiled agent. Stream it with
        ``agent.stream(..., stream_mode=["updates", "custom"])`` to see live
        workflow events (see :mod:`deepflow.events`).
    """
    worker_model = workflow_model if workflow_model is not None else model
    shared_tools: list[Any] = list(tools or [])

    compiled: list[CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            compiled.append({"name": spec["name"], "description": spec["description"], "runnable": spec["runnable"]})
            continue
        worker = create_deep_agent(
            model=spec.get("model", worker_model),
            tools=list(spec.get("tools", shared_tools)),
            backend=backend,
            system_prompt=spec.get("system_prompt", DEFAULT_WORKER_PROMPT),
            name=spec["name"],
        )
        compiled.append({"name": spec["name"], "description": spec["description"], "runnable": worker})

    if not any(c["name"] == "general-purpose" for c in compiled):
        general = create_deep_agent(
            model=worker_model,
            tools=shared_tools,
            backend=backend,
            system_prompt=DEFAULT_WORKER_PROMPT,
            name="general-purpose",
        )
        compiled.insert(0, {"name": "general-purpose", "description": _GENERAL_PURPOSE_DESCRIPTION, "runnable": general})

    middleware: list[Any] = [WorkflowMiddleware(subagents=compiled, max_concurrency=max_concurrency, max_steps=max_steps)]
    if enable_todos:
        middleware.append(
            TaskListMiddleware(
                model=worker_model,
                tools=shared_tools,
                backend=backend,
                batch_size=todo_batch_size,
                max_workers=todo_max_workers,
            )
        )
    return create_deep_agent(
        model=model,
        tools=shared_tools or None,
        backend=backend,
        system_prompt=system_prompt,
        middleware=middleware,
        **kwargs,
    )


def create_tasklist_agent(
    model: Any,
    *,
    tools: Sequence[Any] | None = None,
    backend: Any = None,
    system_prompt: str | None = None,
    worker_model: Any = None,
    worker_system_prompt: str = TASKLIST_WORKER_PROMPT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs: Any,
) -> Any:
    """Create an agent with **task-list mode** — scalable dispatch over many to-dos.

    The agent gets `count_todos` / `add_todos` / `process_todos`. Seed the store
    with ``tasks`` at invoke time (``deepflow.make_todos([...])``), or let the
    agent populate it via ``add_todos``. ``process_todos`` fans the pending to-dos
    out to worker sub-agents in disjoint batches; each worker sees only its slice.

    Both the orchestrator and the workers are Deep Agents (filesystem + ``execute``
    + compaction) **without** ``task``/``workflow`` — workers drain their slice and
    nothing more.

    Args:
        model: Orchestrator model (a ``provider:model`` string or a chat model).
        tools: Tools shared by the orchestrator and the workers (e.g. the domain
            tools each to-do needs).
        backend: Backend shared by orchestrator and workers (use a sandbox/shell
            backend to give workers a real filesystem + shell).
        system_prompt: Extra system prompt for the orchestrator.
        worker_model: Model for the workers (defaults to ``model``) — handy for
            running the many workers on a cheaper/faster model.
        worker_system_prompt: System prompt for the workers.
        batch_size: Default to-dos handed to each worker.
        max_workers: Default workers running at once.
        **kwargs: Forwarded to ``langchain.agents.create_agent`` for the orchestrator.

    Returns:
        A compiled agent. Stream it with
        ``agent.stream(..., stream_mode=["updates", "custom"])`` to see live
        task-list events (see :mod:`deepflow.events`).
    """
    from langchain.agents import create_agent  # noqa: PLC0415

    shared_tools = list(tools or [])
    middleware = orchestrator_middleware(
        model,
        worker_model=worker_model,
        tools=shared_tools,
        backend=backend,
        batch_size=batch_size,
        max_workers=max_workers,
        worker_system_prompt=worker_system_prompt,
    )
    return create_agent(model, tools=shared_tools, system_prompt=system_prompt, middleware=middleware, **kwargs)
