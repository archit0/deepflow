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

    middleware = WorkflowMiddleware(subagents=compiled, max_concurrency=max_concurrency, max_steps=max_steps)
    return create_deep_agent(
        model=model,
        tools=shared_tools or None,
        backend=backend,
        system_prompt=system_prompt,
        middleware=[middleware],
        **kwargs,
    )
