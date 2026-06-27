"""`WorkflowMiddleware` — exposes a streaming `workflow` tool to an agent.

This middleware is intentionally decoupled from `deepagents` internals: it takes
pre-compiled sub-agents (any LangGraph runnable with a ``messages`` state key)
and orchestrates them. Wire it into any agent that accepts middleware, e.g.
``deepagents.create_deep_agent(middleware=[WorkflowMiddleware(...)])`` — or use
:func:`deepflow.create_workflow_agent` for the batteries-included path.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from deepflow.engine import arun_workflow, run_workflow
from deepflow.spec import DEFAULT_MAX_STEPS, WorkflowSpec, WorkflowToolArgs, validate_workflow


class CompiledSubAgent(TypedDict):
    """A sub-agent the workflow can delegate steps to."""

    name: str
    description: str
    runnable: Runnable
    model: NotRequired[Any]  # ignored here; accepted so specs round-trip cleanly


def _default_concurrency() -> int:
    import os  # noqa: PLC0415

    return min(8, max(1, os.cpu_count() or 4))


WORKFLOW_TOOL_DESCRIPTION = """Run a declarative, multi-agent workflow in a single call.

A workflow is an ordered list of **phases**; each phase has **steps** that each delegate to a sub-agent.

- Phases run **sequentially** (phase N+1 starts only after phase N finishes).
- Steps **within a phase** run **concurrently** — this is how you fan out work.
- A step in a later phase can consume an earlier step's output by embedding `{{step_id}}` in its `prompt`; list every referenced id in that step's `depends_on`.
- Give every step a short `description` (a few words); it appears in a plan preview before the run.

The engine runs the whole plan autonomously and returns only the **final phase's output** — each step runs in its own isolated sub-agent, so your context stays small.

Available sub-agent types:
{available_agents}

## When to use it
- Multi-stage work where later stages depend on earlier ones (research → verify → synthesize).
- Fan-out then fan-in: many independent subtasks in parallel, then combine.

## When NOT to use it
- A single delegated task, or trivial work — just call tools / `task` directly.
- Orchestration that must branch on intermediate results — drive that yourself.

## Example
```json
{
  "phases": [
    {"title": "Research", "steps": [
      {"id": "a", "subagent_type": "general-purpose", "description": "Research A", "prompt": "Research topic A."},
      {"id": "b", "subagent_type": "general-purpose", "description": "Research B", "prompt": "Research topic B."}
    ]},
    {"title": "Synthesize", "steps": [
      {"id": "s", "subagent_type": "general-purpose", "description": "Synthesize", "depends_on": ["a", "b"],
       "prompt": "Compare and synthesize:\\n\\nA: {{a}}\\n\\nB: {{b}}"}
    ]}
  ]
}
```
"""  # noqa: E501

WORKFLOW_SYSTEM_PROMPT = """## `workflow` (multi-agent orchestration)

You have a `workflow` tool that runs a declarative, multi-stage plan of sub-agents in a single call. Reach for it when an objective decomposes into stages that build on each other, or into independent subtasks that should run in parallel and then be combined.

- Phases run sequentially; steps within a phase run in parallel (fan-out).
- A later step consumes earlier outputs with `{{step_id}}` (fan-in); list them in `depends_on`. Give each step a short `description`.

Prefer `workflow` when the orchestration is fixed in advance — it runs autonomously, keeps each step's context isolated, and returns only the final result. For one-off delegation or orchestration that must branch on intermediate results, just work directly."""  # noqa: E501


def build_workflow_tool(
    subagents: Sequence[CompiledSubAgent],
    *,
    description: str | None = None,
    private_state_keys: frozenset[str] = frozenset(),
    max_concurrency: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> BaseTool:
    """Create the `workflow` tool backed by the given compiled sub-agents."""
    runnables: dict[str, Runnable] = {s["name"]: s["runnable"] for s in subagents}
    available = frozenset(runnables)
    concurrency = max_concurrency if max_concurrency is not None else _default_concurrency()

    listing = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)
    source = WORKFLOW_TOOL_DESCRIPTION if description is None else description
    tool_description = source.replace("{available_agents}", listing)

    def _prepare(phases: Any, runtime: ToolRuntime) -> tuple[WorkflowSpec | None, str | None]:
        if not runtime.tool_call_id:
            msg = "Tool call ID is required for workflow invocation"
            raise ValueError(msg)
        try:
            spec = WorkflowSpec(phases=phases)
        except Exception as exc:  # noqa: BLE001 - surface validation errors to the model
            return None, f"Invalid workflow spec: {exc}"
        error = validate_workflow(spec, available=available, max_steps=max_steps)
        return (None, error) if error else (spec, None)

    def workflow(phases: list[Any], runtime: ToolRuntime) -> str | Command:
        try:
            spec, error = _prepare(phases, runtime)
            if spec is None:
                return error or "Invalid workflow."
            return run_workflow(spec, runnables, runtime, private_keys=private_state_keys, concurrency=concurrency)
        except Exception as exc:  # noqa: BLE001 - always hand the model a correctable message
            return f"Workflow failed: {exc}"

    async def aworkflow(phases: list[Any], runtime: ToolRuntime) -> str | Command:
        try:
            spec, error = _prepare(phases, runtime)
            if spec is None:
                return error or "Invalid workflow."
            return await arun_workflow(spec, runnables, runtime, private_keys=private_state_keys, concurrency=concurrency)
        except Exception as exc:  # noqa: BLE001 - always hand the model a correctable message
            return f"Workflow failed: {exc}"

    return StructuredTool.from_function(
        name="workflow",
        func=workflow,
        coroutine=aworkflow,
        description=tool_description,
        infer_schema=False,
        args_schema=WorkflowToolArgs,
    )


class WorkflowMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Adds a streaming, declarative `workflow` tool to an agent.

    Args:
        subagents: Compiled sub-agents the workflow can delegate steps to. Each
            is ``{name, description, runnable}`` where ``runnable`` is any
            LangGraph agent whose state has a ``messages`` key.
        system_prompt: Guidance appended to the agent's system prompt.
        description: Custom `workflow` tool description (supports the
            ``{available_agents}`` placeholder).
        max_concurrency: Max steps running at once within a phase.
        max_steps: Max total steps in a single workflow.
    """

    def __init__(
        self,
        *,
        subagents: Sequence[CompiledSubAgent],
        system_prompt: str | None = WORKFLOW_SYSTEM_PROMPT,
        description: str | None = None,
        private_state_keys: frozenset[str] | None = None,
        max_concurrency: int | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        """Build the `workflow` tool from the given sub-agents."""
        super().__init__()
        if not subagents:
            msg = "At least one sub-agent must be provided for workflow mode"
            raise ValueError(msg)
        self.system_prompt = system_prompt
        self.subagent_names = frozenset(s["name"] for s in subagents)
        self.tools = [
            build_workflow_tool(
                subagents,
                description=description,
                private_state_keys=private_state_keys or frozenset(),
                max_concurrency=max_concurrency,
                max_steps=max_steps,
            )
        ]

    def _augment(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        if self.system_prompt is None:
            return request
        current = request.system_message
        text = current.text if isinstance(current, SystemMessage) else (str(current) if current else "")
        joined = f"{text}\n\n{self.system_prompt}" if text else self.system_prompt
        return request.override(system_message=SystemMessage(content=joined))

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Append the workflow guidance to the system prompt, then defer."""
        return handler(self._augment(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Append the workflow guidance to the system prompt, then defer."""
        return await handler(self._augment(request))
