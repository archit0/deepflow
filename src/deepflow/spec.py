"""Declarative workflow spec: phases, steps, validation, and templating.

A workflow is an ordered list of phases. Phases run sequentially; the steps
inside a phase run concurrently (fan-out); a step in a later phase consumes an
earlier step's output via ``{{step_id}}`` templating (fan-in).
"""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_MAX_STEPS = 25
"""Maximum number of steps allowed in a single workflow (runaway guard)."""

_TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)(?:\.output)?\s*\}\}")
"""Matches ``{{step_id}}`` and ``{{step_id.output}}`` placeholders."""


class WorkflowStep(BaseModel):
    """A single sub-agent invocation within a phase."""

    id: str = Field(description="Unique id for this step; later steps reference it via {{id}}.")
    subagent_type: str = Field(description="Which sub-agent runs this step (one of the available types).")
    prompt: str = Field(description="The complete task for the sub-agent. Embed {{step_id}} to consume an earlier step's output.")
    description: str | None = Field(default=None, description="Short human label shown in the plan preview before the run.")
    depends_on: list[str] = Field(
        default_factory=list, description="Ids of earlier-phase steps this step consumes; every {{id}} used must be listed here."
    )

    @field_validator("id", "subagent_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return value


class WorkflowPhase(BaseModel):
    """A group of steps that run concurrently. Phases run in declaration order."""

    title: str = Field(description="Short label for this phase (shown in progress output).")
    steps: list[WorkflowStep] = Field(description="Steps to run concurrently in this phase (at least one).")

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, value: list[WorkflowStep]) -> list[WorkflowStep]:
        if not value:
            msg = "each phase must contain at least one step"
            raise ValueError(msg)
        return value


class WorkflowSpec(BaseModel):
    """A full workflow: an ordered list of phases. Used for validation."""

    phases: list[WorkflowPhase] = Field(description="Ordered phases; run sequentially.")

    @field_validator("phases")
    @classmethod
    def _non_empty(cls, value: list[WorkflowPhase]) -> list[WorkflowPhase]:
        if not value:
            msg = "a workflow must contain at least one phase"
            raise ValueError(msg)
        return value


class WorkflowToolArgs(BaseModel):
    """Permissive argument schema advertised by the ``workflow`` tool.

    Kept loose (``list[Any]``) on purpose so a slightly-malformed plan reaches
    the engine and is answered with an actionable message, instead of an opaque
    tool-call rejection at the schema boundary.
    """

    phases: list[Any] = Field(
        description=(
            "Ordered list of phase objects, each {title, steps: [{id, subagent_type, description, prompt, depends_on}]}. "
            "Phases run sequentially; steps within a phase run in parallel."
        )
    )


def template_refs(prompt: str) -> set[str]:
    """Return the set of step ids referenced via ``{{...}}`` in ``prompt``."""
    return set(_TEMPLATE_RE.findall(prompt))


def render_prompt(prompt: str, results: dict[str, str]) -> str:
    """Substitute ``{{step_id}}`` / ``{{step_id.output}}`` with prior outputs."""
    return _TEMPLATE_RE.sub(lambda m: results.get(m.group(1), m.group(0)), prompt)


def validate_workflow(  # noqa: C901, PLR0911
    spec: WorkflowSpec,
    *,
    available: frozenset[str],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> str | None:
    """Validate a spec; return the first problem as a message, or ``None``.

    Rules: unique step ids; known ``subagent_type``; ``depends_on`` points only
    to strictly-earlier phases; every ``{{id}}`` in a prompt is declared in
    ``depends_on``; total steps within ``max_steps``.
    """
    total = sum(len(p.steps) for p in spec.phases)
    if total > max_steps:
        return f"Workflow has {total} steps, over the limit of {max_steps}. Use fewer, broader steps."

    id_phase: dict[str, int] = {}
    for idx, phase in enumerate(spec.phases):
        for step in phase.steps:
            if step.id in id_phase:
                return f"Duplicate step id '{step.id}'. Step ids must be unique across the workflow."
            id_phase[step.id] = idx

    for idx, phase in enumerate(spec.phases):
        for step in phase.steps:
            if step.subagent_type not in available:
                allowed = ", ".join(f"`{n}`" for n in sorted(available))
                return f"Step '{step.id}' uses unknown subagent '{step.subagent_type}'. Available: {allowed}."
            for dep in step.depends_on:
                if dep not in id_phase:
                    return f"Step '{step.id}' depends on unknown step '{dep}'."
                if id_phase[dep] >= idx:
                    return f"Step '{step.id}' depends on '{dep}', which is in the same or a later phase. Dependencies must be earlier."
            missing = template_refs(step.prompt) - set(step.depends_on)
            if missing:
                return f"Step '{step.id}' references {{{{{', '.join(sorted(missing))}}}}} but does not list them in `depends_on`."
    return None


def plan_payload(spec: WorkflowSpec) -> dict[str, Any]:
    """Build the ``plan`` event emitted before a workflow runs."""
    return {
        "phase_count": len(spec.phases),
        "step_count": sum(len(p.steps) for p in spec.phases),
        "phases": [
            {
                "index": i,
                "title": p.title,
                "steps": [
                    {"id": s.id, "subagent_type": s.subagent_type, "description": s.description, "depends_on": list(s.depends_on)} for s in p.steps
                ],
            }
            for i, p in enumerate(spec.phases)
        ],
    }


def aggregate_output(spec: WorkflowSpec, results: dict[str, str]) -> str:
    """Build the orchestrator-facing result from the final phase's outputs."""
    final_steps = spec.phases[-1].steps
    if len(final_steps) == 1:
        return results.get(final_steps[0].id, "")
    return "\n\n".join(f"## {s.id}\n{results.get(s.id, '')}" for s in final_steps)
