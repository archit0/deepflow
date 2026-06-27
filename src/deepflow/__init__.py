"""deepflow — streaming, declarative multi-agent workflows for Deep Agents.

Give an agent one `workflow` tool and it can author a phase/step plan in a
single call: phases run sequentially, steps within a phase run in parallel
(fan-out), and later steps consume earlier outputs via ``{{step_id}}`` (fan-in).
Each step runs in its own isolated sub-agent, and the whole run streams live.

Quickstart::

    from deepflow import create_workflow_agent

    agent = create_workflow_agent(model="openai:gpt-5.5")
    for mode, chunk in agent.stream(
        {"messages": "Research X and Y, then compare them."},
        stream_mode=["updates", "custom"],
    ):
        ...  # `custom` chunks are {"deepflow": {"event": ...}}

Built on top of Deep Agents (https://github.com/langchain-ai/deepagents).
"""

from deepflow import events
from deepflow.agent import WorkflowSubAgent, create_workflow_agent
from deepflow.middleware import CompiledSubAgent, WorkflowMiddleware, build_workflow_tool
from deepflow.spec import (
    WorkflowPhase,
    WorkflowSpec,
    WorkflowStep,
    WorkflowToolArgs,
    plan_payload,
    render_prompt,
    validate_workflow,
)

__version__ = "0.1.1"

__all__ = [
    "CompiledSubAgent",
    "WorkflowMiddleware",
    "WorkflowPhase",
    "WorkflowSpec",
    "WorkflowStep",
    "WorkflowSubAgent",
    "WorkflowToolArgs",
    "__version__",
    "build_workflow_tool",
    "create_workflow_agent",
    "events",
    "plan_payload",
    "render_prompt",
    "validate_workflow",
]
