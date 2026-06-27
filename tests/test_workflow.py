"""Unit tests for the deepflow engine, spec validation, and templating.

The engine only needs runnables that expose ``stream`` / ``astream`` yielding
state dicts with a ``messages`` key, so the sub-agents here are tiny fakes — no
real models or network.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from deepflow import events
from deepflow.engine import arun_workflow, run_workflow
from deepflow.spec import (
    DEFAULT_MAX_STEPS,
    WorkflowSpec,
    render_prompt,
    validate_workflow,
)

_AVAIL = frozenset({"general-purpose", "researcher", "synth"})


class FakeAgent:
    """Sub-agent stub: replies with a fixed string (or echoes the last prompt)."""

    def __init__(self, reply: str | None = None, *, echo: bool = False) -> None:
        self.reply = reply or ""
        self.echo = echo

    def _final(self, state: dict[str, Any]) -> dict[str, Any]:
        text = self.reply
        if self.echo:
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    text = f"ECHO:{msg.text}"
                    break
        return {"messages": [*state.get("messages", []), AIMessage(content=text)]}

    def invoke(self, state, config=None):  # noqa: ARG002
        return self._final(state)

    def stream(self, state, config=None, *, stream_mode=None):  # noqa: ARG002
        yield self._final(state)

    async def astream(self, state, config=None, *, stream_mode=None):  # noqa: ARG002
        yield self._final(state)


def _runtime(writer=None, state=None):
    return SimpleNamespace(state=state or {"messages": []}, tool_call_id="call-1", stream_writer=writer)


def _collector():
    seen: list[dict] = []

    def writer(payload: dict) -> None:
        if isinstance(payload, dict) and events.NAMESPACE in payload:
            seen.append(payload[events.NAMESPACE])

    return seen, writer


def _spec(phases):
    return WorkflowSpec(phases=phases)


# --------------------------------------------------------------------------- #
# Spec validation + templating (pure)
# --------------------------------------------------------------------------- #
def test_valid_spec():
    spec = _spec(
        [
            {"title": "a", "steps": [{"id": "x", "subagent_type": "researcher", "prompt": "go"}]},
            {"title": "b", "steps": [{"id": "y", "subagent_type": "synth", "prompt": "use {{x}}", "depends_on": ["x"]}]},
        ]
    )
    assert validate_workflow(spec, available=_AVAIL) is None


def test_forward_dependency_rejected():
    spec = _spec(
        [
            {
                "title": "p",
                "steps": [
                    {"id": "x", "subagent_type": "researcher", "prompt": "{{y}}", "depends_on": ["y"]},
                    {"id": "y", "subagent_type": "researcher", "prompt": "hi"},
                ],
            }
        ]
    )
    assert "earlier" in (validate_workflow(spec, available=_AVAIL) or "")


def test_unknown_subagent_rejected():
    spec = _spec([{"title": "p", "steps": [{"id": "x", "subagent_type": "nope", "prompt": "hi"}]}])
    assert "unknown subagent" in (validate_workflow(spec, available=_AVAIL) or "")


def test_template_ref_needs_depends_on():
    spec = _spec(
        [
            {"title": "p", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "hi"}]},
            {"title": "q", "steps": [{"id": "b", "subagent_type": "synth", "prompt": "use {{a}}"}]},
        ]
    )
    assert "depends_on" in (validate_workflow(spec, available=_AVAIL) or "")


def test_max_steps():
    steps = [{"id": f"s{i}", "subagent_type": "researcher", "prompt": "x"} for i in range(DEFAULT_MAX_STEPS + 1)]
    spec = _spec([{"title": "p", "steps": steps}])
    assert "limit" in (validate_workflow(spec, available=_AVAIL) or "")


def test_render_prompt():
    assert render_prompt("a={{a}} b={{ b.output }}", {"a": "1", "b": "2"}) == "a=1 b=2"
    assert render_prompt("x={{missing}}", {}) == "x={{missing}}"


# --------------------------------------------------------------------------- #
# Engine — sync
# --------------------------------------------------------------------------- #
def test_sync_executes_and_passes_data():
    runnables = {"researcher": FakeAgent("FINDING"), "synth": FakeAgent(echo=True)}
    spec = _spec(
        [
            {"title": "Research", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "research"}]},
            {"title": "Synth", "steps": [{"id": "s", "subagent_type": "synth", "depends_on": ["a"], "prompt": "Got: {{a}}"}]},
        ]
    )
    seen, writer = _collector()
    cmd = run_workflow(spec, runnables, _runtime(writer))
    tool_msg = cmd.update["messages"][0]
    assert tool_msg.content == "ECHO:Got: FINDING"
    kinds = [e["event"] for e in seen]
    assert kinds[0] == events.PLAN
    assert events.STEP_DONE in kinds
    assert events.WORKFLOW_DONE in kinds


def test_sync_parallel_fan_out():
    runnables = {"researcher": FakeAgent("R"), "synth": FakeAgent(echo=True)}
    spec = _spec(
        [
            {
                "title": "Research",
                "steps": [
                    {"id": "a", "subagent_type": "researcher", "prompt": "a"},
                    {"id": "b", "subagent_type": "researcher", "prompt": "b"},
                ],
            },
            {"title": "Synth", "steps": [{"id": "s", "subagent_type": "synth", "depends_on": ["a", "b"], "prompt": "{{a}}+{{b}}"}]},
        ]
    )
    cmd = run_workflow(spec, runnables, _runtime())
    assert cmd.update["messages"][0].content == "ECHO:R+R"


# --------------------------------------------------------------------------- #
# Engine — async
# --------------------------------------------------------------------------- #
def test_async_executes():
    runnables = {"researcher": FakeAgent("ASYNC")}
    spec = _spec([{"title": "P", "steps": [{"id": "only", "subagent_type": "researcher", "prompt": "x"}]}])
    seen, writer = _collector()
    cmd = asyncio.run(arun_workflow(spec, runnables, _runtime(writer)))
    assert cmd.update["messages"][0].content == "ASYNC"
    # step_done must arrive before phase_done (real-time, not batched after the phase).
    kinds = [e["event"] for e in seen]
    assert kinds.index(events.STEP_DONE) < kinds.index(events.PHASE_DONE)


def test_step_event_forwards_subagent_activity():
    # step_event (live sub-agent activity) is forwarded on the async path, which
    # runs in one event-loop thread where the stream writer is reachable.
    runnables = {"researcher": FakeAgent("HELLO")}
    spec = _spec([{"title": "P", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "x"}]}])
    seen, writer = _collector()
    asyncio.run(arun_workflow(spec, runnables, _runtime(writer)))
    step_events = [e for e in seen if e["event"] == events.STEP_EVENT and e["id"] == "a"]
    assert any(e.get("kind") == "message" for e in step_events)
