"""Streaming-first execution engine for declarative workflows.

Design goals:

- **Real-time events.** Every progress event (``plan`` → ``phase_start`` →
  ``step_start`` → ``step_event`` → ``step_done`` → ``phase_done`` →
  ``workflow_done``) is emitted the moment it happens. Crucially, ``step_done``
  fires when *that* step settles — not batched at the end of the phase — so a
  UI streaming the run sees parallel steps complete independently.
- **Live sub-agent activity.** Each step runs the sub-agent with ``astream`` /
  ``stream`` and forwards its new messages as ``step_event``s, so you can show
  what a step is doing while it runs, not just that it finished.
- **The reader matches the writer.** Results are extracted from the same final
  state the stream surfaces; the event names live in :mod:`deepflow.events`.
"""

import asyncio
import contextvars
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command

from deepflow import events
from deepflow._subagent import extract_text, merge_delta, prepare_state, state_delta
from deepflow.spec import WorkflowSpec, WorkflowStep, aggregate_output, plan_payload, render_prompt

logger = logging.getLogger(__name__)

_SUBAGENT_CONFIG: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}


def _summarize_message(msg: Any) -> dict[str, Any]:
    """Compact, serializable summary of a sub-agent message for a step_event."""
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            return {"kind": "tool_call", "tools": [c.get("name") for c in msg.tool_calls]}
        text = msg.text.strip() if msg.text else ""
        if text:
            return {"kind": "message", "preview": text[:140]}
    if isinstance(msg, ToolMessage):
        return {"kind": "tool_result", "name": getattr(msg, "name", None)}
    return {"kind": "other"}


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


def _result_command(spec: WorkflowSpec, results: dict[str, str], delta: dict[str, Any], tool_call_id: str) -> Command:
    return Command(update={**delta, "messages": [ToolMessage(aggregate_output(spec, results), tool_call_id=tool_call_id)]})


# --------------------------------------------------------------------------- #
# Async (the streaming path)
# --------------------------------------------------------------------------- #
async def _arun_step(
    runnable: Runnable,
    step: WorkflowStep,
    base_state: Mapping[str, Any],
    results: dict[str, str],
    private_keys: frozenset[str],
    emit: _Emitter,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, dict[str, Any]]:
    async with semaphore:
        emit(events.STEP_START, id=step.id, subagent=step.subagent_type)
        state = prepare_state(base_state, render_prompt(step.prompt, results), private_keys=private_keys)
        last: Mapping[str, Any] | None = None
        seen = 0
        try:
            async for chunk in runnable.astream(state, _SUBAGENT_CONFIG, stream_mode="values"):
                last = chunk if isinstance(chunk, dict) else None
                msgs = last.get("messages", []) if last else []
                for msg in msgs[seen:]:
                    emit(events.STEP_EVENT, id=step.id, **_summarize_message(msg))
                seen = len(msgs)
        except Exception as exc:
            logger.exception("deepflow step '%s' failed", step.id)
            emit(events.STEP_ERROR, id=step.id, error=str(exc))
            return step.id, f"[step '{step.id}' failed: {exc}]", {}
        emit(events.STEP_DONE, id=step.id)
        return step.id, extract_text(last), state_delta(last)


async def arun_workflow(
    spec: WorkflowSpec,
    runnables: Mapping[str, Runnable],
    runtime: Any,
    *,
    private_keys: frozenset[str] = frozenset(),
    concurrency: int = 8,
) -> Command:
    """Execute a workflow, streaming events as it runs (async)."""
    emit = _Emitter(getattr(runtime, "stream_writer", None))
    emit(events.PLAN, **plan_payload(spec))
    results: dict[str, str] = {}
    working: dict[str, Any] = dict(runtime.state)
    delta: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(concurrency)
    for index, phase in enumerate(spec.phases):
        emit(events.PHASE_START, index=index, title=phase.title)
        outcomes = await asyncio.gather(
            *(_arun_step(runnables[s.subagent_type], s, working, results, private_keys, emit, semaphore) for s in phase.steps)
        )
        for sid, text, step_delta in outcomes:
            results[sid] = text
            merge_delta(delta, step_delta)
        merge_delta(working, delta)
        emit(events.PHASE_DONE, index=index, title=phase.title)
    emit(events.WORKFLOW_DONE)
    return _result_command(spec, results, delta, runtime.tool_call_id)


# --------------------------------------------------------------------------- #
# Sync (events fire from worker threads via a copied context)
# --------------------------------------------------------------------------- #
def _run_step(
    runnable: Runnable,
    step: WorkflowStep,
    base_state: Mapping[str, Any],
    results: dict[str, str],
    private_keys: frozenset[str],
    emit: _Emitter,
) -> tuple[str, str, dict[str, Any]]:
    emit(events.STEP_START, id=step.id, subagent=step.subagent_type)
    state = prepare_state(base_state, render_prompt(step.prompt, results), private_keys=private_keys)
    last: Mapping[str, Any] | None = None
    seen = 0
    try:
        for chunk in runnable.stream(state, _SUBAGENT_CONFIG, stream_mode="values"):
            last = chunk if isinstance(chunk, dict) else None
            msgs = last.get("messages", []) if last else []
            for msg in msgs[seen:]:
                emit(events.STEP_EVENT, id=step.id, **_summarize_message(msg))
            seen = len(msgs)
    except Exception as exc:
        logger.exception("deepflow step '%s' failed", step.id)
        emit(events.STEP_ERROR, id=step.id, error=str(exc))
        return step.id, f"[step '{step.id}' failed: {exc}]", {}
    emit(events.STEP_DONE, id=step.id)
    return step.id, extract_text(last), state_delta(last)


def run_workflow(
    spec: WorkflowSpec,
    runnables: Mapping[str, Runnable],
    runtime: Any,
    *,
    private_keys: frozenset[str] = frozenset(),
    concurrency: int = 8,
) -> Command:
    """Execute a workflow, streaming events as it runs (sync)."""
    emit = _Emitter(getattr(runtime, "stream_writer", None))
    emit(events.PLAN, **plan_payload(spec))
    results: dict[str, str] = {}
    working: dict[str, Any] = dict(runtime.state)
    delta: dict[str, Any] = {}
    for index, phase in enumerate(spec.phases):
        emit(events.PHASE_START, index=index, title=phase.title)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Run each step in a copy of the current context so worker threads
            # inherit the stream writer, callbacks, and tracing; collect results
            # as each settles (events already fired in real time inside the step).
            def _go(step: WorkflowStep) -> tuple[str, str, dict[str, Any]]:
                return contextvars.copy_context().run(_run_step, runnables[step.subagent_type], step, working, results, private_keys, emit)

            futures = [pool.submit(_go, s) for s in phase.steps]
            for future in as_completed(futures):
                sid, text, step_delta = future.result()
                results[sid] = text
                merge_delta(delta, step_delta)
        merge_delta(working, delta)
        emit(events.PHASE_DONE, index=index, title=phase.title)
    emit(events.WORKFLOW_DONE)
    return _result_command(spec, results, delta, runtime.tool_call_id)
