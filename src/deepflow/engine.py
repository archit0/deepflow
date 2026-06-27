"""Streaming-first execution engine for declarative workflows.

Design goals:

- **Real-time events.** Every progress event (``plan`` → ``phase_start`` →
  ``step_start`` → ``step_event`` → ``step_done`` → ``phase_done`` →
  ``workflow_done``) is emitted the moment it happens. Crucially, ``step_done``
  fires when *that* step settles — not batched at the end of the phase — so a
  UI streaming the run sees parallel steps complete independently.
- **Live sub-agent activity.** On the async path each step streams its
  sub-agent and forwards new messages as ``step_event``s, so you can show what a
  step is doing while it runs. (The sync path emits ``step_start`` / ``step_done``
  from the tool's own thread — where the stream writer is reachable — and skips
  the per-message ``step_event`` firehose.)
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
# Sync
# --------------------------------------------------------------------------- #
def _run_step_core(
    runnable: Runnable,
    step: WorkflowStep,
    base_state: Mapping[str, Any],
    results: dict[str, str],
    private_keys: frozenset[str],
) -> tuple[str, dict[str, Any]]:
    """Run one step in a worker thread without emitting.

    A LangGraph stream writer only reaches the stream from the tool's own
    thread, so the sync executor emits ``step_start`` / ``step_done`` from the
    main thread instead (``step_done`` still fires as each step settles). The
    richer ``step_event`` stream is available on the async path.
    """
    state = prepare_state(base_state, render_prompt(step.prompt, results), private_keys=private_keys)
    result = runnable.invoke(state, _SUBAGENT_CONFIG)
    return extract_text(result), state_delta(result)


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
        for step in phase.steps:
            emit(events.STEP_START, id=step.id, subagent=step.subagent_type)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # copy_context() is evaluated HERE (the tool's thread), where the
            # parent run's callbacks/tracing live, then handed to the worker via
            # ctx.run — so the sub-agent inherits them. (Copying inside the worker
            # would capture the worker's empty context and lose them.) step_done
            # is emitted from this thread as each settles for real-time streaming.
            futures = {}
            for step in phase.steps:
                ctx = contextvars.copy_context()  # captured in THIS thread (has the callbacks)
                future = pool.submit(ctx.run, _run_step_core, runnables[step.subagent_type], step, working, results, private_keys)
                futures[future] = step.id
            for future in as_completed(futures):
                step_id = futures[future]
                try:
                    text, step_delta = future.result()
                except Exception as exc:
                    logger.exception("deepflow step '%s' failed", step_id)
                    results[step_id] = f"[step '{step_id}' failed: {exc}]"
                    emit(events.STEP_ERROR, id=step_id, error=str(exc))
                    continue
                results[step_id] = text
                merge_delta(delta, step_delta)
                emit(events.STEP_DONE, id=step_id)
        merge_delta(working, delta)
        emit(events.PHASE_DONE, index=index, title=phase.title)
    emit(events.WORKFLOW_DONE)
    return _result_command(spec, results, delta, runtime.tool_call_id)
