"""Event schema emitted by the workflow engine during a run.

Every event is written to the LangGraph custom stream as
``{"deepflow": {"event": <name>, ...}}`` so a consumer reading
``agent.stream(..., stream_mode=["updates", "custom"])`` can render the run as
it happens. Keeping these names in one place means the engine that *emits* them
and any reader that *consumes* them never drift apart.
"""

NAMESPACE = "deepflow"
"""Top-level key under which every event is emitted on the custom stream."""

PLAN = "plan"
"""Emitted once, before any step runs, carrying the full phase/step plan."""

PHASE_START = "phase_start"
"""A phase began. Fields: ``index``, ``title``."""

PHASE_DONE = "phase_done"
"""A phase finished (all its steps settled). Fields: ``index``, ``title``."""

STEP_START = "step_start"
"""A step began executing. Fields: ``id``, ``subagent``."""

STEP_EVENT = "step_event"
"""Live activity forwarded from inside a running step's sub-agent.

Fields: ``id`` plus a summary of the latest message (``kind`` is one of
``message`` / ``tool_call`` / ``tool_result`` / ``other`` with extra fields).
"""

STEP_DONE = "step_done"
"""A step finished successfully. Fields: ``id``. Fires the moment the step
settles, not at the end of the phase."""

STEP_ERROR = "step_error"
"""A step failed (its error is isolated). Fields: ``id``, ``error``."""

WORKFLOW_DONE = "workflow_done"
"""The whole workflow finished; the final result is on its way back."""
