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


# --------------------------------------------------------------------------- #
# Task-list mode (scalable to-do dispatch). Same NAMESPACE; distinct event
# names, so one reader can render both a workflow and a task-list run.
# --------------------------------------------------------------------------- #
TASKLIST_PLAN = "tasklist_plan"
"""Emitted once before dispatch. Fields: ``total``, ``pending``, ``batch_size``,
``worker_count`` — the orchestrator's plan, derived from counts (never contents)."""

BATCH_START = "batch_start"
"""A worker has been handed its slice. Fields: ``worker``, ``size``, ``todos``
(``[{id, content}]``) — the ENTIRE visible context of that worker, i.e. exactly
what its ``read_todos`` can return. No worker ever sees more than this."""

WORKER_READ = "worker_read"
"""A running worker called ``read_todos``. Fields: ``worker``, ``returned``,
``ids`` — proof it sees only its own slice, not the global store."""

BATCH_DONE = "batch_done"
"""A worker finished its slice. Fields: ``worker``, ``results``
(``[{id, status, result}]``) — the status it wrote for each of its to-dos."""

TASKLIST_DONE = "tasklist_done"
"""Dispatch finished. Fields: ``done``, ``failed``, ``pending``, ``in_progress``
— the final status rollup the orchestrator receives (its whole footprint)."""

CHECK_FAILED = "check_failed"
"""A completed to-do's deterministic acceptance ``check`` failed, so the engine
flipped it ``done`` → ``failed``. Fields: ``worker``, ``id``, ``output`` — proof
that "done" means *verified* done, not self-attested."""

VERIFY_PLAN = "verify_plan"
"""A sampled verification pass began. Fields: ``done``, ``sampled``,
``batch_size``, ``worker_count`` — how much of the completed work is being
independently re-checked (sublinear QA, not a full pass)."""

VERIFY_DONE = "verify_done"
"""Verification finished. Fields: ``sampled``, ``reverted`` — how many sampled
to-dos were re-checked and how many were flipped back to ``failed`` for retry."""
