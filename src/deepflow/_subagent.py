"""Small helpers for invoking sub-agents and reading their results.

These are intentionally self-contained (no `deepagents` private imports) so the
package stays decoupled from `deepagents` internals — a sub-agent is just any
LangGraph runnable whose state has a ``messages`` key.
"""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

EXCLUDED_STATE_KEYS = frozenset({"messages", "todos", "tasks", "structured_response"})
"""Keys not forwarded into a workflow sub-agent (and not merged back).

``messages`` is replaced with the step's prompt; ``todos`` (planning),
``tasks`` (the task-list store) and ``structured_response`` are agent-local and
must not leak across steps.
"""


def prepare_state(
    state: Mapping[str, Any],
    prompt: str,
    *,
    private_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build a fresh sub-agent input state from the parent state.

    Strips excluded + private keys, then seeds a single ``HumanMessage``
    carrying the rendered step prompt.
    """
    out = {k: v for k, v in state.items() if k not in EXCLUDED_STATE_KEYS and k not in private_keys}
    out["messages"] = [HumanMessage(content=prompt)]
    return out


def extract_text(result: Mapping[str, Any] | None) -> str:
    """Return a sub-agent's textual result from its final state.

    Prefers a JSON-serialized ``structured_response``; otherwise the last
    non-empty ``AIMessage`` text.
    """
    if not result:
        return ""
    structured = result.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            return structured.model_dump_json()
        if dataclasses.is_dataclass(structured) and not isinstance(structured, type):
            return json.dumps(dataclasses.asdict(structured))
        return json.dumps(structured)
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            text = msg.text.rstrip() if msg.text else ""
            if text:
                return text
    return ""


def state_delta(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return sub-agent state updates safe to merge into the parent state."""
    return {k: v for k, v in (result or {}).items() if k not in EXCLUDED_STATE_KEYS}


def merge_delta(acc: dict[str, Any], new: Mapping[str, Any]) -> None:
    """Merge a sub-agent's state delta into ``acc`` in place.

    Dict-valued keys (e.g. the filesystem map) are shallow-merged so writes to
    distinct files from parallel steps all survive; the later write wins per
    key. Other values are replaced wholesale.
    """
    for key, value in new.items():
        existing = acc.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            acc[key] = {**existing, **value}
        else:
            acc[key] = value
