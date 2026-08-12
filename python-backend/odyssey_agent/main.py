"""Dystopic code-mode entrypoint for the airline customer-service agent.

The platform imports this file and calls ``run(task_input, *, proxy_url,
run_token)`` inside an E2B sandbox whose kernel thread already has a running
asyncio loop — so the OpenAI Agents SDK ``Runner.run`` must execute on a fresh
thread (``run_async_in_thread``).  Tools reach the simulated world through the
proxy credentials carried on :class:`~odyssey_agent.airline.PortState`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Dict

# Make sibling modules importable whether this file is the source-root entrypoint
# (uploaded snapshot: entrypoint_file="main.py") or fetched from the repo clone
# in CI (entrypoint_file="python-backend/odyssey_agent/main.py").
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import InputGuardrailTripwireTriggered, Runner
from agents.exceptions import MaxTurnsExceeded

from airline import PortState, build_root_agent

REFUSAL_TEXT = "Sorry, I can only answer questions related to airline travel."
MAX_TURNS = 14


def run_async_in_thread(coro_factory):
    """Await ``coro_factory()`` on a fresh thread with its own event loop."""
    out: Dict[str, Any] = {}

    def _target() -> None:
        try:
            out["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 — surfaced to caller
            out["error"] = exc

    worker = threading.Thread(target=_target, name="agent-runner")
    worker.start()
    worker.join()
    if "error" in out:
        raise out["error"]
    return out["result"]


def _instruction_from(task_input: dict) -> Any:
    """Extract the agent input from the scenario payload.

    Supports single-shot (``user_instruction``) and replayed multi-turn
    transcripts (``messages`` / ``input_items``).
    """
    if not isinstance(task_input, dict):
        return str(task_input)
    for key in ("messages", "input_items", "conversation"):
        val = task_input.get(key)
        if isinstance(val, list) and val:
            return val
    return task_input.get("user_instruction") or json.dumps(task_input)


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Platform entrypoint. Returns ``{"final_response": str, "metadata": dict}``."""
    instruction = _instruction_from(task_input or {})
    state = PortState(proxy_url=proxy_url, run_token=run_token)
    agent = build_root_agent()

    try:
        result = run_async_in_thread(
            lambda: Runner.run(agent, instruction, context=state, max_turns=MAX_TURNS)
        )
    except InputGuardrailTripwireTriggered:
        # A relevance/jailbreak guardrail fired — the correct behavior is refusal.
        return {
            "final_response": REFUSAL_TEXT,
            "metadata": {"outcome": "guardrail_refusal"},
        }
    except MaxTurnsExceeded:
        return {
            "final_response": (
                "I wasn't able to fully complete that within the allotted steps. "
                "Here is the current state of your request; please let me know how to proceed."
            ),
            "metadata": {"outcome": "max_turns_exceeded", "last_agent": state.confirmation_number},
        }
    except Exception as exc:  # noqa: BLE001 — never return an empty final_response
        return {
            "final_response": "The agent hit an unexpected error and could not complete the request.",
            "metadata": {"outcome": "error", "detail": str(exc)[:400]},
        }

    final = getattr(result, "final_output", None)
    final_text = str(final) if final not in (None, "") else "The agent completed without a textual response."
    last_agent = getattr(getattr(result, "last_agent", None), "name", None)

    # Rich transcript for the trace (best-effort; malformed messages are dropped
    # by the platform, never fatal).
    messages = None
    try:
        _valid = {"assistant", "system", "tool", "user"}
        messages = [m for m in result.to_input_list() if isinstance(m, dict) and m.get("role") in _valid] or None
    except Exception:
        messages = None

    return {
        "final_response": final_text,
        "messages": messages,
        "metadata": {
            "outcome": "completed",
            "last_agent": last_agent,
            "confirmation_number": state.confirmation_number,
            "seat_number": state.seat_number,
            "flight_number": state.flight_number,
            "compensation_case_id": state.compensation_case_id,
        },
    }
