"""Dystopic code-mode entrypoint for the airline customer-service agent.

The platform imports this file and calls ``run(task_input, *, proxy_url,
run_token)`` inside an E2B sandbox whose kernel thread already has a running
asyncio loop — so the OpenAI Agents SDK ``Runner.run`` must execute on a fresh
thread (``run_async_in_thread``), with the Dystopic envelope re-bound *inside*
that thread (ContextVars are thread-local).

Multi-agent instrumentation wired here:

* ``_DemoRunHooks`` posts a native ``handoff`` trace edge on every control
  transfer (the graded lane the reconstructed graph unions), AND emits a
  ``handoff_traversal`` telemetry span beside it (the display-only rationale
  lane) — deliberately exercising both multi-agent channels in one run.
* A ``note`` span marks dispatch start; a ``state_snapshot`` span records the
  port's final itinerary state.
* Per-call actor attribution is done by the tools themselves
  (``dystopic_function_tool`` in ``airline.py``) — the hooks only keep the
  ambient label fresh and emit edges.
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

from agents import InputGuardrailTripwireTriggered, RunHooks, Runner
from agents.exceptions import MaxTurnsExceeded

from dystopic.odyssey import Envelope
from dystopic.odyssey.adapters.openai_agents import actor_label_for, bind_current_actor
from dystopic.odyssey.context import set_current
from dystopic.odyssey.telemetry import async_safe_emit
from dystopic.odyssey.traces import async_safe_post_handoff

from airline import PortState, build_root_agent, name_to_actor

REFUSAL_TEXT = "Sorry, I can only answer questions related to airline travel."
MAX_TURNS = 14


class _DemoRunHooks(RunHooks):
    """Actor rebinding + handoff edges on BOTH lanes (trace + telemetry)."""

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        bind_current_actor(agent, name_to_actor=name_to_actor)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        bind_current_actor(agent, name_to_actor=name_to_actor)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        # Rebind first so any tool call the target makes before its own
        # on_*_start fires is still attributed to the new agent.
        bind_current_actor(to_agent, name_to_actor=name_to_actor)
        frm = actor_label_for(from_agent, name_to_actor=name_to_actor)
        to = actor_label_for(to_agent, name_to_actor=name_to_actor)
        if not (frm and to):
            return
        # Native trace edge — the graded/observed lane.
        await async_safe_post_handoff(frm, to)
        # Telemetry span — the display-side rationale lane.
        await async_safe_emit(
            "handoff_traversal",
            {
                "from_sub_agent": frm,
                "to_sub_agent": to,
                "reason": (
                    f"{getattr(from_agent, 'name', frm)} transferred control to "
                    f"{getattr(to_agent, 'name', to)} (OpenAI Agents SDK handoff)"
                ),
            },
        )


def run_async_in_thread(coro_factory, envelope: Envelope):
    """Await ``coro_factory()`` on a fresh thread with its own event loop.

    The envelope is re-bound *inside* the thread target — ContextVars are
    thread-local, so a binding on the caller's thread would not reach the
    coroutine (every ``async_proxy_call`` would raise ``LookupError``).
    """
    out: Dict[str, Any] = {}

    def _target() -> None:
        try:
            with set_current(envelope):
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


_VALID_ROLES = {"assistant", "system", "tool", "user"}


def _to_wire_messages(items: list) -> list:
    """Convert an Agents SDK ``to_input_list()`` into wire-contract messages."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        kind = item.get("type")
        if role in _VALID_ROLES:
            content = item.get("content")
            if isinstance(content, list):
                # Flatten output_text parts to a plain string when possible.
                texts = [
                    p.get("text")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                content = "\n".join(t for t in texts if t) or None
            out.append({"role": role, "content": content})
        elif kind == "function_call":
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            out.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "name": item.get("name"),
                            "arguments": args,
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            output = item.get("output")
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
    return out


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Platform entrypoint. Returns ``{"final_response": str, "metadata": dict}``."""
    # The platform injects DYSTOPIC_* env vars before calling us; the entrypoint
    # args carry the same values, so seed the env for a local/direct invocation.
    os.environ.setdefault("DYSTOPIC_ODYSSEY_PROXY_URL", proxy_url)
    os.environ.setdefault("DYSTOPIC_RUN_TOKEN", run_token)
    envelope = Envelope.from_env()

    instruction = _instruction_from(task_input or {})
    state = PortState()
    agent = build_root_agent()

    async def _run():
        await async_safe_emit(
            "note",
            {
                "text": (
                    "dispatch start: "
                    + ("multi-turn transcript" if isinstance(instruction, list) else "single-shot")
                )
            },
        )
        result = await Runner.run(
            agent,
            instruction,
            context=state,
            hooks=_DemoRunHooks(),
            max_turns=MAX_TURNS,
        )
        await async_safe_emit(
            "state_snapshot",
            {
                "snapshot": {
                    "passenger_name": state.passenger_name,
                    "confirmation_number": state.confirmation_number,
                    "seat_number": state.seat_number,
                    "flight_number": state.flight_number,
                    "compensation_case_id": state.compensation_case_id,
                    "special_service_note": state.special_service_note,
                },
                "label": "final-itinerary",
            },
        )
        return result

    try:
        result = run_async_in_thread(_run, envelope)
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
            "metadata": {"outcome": "max_turns_exceeded", "confirmation_number": state.confirmation_number},
        }
    except Exception as exc:  # noqa: BLE001 — never return an empty final_response
        return {
            "final_response": "The agent hit an unexpected error and could not complete the request.",
            "metadata": {"outcome": "error", "detail": str(exc)[:400]},
        }

    final = getattr(result, "final_output", None)
    final_text = (
        str(final) if final not in (None, "") else "The agent completed without a textual response."
    )
    last_agent = getattr(getattr(result, "last_agent", None), "name", None)

    # Rich transcript for the trace (best-effort; malformed messages are dropped
    # by the platform, never fatal). ``to_input_list`` mixes role-shaped rows
    # with typed items (``function_call`` / ``function_call_output``) that carry
    # no ``role`` — convert those to the wire contract's assistant-tool_calls /
    # tool-row shapes so the rich trace shows the tool activity too.
    messages = None
    try:
        messages = _to_wire_messages(result.to_input_list()) or None
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
