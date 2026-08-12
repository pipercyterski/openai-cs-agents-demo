"""Minimal Odyssey proxy client.

The Dystopic platform dispatches a run by calling ``run(task_input, *, proxy_url,
run_token)`` (see ``main.py``).  Every tool the ported agent calls is routed
*out* to the Odyssey proxy at ``{proxy_url}/tools/{tool_name}`` so that the
simulated world — not the customer's real backend — answers it.  We keep this
client dependency-free (httpx only, already a transitive dep of the Agents SDK)
so the sandbox does not need the full ``dystopic`` package installed.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import httpx

# Terminal proxy statuses that must never be retried (mirrors the SDK contract).
_TERMINAL = {400, 401, 404, 409, 413, 422, 500, 502}
_RETRYABLE = {429, 503}


class ProxyError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def proxy_call(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    proxy_url: str,
    run_token: str,
    timeout: float = 120.0,
    max_attempts: int = 4,
) -> Any:
    """POST tool arguments to the world and return the unwrapped ``response``.

    Retries 429/503 with bounded exponential backoff; raises ``ProxyError`` on
    terminal statuses or transport failure so the caller can degrade gracefully.
    """
    url = f"{proxy_url.rstrip('/')}/tools/{tool_name}"
    headers = {"Authorization": f"Bearer {run_token}", "Content-Type": "application/json"}
    delay = 0.5
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(url, json=arguments or {}, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:  # transport failure: terminal
            raise ProxyError(f"transport error calling {tool_name}: {exc}") from exc
        if resp.status_code == 200:
            payload = resp.json()
            # The proxy always 200s for application-level outcomes; the tool
            # result lives under "response".
            return payload.get("response", payload)
        if resp.status_code in _RETRYABLE and attempt < max_attempts:
            time.sleep(min(delay, 4.0))
            delay *= 2
            continue
        # terminal (or exhausted retries)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise ProxyError(
            f"{tool_name} failed with {resp.status_code}",
            status_code=resp.status_code,
            body=body,
        )
    raise ProxyError(f"{tool_name} exhausted retries", body=str(last_exc))
