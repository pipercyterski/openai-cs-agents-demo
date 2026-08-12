"""Repo-root entrypoint shim for Dystopic CI.

When the platform runs a CI regression check it clones this repo and imports the
umbrella agent's ``entrypoint_file`` from the repo root (``/home/user/agent``).
The actual ported agent lives in ``python-backend/odyssey_agent``; this shim
lets the agent keep ``entrypoint_file="main.py"`` (the same value the uploaded
snapshot uses) while the real code stays in its package.

Local/in-app checks upload ``python-backend/odyssey_agent`` directly and never
touch this file.
"""
from __future__ import annotations

import importlib.util
import os

_PORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python-backend", "odyssey_agent", "main.py")
_spec = importlib.util.spec_from_file_location("odyssey_port_main", _PORT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # the port's main.py puts its own dir on sys.path

run = _mod.run  # the Dystopic entrypoint: run(task_input, *, proxy_url, run_token)
