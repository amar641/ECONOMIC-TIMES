"""
emit() — the single function demo_live.py calls to tell the browser what
the real pipeline is doing, right when it's doing it.

Deliberately fails silently (prints a warning once, then no-ops) if the
event bus isn't running — so if you forget to start event_bus.py, or the
video recorder's browser tab isn't open, demo_live.py still runs your real
pipeline correctly, it just won't animate anything. Recording never blocks
the underlying demo.
"""

import time
from typing import Any, Dict, Optional

import requests

_BUS_URL = "http://localhost:8000/emit"
_bus_available: Optional[bool] = None  # None = not checked yet


def _check_bus() -> bool:
    global _bus_available
    if _bus_available is not None:
        return _bus_available
    try:
        requests.get("http://localhost:8000/health", timeout=0.5)
        _bus_available = True
    except Exception:
        _bus_available = False
        print(
            "\n[pipeline_events] WARNING: event bus not reachable at "
            "http://localhost:8000 — is `python event_bus.py` running?\n"
            "The pipeline will continue normally, but the browser will show nothing.\n"
        )
    return _bus_available


def emit(event_type: str, payload: Dict[str, Any] = None):
    """
    Send one event to the live browser demo. Never raises — a failed emit
    should never take down the actual pipeline run.
    """
    if not _check_bus():
        return
    try:
        requests.post(
            _BUS_URL,
            json={"type": event_type, "payload": payload or {}},
            timeout=1.0,
        )
    except Exception:
        pass  # never let the demo layer break the real run


def reset_bus():
    """Clear the browser's event replay buffer before starting a fresh run."""
    try:
        requests.post("http://localhost:8000/reset", timeout=1.0)
    except Exception:
        pass


def timed(event_type: str, payload: Dict[str, Any] = None):
    """
    Context manager: emits `{event_type}_start` on enter, `{event_type}_done`
    on exit, with an added `duration_ms` in the done payload. Use this to
    show the browser how long a real step actually took.
    """
    return _TimedEmit(event_type, payload or {})


class _TimedEmit:
    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_type = event_type
        self.payload = payload
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        emit(f"{self.event_type}_start", self.payload)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.time() - self._t0) * 1000, 1)
        done_payload = dict(self.payload)
        done_payload["duration_ms"] = duration_ms
        if exc_type is not None:
            done_payload["error"] = str(exc_val)
            emit(f"{self.event_type}_error", done_payload)
        else:
            emit(f"{self.event_type}_done", done_payload)
