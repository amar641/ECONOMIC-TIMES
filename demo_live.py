"""
IndustrialMind live event producer.

Run `python event_bus.py` in one terminal, then run this script in a second
terminal. It appends synthetic dynamic observations to the JSONL event bus so
the consumer can update HMNN asset state in real time.

This live path does not call Gemini and does not require GEMINI_API_KEY.
Override the event stream path with LIVE_EVENT_BUS_FILE.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

EVENT_BUS_PATH = Path(os.environ.get("LIVE_EVENT_BUS_FILE", "data/live_events.jsonl"))
PUBLISH_DELAY_SECONDS = float(os.environ.get("LIVE_DEMO_DELAY_SECONDS", "1.0"))


def _event(asset_id: str, asset_name: str, asset_type: str, event_type: str,
           attribute: str, claim: str, polarity: float,
           source_type: str = "sensor_data") -> Dict:
    observation_id = f"LIVE-{uuid.uuid4().hex[:8]}"
    return {
        "observation_id": observation_id,
        "asset_id": asset_id,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "event_type": event_type,
        "attribute": attribute,
        "claim": claim,
        "polarity": polarity,
        "confidence": 0.92,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_type": source_type,
        "document_id": observation_id,
        "raw_excerpt": claim,
    }


def _live_events() -> Iterable[Dict]:
    yield _event(
        "P-101",
        "Centrifugal Pump P-101",
        "Centrifugal Pump",
        "Sensor Data",
        "Seal Leakage",
        "Minor seal leakage detected at pump housing.",
        -0.35,
    )
    yield _event(
        "P-101",
        "Centrifugal Pump P-101",
        "Centrifugal Pump",
        "Inspection",
        "Seal Clearance",
        "Seal clearance has crossed the maintenance threshold.",
        -0.75,
        source_type="inspection_report",
    )
    yield _event(
        "V-204",
        "Pressure Vessel V-204",
        "Pressure Vessel",
        "Inspection",
        "Shell Condition",
        "No corrosion or pressure-retaining defects observed.",
        0.85,
        source_type="inspection_report",
    )
    yield _event(
        "P-101",
        "Centrifugal Pump P-101",
        "Centrifugal Pump",
        "Maintenance",
        "Seal Replacement",
        "Mechanical seal replaced and pump returned to service.",
        0.9,
        source_type="maintenance_log",
    )


def publish_events(path: Path = EVENT_BUS_PATH) -> None:
    """Append demo events to the local JSONL bus for event_bus.py to consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Publishing live demo events to {path}", flush=True)
    with path.open("a", encoding="utf-8") as stream:
        for event in _live_events():
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            print(
                f"Published {event['observation_id']}: "
                f"{event['asset_id']} | {event['attribute']} | {event['claim']}",
                flush=True,
            )
            time.sleep(PUBLISH_DELAY_SECONDS)
    print("Done. Leave event_bus.py running to receive future events.", flush=True)


if __name__ == "__main__":
    publish_events()
