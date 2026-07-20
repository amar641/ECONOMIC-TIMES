"""
IndustrialMind live event bus consumer.

Run this in one terminal, then run `python demo_live.py` in another terminal.
The consumer tails a local JSONL event file, converts each event into an
Observation, feeds it into PlantMemory, and prints dashboard updates.

This intentionally avoids external services so the live demo works without a
Gemini API key. Override the event stream path with LIVE_EVENT_BUS_FILE.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

from industrialmind.hmnn_engine import Observation, PlantMemory

EVENT_BUS_PATH = Path(os.environ.get("LIVE_EVENT_BUS_FILE", "data/live_events.jsonl"))
POLL_SECONDS = float(os.environ.get("LIVE_EVENT_BUS_POLL_SECONDS", "0.5"))


def _event_to_observation(event: Dict) -> Observation:
    """Convert one JSON event from demo_live.py into a HMNN Observation."""
    return Observation(
        observation_id=event["observation_id"],
        asset_id=event["asset_id"],
        asset_name=event.get("asset_name", event["asset_id"]),
        asset_type=event.get("asset_type", "Unknown"),
        event_type=event.get("event_type", "Sensor Data"),
        attribute=event.get("attribute", "General"),
        claim=event["claim"],
        polarity=float(event.get("polarity", 0.0)),
        confidence=float(event.get("confidence", 0.9)),
        timestamp=event.get("timestamp", ""),
        source_type=event.get("source_type", "sensor_data"),
        document_id=event.get("document_id", event["observation_id"]),
        raw_excerpt=event.get("raw_excerpt", event["claim"]),
    )


def _print_dashboard(memory: PlantMemory) -> None:
    print("\nAsset dashboard")
    print("-" * 88)
    for asset in memory.all_assets_summary():
        print(
            f"{asset['asset_id']:8s} {asset['asset_name'][:32]:32s} "
            f"status={asset['status']:9s} risk={asset['risk_label']:9s} "
            f"health={asset['health_score']:.3f} confidence={asset['confidence']:.3f} "
            f"obs={asset['n_observations']}"
        )
    print("-" * 88, flush=True)


def consume_events(path: Path = EVENT_BUS_PATH) -> None:
    """Tail the event file forever and ingest each appended JSON event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    memory = PlantMemory()
    print(f"IndustrialMind event bus listening on {path}", flush=True)
    print("Waiting for events. In another terminal run: python demo_live.py", flush=True)

    with path.open("r", encoding="utf-8") as stream:
        # Read existing lines first, then keep waiting for appended events.
        # This makes the two-terminal demo visible even if demo_live.py is run
        # before event_bus.py starts.
        while True:
            line = stream.readline()
            if not line:
                time.sleep(POLL_SECONDS)
                continue

            try:
                event = json.loads(line)
                observation = _event_to_observation(event)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"Skipping invalid event: {exc}", flush=True)
                continue

            result = memory.ingest_observations([observation], doc_class="dynamic")
            print(
                f"\nReceived {observation.observation_id}: "
                f"{observation.asset_id} | {observation.attribute} | {observation.claim}"
            )
            print(f"HMNN update: {result['assets_updated']}", flush=True)
            _print_dashboard(memory)


if __name__ == "__main__":
    consume_events()
