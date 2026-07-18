# IndustrialMind — Live Pipeline Demo

Real-time browser visualization driven by your **actual** pipeline run — not
a scripted animation. When `demo_live.py` calls the real `IndustrialBrain`,
the real `hmnn_engine`, the real Gemini API, the browser lights up in
lockstep with whatever is actually happening, including real timing.

## How it works

```
demo_live.py  →  event_bus.py  →  index.html (browser)
(your real       (WebSocket        (renders the flow,
 pipeline)        bridge)           HMNN math, dashboard)
```

- `demo_live.py` calls the **exact same methods** `demo.py` already calls
  (`IndustrialBrain.ingest_document`'s internals, `.ask()`,
  `.asset_dashboard()`, `.asset_detail()`). Nothing in your `industrialmind/`
  package is modified.
- After each real step, it calls `emit(event_type, payload)` from
  `pipeline_events.py`, which POSTs to the event bus.
- `event_bus.py` is a tiny FastAPI server that broadcasts every event to any
  connected browser tab over a WebSocket, and also keeps a replay log — so
  you can open the browser before OR after starting the pipeline run.
- `index.html` is a single self-contained file (no build step, no npm) that
  connects to the bus and renders two views: **Worker** (a phone mockup
  showing a field technician asking a plain-language question and getting a
  grounded answer) and **Engine** (the fleet dashboard, live φ/µ/η/scale
  readouts, the document flow log, and per-asset evidence timelines).

## File placement

Drop these four files into your **project root**, next to your existing
`demo.py`:

```
your-repo/
├── industrialmind/          <- unchanged, untouched
├── demo.py                  <- unchanged, your original
├── demo_live.py             <- NEW
├── event_bus.py             <- NEW
├── pipeline_events.py       <- NEW
├── index.html                <- NEW
├── requirements.txt
└── .env
```

## One-time setup

```bash
pip install fastapi uvicorn websockets requests
```

(Your existing `requirements.txt` already covers numpy/dotenv/google-genai/
networkx — this just adds the four demo-only dependencies. Feel free to
`pip freeze` these into a separate `requirements-demo.txt` so you don't
touch your submitted `requirements.txt`.)

## Running it (three steps, three terminals)

**Terminal 1 — start the event bus** (leave running):
```bash
python event_bus.py
```
You should see it start on `http://localhost:8000`.

**Browser — open the visualization**:
```
Open index.html directly in Chrome/Firefox (double-click it, or
File > Open). No local server needed for the HTML file itself — it's
just a static file that connects OUT to ws://localhost:8000/ws.
```
It will say "connecting…" then "connected · localhost:8000" in the header.
At this point it's just waiting — the rail is grayed out, the dashboard is
empty. This is correct; it's waiting for `demo_live.py`.

**Terminal 2 — run the real pipeline**:
```bash
python demo_live.py
```
This requires `GEMINI_API_KEY` exactly like `demo.py` does. As it runs, the
browser updates live: dataset generation, brain init, each document's
extraction → classification → HMNN/knowledge-graph routing → RAG indexing,
the fleet dashboard populating, the worker's opening question getting a
real grounded answer, and the remaining sample queries.

## Recording

1. Do a full dry run first (steps above) to confirm your Gemini key works
   and the run completes without hitting a rate limit — this is your
   biggest risk, per the earlier plan. Fix any issues before recording.
2. When ready to record: **restart the event bus** (`Ctrl+C`, rerun
   `python event_bus.py`) so the replay log is empty, **reload the browser
   tab**, start your screen recorder, *then* run `python demo_live.py`.
3. The real run takes as long as your real API calls take — 27 documents
   through Gemini extraction + classification + RAG indexing will likely
   run several minutes. You have two options:
   - Record the whole thing and cut it down in editing (recommended —
     gives you real footage to choose the best moments from).
   - Trim `SAMPLE_QUERIES` / the document set in `ark25_generator.py`
     temporarily for a shorter live take, if you want a tighter one-shot
     recording. Don't do this to your submitted dataset, only to a demo
     copy.
4. The **Worker view** (default tab) is your opening shot — Rajesh's
   question and the grounded answer appear there automatically as soon as
   `worker_query_done` fires, roughly 80% through the run. You can switch
   to **Engine view** any time (top-right toggle) to show the dashboard,
   the live φ/µ/η numbers changing during `hmnn_updated` events, and the
   document flow log filling in — this is the strongest part to linger on
   for the "under the hood" section of your video.
5. Click an asset card in Engine view any time after its `asset_detail`
   event has arrived to pull up its evidence timeline (polarity bars per
   observation) — do this for P-101 to show the degrade-then-recover arc,
   and consider doing it again for C-102 to show the contradictory
   operator-log entries sitting against the flagged vibration reading.

## If something doesn't show up

- **Browser stuck on "connecting…"**: `event_bus.py` isn't running, or a
  firewall/browser extension is blocking `localhost:8000`. Check
  `curl http://localhost:8000/health` returns `{"status":"ok",...}`.
- **Bus running, browser connected, but nothing animates when you run
  `demo_live.py`**: check the terminal running `demo_live.py` for a
  `[pipeline_events] WARNING: event bus not reachable` message — if you see
  it, the bus was not up yet when `demo_live.py` started; just rerun
  `demo_live.py` (the bus check happens once, lazily, on the first `emit()`
  call).
- **`demo_live.py` fails immediately with a `RuntimeError` about
  `GEMINI_API_KEY`**: same as `demo.py` — your `.env` isn't set up. This is
  unrelated to the visualization layer.
- **Some documents show 0 observations**: that's real pipeline behavior
  being faithfully reported, not a bug in the demo layer — same as it would
  behave in `demo.py`.

## What's real vs. what's rendering polish

Real, sourced directly from your engine's return values, with no
fabrication or estimation anywhere in this layer:
- Every φ, µ, η, scale(t) value shown (straight from
  `PlantMemory.ingest_observations()`'s `assets_updated` summaries).
- Every health score, status label, confidence value.
- Every RAG answer and its citations (straight from `RAGEngine.ask()`).
- Every document's classification (source_type, doc_class, authority,
  matched_rule) — straight from `classify_document()`.
- All timing (`duration_ms` on each stage) — measured with `time.time()`
  around the real calls, not simulated.

Rendering-only (not claims about your system, just presentation choices):
- The phone mockup chrome, the "thinking…" dots while waiting for a real
  `brain.ask()` call to return, the control-room color palette, card
  flash-on-update animation.
