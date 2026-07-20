# IndustrialMind

IndustrialMind is a Python prototype for an industrial knowledge-intelligence pipeline. It turns refinery documents into structured observations, separates static reference knowledge from dynamic operational evidence, updates asset belief states with the HMNN evidence engine, and answers asset questions through a Gemini-backed RAG explanation layer.

## Pipeline

1. **Dataset generation** creates the synthetic ARK-25 refinery corpus under `data/ark25/`.
2. **Extraction** uses Gemini to convert document text into atomic `Observation` records.
3. **Classification** deterministically assigns each document a source type and routes it as static or dynamic evidence.
4. **Static routing** stores reference observations in a NetworkX knowledge graph.
5. **Dynamic routing** updates per-asset health, risk, confidence, momentum, consensus, and entropy in the HMNN plant memory.
6. **RAG answering** retrieves indexed observations and static references, then asks Gemini to explain the already-computed belief state with citations.

## Repository layout

```text
demo.py                              # End-to-end ARK-25 demo runner
industrialmind/brain.py              # Pipeline orchestrator and public API
industrialmind/classification.py     # Deterministic source/doc-class router
industrialmind/config.py             # Environment-based Gemini configuration
industrialmind/extraction.py         # Gemini-backed observation extraction
industrialmind/hmnn_engine.py        # Pure-Python/Numpy HMNN evidence engine
industrialmind/knowledge_graph.py    # Static reference graph
industrialmind/rag.py                # In-memory vector store + Gemini explanations
industrialmind/dataset/ark25_generator.py  # Synthetic ARK-25 document generator
```

## Requirements

- Python 3.10+
- A Gemini API key available as `GEMINI_API_KEY`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=<your-key>
```

The optional model environment variables in `.env.example` can override the extraction, explanation, and embedding models.

## Running the demo

```bash
python demo.py
```

The demo generates ARK-25 documents, ingests them through the full pipeline, prints the asset health dashboard, and runs sample RAG questions. The demo calls Gemini for extraction, embeddings, and answers, so it requires a valid API key and network access.

## Live local event demo

To watch HMNN asset state update from streaming events without calling Gemini, start the consumer in one terminal:

```bash
python event_bus.py
```

Then publish synthetic events from a second terminal:

```bash
python demo_live.py
```

Both scripts use the local JSONL stream at `data/live_events.jsonl` by default. Set `LIVE_EVENT_BUS_FILE=/path/to/events.jsonl` to point both terminals at a different stream, or set `LIVE_DEMO_DELAY_SECONDS=0.2` to speed up the producer.

## Programmatic usage

```python
from industrialmind.brain import IndustrialBrain

brain = IndustrialBrain()
result = brain.ingest_document(
    filename="P-101_Inspection_Report_2026-04-18.txt",
    document_text="Inspection Report — Centrifugal Pump P-101 ...",
    document_id="DOC-001",
)

print(result.doc_class)
print(brain.asset_dashboard())
print(brain.ask("Why is P-101 high risk?", asset_id="P-101"))
```

## Local checks

```bash
python -m compileall .
```

For deeper runtime validation, run `python demo.py` after configuring `GEMINI_API_KEY`.
