# IndustrialMind — Local Ollama Pipeline

IndustrialMind processes industrial documents through local LLM extraction,
deterministic classification, an HMNN evidence engine, a knowledge graph, and
local RAG. No Gemini API key, cloud embedding API, or usage quota is required.

`qwen2.5:7b` running in Ollama performs document extraction and answer
generation. Evidence retrieval uses deterministic local token-overlap ranking.

## Prerequisites

- Python 3.10+
- An Ollama Docker container exposed on `http://localhost:11434`
- The `qwen2.5:7b` model available in that container

Set `.env` from `.env.example` if your endpoint/model differs:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=180
```

To verify the container independently, open `http://localhost:11434/api/tags`.
The application performs the same check at startup and reports a clear error if
the server or model is unavailable.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the terminal pipeline

```powershell
python demo.py
```

The terminal prints every major flow event: document, extraction count,
classification/routing decision, local-RAG indexing, HMNN dashboard, and query.

## Run the live browser visualization

In terminal 1:

```powershell
python event_bus.py
```

Open `index.html` in a browser, then run in terminal 2:

```powershell
python demo_live.py
```

`demo_live.py` prints every emitted event and payload with a `[FLOW]` prefix in
addition to publishing it to the browser. The UI connects to
`ws://localhost:8000/ws` and replays events if opened late.

## Architecture

```text
Documents → Ollama extraction → deterministic classification
          → static: Knowledge Graph + local RAG index
          → dynamic: HMNN Evidence Engine + local RAG index
          → local qwen2.5:7b explanation with source citations
```
