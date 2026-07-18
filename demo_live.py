"""
IndustrialMind — LIVE demo, driven by the real pipeline.

This is demo.py's logic, unchanged in substance, with emit() calls added
around each real step so the browser visualization (index.html) animates
in lockstep with what's actually happening — not a scripted replay.

SETUP (three terminals / processes):
  1. python event_bus.py         <- starts the WebSocket bridge (port 8000)
  2. open index.html in a browser <- connects to the bus, waits for events
  3. python demo_live.py          <- runs the REAL pipeline, emits as it goes

Nothing in industrialmind/ is modified. This file only calls the same
public methods demo.py already calls (IndustrialBrain.ingest_document,
.ask, .asset_dashboard, .asset_detail) — it just also tells the browser
about it in real time.

Requires GEMINI_API_KEY exactly as demo.py does.
"""

import sys
import time

from industrialmind.brain import IndustrialBrain
from industrialmind.classification import classify_document
from industrialmind.dataset.ark25_generator import generate, write_to_disk, as_ingest_batch

from pipeline_events import emit, reset_bus, timed

# The same three sample queries from demo.py, tagged with which worker/asset
# they represent for the browser's "worker view" panel.
SAMPLE_QUERIES = [
    ("Why is pump P-101 flagged as high risk?", "P-101"),
    ("What is the compliance status of storage tank T-501 and why?", "T-501"),
    ("Is compressor C-102 healthy? There seemed to be conflicting reports.", "C-102"),
]

# Which query plays as the "worker on rounds" opening beat of the video.
# Change this if you want a different asset to open the demo.
WORKER_OPENING_QUERY = SAMPLE_QUERIES[0]


def main():
    reset_bus()
    emit("run_start", {"title": "IndustrialMind — Live Pipeline Run"})

    # ── Step 1: generate ARK-25 ──────────────────────────────────────────────
    with timed("generate_dataset"):
        docs = generate()
        out_dir = write_to_disk(docs)
    emit("dataset_ready", {
        "count": len(docs),
        "output_dir": out_dir,
        "filenames": [d.filename for d in docs],
    })

    # ── Step 2: initialise the brain (real Gemini client construction) ──────
    with timed("brain_init"):
        try:
            brain = IndustrialBrain()
        except RuntimeError as e:
            emit("fatal_error", {"message": str(e)})
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
    emit("brain_ready", {})

    # ── Step 3: ingest every document through the REAL pipeline ─────────────
    # We don't call brain.ingest_batch() as a single opaque call — we loop
    # so we can emit per-document, per-stage events using the exact same
    # calls brain.ingest_document() makes internally.
    batch = as_ingest_batch(docs)
    results = []

    for i, doc in enumerate(batch):
        filename, text, document_id = doc["filename"], doc["text"], doc["document_id"]

        emit("document_start", {
            "index": i,
            "total": len(batch),
            "filename": filename,
            "document_id": document_id,
        })

        # -- Extraction (real Gemini call, via the same extractor brain uses) --
        with timed("extraction", {"filename": filename, "document_id": document_id}):
            observations = brain._extractor.extract(text, document_id, filename)

        emit("extraction_done_detail", {
            "filename": filename,
            "document_id": document_id,
            "n_observations": len(observations),
            "observations_preview": [
                {
                    "asset_id": o.asset_id,
                    "attribute": o.attribute,
                    "claim": o.claim,
                    "polarity": round(o.polarity, 3),
                    "event_type": o.event_type,
                }
                for o in observations[:5]
            ],
        })

        # -- Classification (real, deterministic — not the LLM) --
        with timed("classification", {"filename": filename, "document_id": document_id}):
            classification = classify_document(
                filename, text,
                extractor_hint=observations[0].source_type if observations else "",
            )

        for obs in observations:
            obs.source_type = classification.source_type
            obs.authority = classification.authority
            obs.doc_class = classification.doc_class

        emit("classification_done", {
            "filename": filename,
            "document_id": document_id,
            "source_type": classification.source_type,
            "doc_class": classification.doc_class,
            "authority": round(classification.authority, 3),
            "matched_rule": classification.matched_rule,
        })

        # -- Route: static -> Knowledge Graph, dynamic -> HMNN --
        if classification.doc_class == "static":
            with timed("knowledge_graph_update", {"filename": filename}):
                engine_result = brain.knowledge_graph.add_static_observations(observations)
                brain.plant_memory.ingest_observations(observations, doc_class="static")
            emit("knowledge_graph_updated", {
                "filename": filename,
                "document_id": document_id,
                "assets_touched": engine_result.get("assets_touched", []),
                "references_added": engine_result.get("references_added", 0),
            })
        else:
            with timed("hmnn_update", {"filename": filename}):
                engine_result = brain.plant_memory.ingest_observations(observations, doc_class="dynamic")
            # engine_result["assets_updated"] contains the REAL phi/mu/eta/scale
            # values computed by hmnn_engine.py for this exact ingestion.
            emit("hmnn_updated", {
                "filename": filename,
                "document_id": document_id,
                "assets_updated": engine_result.get("assets_updated", []),
            })

        # -- RAG indexing (real embedding call) --
        if observations:
            with timed("rag_index", {"filename": filename}):
                brain._rag.index_observations(observations, doc_class=classification.doc_class)
            emit("rag_indexed", {"filename": filename, "n_items": len(observations)})

        emit("document_done", {
            "index": i,
            "total": len(batch),
            "filename": filename,
            "document_id": document_id,
            "doc_class": classification.doc_class,
            "observations_extracted": len(observations),
        })

        results.append({
            "filename": filename,
            "doc_class": classification.doc_class,
            "observations_extracted": len(observations),
        })

    # ── Step 4: full dashboard snapshot, after all documents ingested ───────
    dashboard = brain.asset_dashboard()
    emit("dashboard_snapshot", {"assets": dashboard})

    for asset in dashboard:
        detail = brain.asset_detail(asset["asset_id"])
        emit("asset_detail", {"asset_id": asset["asset_id"], "detail": detail})

    static_n = sum(1 for r in results if r["doc_class"] == "static")
    dynamic_n = sum(1 for r in results if r["doc_class"] == "dynamic")
    total_obs = sum(r["observations_extracted"] for r in results)
    emit("ingestion_summary", {
        "total_documents": len(results),
        "static_documents": static_n,
        "dynamic_documents": dynamic_n,
        "total_observations": total_obs,
    })

    # ── Step 5: the worker's opening question, answered FIRST ───────────────
    # This lets index.html show "Rajesh asks a question" -> real grounded
    # answer, using the true brain.ask() call, before diving into the
    # architecture/math explanation.
    worker_query, worker_asset = WORKER_OPENING_QUERY
    emit("worker_query_start", {"query": worker_query, "asset_id": worker_asset})
    with timed("rag_query", {"query": worker_query, "asset_id": worker_asset}):
        worker_answer = brain.ask(worker_query, asset_id=worker_asset)
    emit("worker_query_done", {
        "query": worker_query,
        "asset_id": worker_asset,
        "answer": worker_answer["answer"],
        "citations": worker_answer["citations"],
        "belief_state_confidence": worker_answer["belief_state_confidence"],
        "asset_status": worker_answer["asset_status"],
    })

    # ── Step 6: remaining sample queries, for the "under the hood" section ──
    for query, asset_id in SAMPLE_QUERIES:
        if (query, asset_id) == WORKER_OPENING_QUERY:
            continue  # already asked above
        emit("query_start", {"query": query, "asset_id": asset_id})
        with timed("rag_query", {"query": query, "asset_id": asset_id}):
            answer = brain.ask(query, asset_id=asset_id)
        emit("query_done", {
            "query": query,
            "asset_id": asset_id,
            "answer": answer["answer"],
            "citations": answer["citations"],
            "belief_state_confidence": answer["belief_state_confidence"],
            "asset_status": answer["asset_status"],
        })

    stats = brain.stats()
    emit("run_complete", {"stats": stats})
    print("\nLive run complete. Stats:", stats)


if __name__ == "__main__":
    main()
