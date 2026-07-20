"""
Sends a synthetic but realistically-shaped event sequence to the bus, so
we can verify index.html renders every event type correctly BEFORE running
against the real Ollama-backed pipeline. Mirrors demo_live.py's exact
payload shapes for hmnn_updated, worker_query_done, etc.
"""
import time
import requests

BUS = "http://localhost:8000"


def emit(event_type, payload=None):
    r = requests.post(f"{BUS}/emit", json={"type": event_type, "payload": payload or {}})
    print(f"  -> {event_type}: {r.json()}")


requests.post(f"{BUS}/reset")
print("=== run_start ===")
emit("run_start", {"title": "Test Run"})
time.sleep(0.3)

print("=== dataset ===")
emit("generate_dataset_start", {})
time.sleep(0.4)
emit("dataset_ready", {"count": 27, "output_dir": "data/ark25", "filenames": ["a.txt"]})
time.sleep(0.3)

print("=== brain init ===")
emit("brain_init_start", {})
time.sleep(0.4)
emit("brain_ready", {})
time.sleep(0.3)

print("=== document 1 (static) ===")
emit("document_start", {"index": 0, "total": 27, "filename": "OEM_Manual_Centrifugal_Pump.txt", "document_id": "ARK25-001"})
time.sleep(0.3)
emit("extraction_start", {"filename": "OEM_Manual_Centrifugal_Pump.txt"})
time.sleep(0.4)
emit("extraction_done_detail", {
    "filename": "OEM_Manual_Centrifugal_Pump.txt", "document_id": "ARK25-001",
    "n_observations": 3,
    "observations_preview": [{"asset_id": "P-101", "attribute": "Seal Threshold", "claim": "0.5mm replacement threshold", "polarity": 0.0, "event_type": "Reference"}]
})
time.sleep(0.2)
emit("classification_done", {
    "filename": "OEM_Manual_Centrifugal_Pump.txt", "document_id": "ARK25-001",
    "source_type": "oem_manual", "doc_class": "static", "authority": 0.98, "matched_rule": "filename:oem"
})
time.sleep(0.3)
emit("knowledge_graph_update_start", {"filename": "OEM_Manual_Centrifugal_Pump.txt"})
time.sleep(0.3)
emit("knowledge_graph_updated", {"filename": "OEM_Manual_Centrifugal_Pump.txt", "document_id": "ARK25-001", "assets_touched": ["P-101"], "references_added": 3})
time.sleep(0.2)
emit("rag_index_start", {"filename": "OEM_Manual_Centrifugal_Pump.txt"})
time.sleep(0.3)
emit("rag_indexed", {"filename": "OEM_Manual_Centrifugal_Pump.txt", "n_items": 3})
time.sleep(0.2)
emit("document_done", {"index": 0, "total": 27, "filename": "OEM_Manual_Centrifugal_Pump.txt", "document_id": "ARK25-001", "doc_class": "static", "observations_extracted": 3})
time.sleep(0.4)

print("=== document 2 (dynamic, HMNN update) ===")
emit("document_start", {"index": 6, "total": 27, "filename": "P-101_Inspection_Report_2026-04-18.txt", "document_id": "ARK25-007"})
time.sleep(0.3)
emit("extraction_done_detail", {
    "filename": "P-101_Inspection_Report_2026-04-18.txt", "document_id": "ARK25-007",
    "n_observations": 2,
    "observations_preview": [{"asset_id": "P-101", "attribute": "Seal Condition", "claim": "Seal face clearance at 0.65mm, exceeding threshold", "polarity": -0.7, "event_type": "Inspection"}]
})
time.sleep(0.3)
emit("classification_done", {
    "filename": "P-101_Inspection_Report_2026-04-18.txt", "document_id": "ARK25-007",
    "source_type": "inspection_report", "doc_class": "dynamic", "authority": 0.92, "matched_rule": "filename:inspection"
})
time.sleep(0.2)
emit("hmnn_update_start", {"filename": "P-101_Inspection_Report_2026-04-18.txt"})
time.sleep(0.5)
emit("hmnn_updated", {
    "filename": "P-101_Inspection_Report_2026-04-18.txt", "document_id": "ARK25-007",
    "assets_updated": [{
        "asset_id": "P-101", "asset_name": "Centrifugal Pump P-101",
        "health_before": 0.62, "health_after": 0.34, "health_delta": -0.28,
        "status": "Warning", "scale": 0.71, "phi": 0.81, "mu": 0.58, "eta": 0.12, "n_obs": 5
    }]
})
time.sleep(0.3)
emit("rag_indexed", {"filename": "P-101_Inspection_Report_2026-04-18.txt", "n_items": 2})
time.sleep(0.2)
emit("document_done", {"index": 6, "total": 27, "filename": "P-101_Inspection_Report_2026-04-18.txt", "document_id": "ARK25-007", "doc_class": "dynamic", "observations_extracted": 2})
time.sleep(0.5)

print("=== dashboard snapshot ===")
emit("dashboard_snapshot", {"assets": [
    {"asset_id": "P-101", "asset_name": "Centrifugal Pump P-101", "health_score": 0.34, "status": "Warning"},
    {"asset_id": "V-204", "asset_name": "Pressure Vessel V-204", "health_score": 0.91, "status": "Healthy"},
    {"asset_id": "E-305", "asset_name": "Shell & Tube Heat Exchanger E-305", "health_score": 0.88, "status": "Healthy"},
    {"asset_id": "C-102", "asset_name": "Reciprocating Compressor C-102", "health_score": 0.76, "status": "Healthy"},
    {"asset_id": "T-501", "asset_name": "Crude Storage Tank T-501", "health_score": 0.55, "status": "Monitor"},
]})
time.sleep(0.3)

emit("asset_detail", {"asset_id": "P-101", "detail": {
    "asset_id": "P-101", "asset_name": "Centrifugal Pump P-101", "asset_type": "Centrifugal Pump",
    "status": "Warning", "confidence": 0.71, "maintenance_state": "Overdue", "compliance_state": "Unknown",
    "hmnn_state": {"phi": 0.81, "mu": 0.58, "eta": 0.12},
    "recent_observations": [
        {"event_type": "Inspection", "timestamp": "2026-01-15T00:00:00", "attribute": "Seal Condition", "claim": "Face clearance 0.2mm, healthy", "polarity": 0.8},
        {"event_type": "Inspection", "timestamp": "2026-03-12T00:00:00", "attribute": "Seal Condition", "claim": "Face clearance 0.45mm, approaching threshold", "polarity": -0.1},
        {"event_type": "Inspection", "timestamp": "2026-04-18T00:00:00", "attribute": "Seal Condition", "claim": "Face clearance 0.65mm, leakage confirmed", "polarity": -0.7},
    ]
}})
time.sleep(0.4)

print("=== worker query ===")
emit("worker_query_start", {"query": "Why is pump P-101 flagged as high risk?", "asset_id": "P-101"})
time.sleep(1.2)
emit("worker_query_done", {
    "query": "Why is pump P-101 flagged as high risk?", "asset_id": "P-101",
    "answer": "P-101's seal clearance progressed from 0.2mm (healthy) in January to 0.65mm by April 18, exceeding the OEM 0.5mm threshold [inspection_report, ARK25-007, 2026-04-18]. The SOP-recommended 30-day replacement window was missed after the March inspection flagged approaching threshold, which the RCA identifies as the root cause of the subsequent seal failure [rca_report, ARK25-013, 2026-05-10].",
    "citations": [
        {"source_type": "inspection_report", "document_id": "ARK25-007", "timestamp": "2026-04-18", "asset_id": "P-101"},
        {"source_type": "rca_report", "document_id": "ARK25-013", "timestamp": "2026-05-10", "asset_id": "P-101"},
    ],
    "belief_state_confidence": 0.71, "asset_status": "Warning",
})
time.sleep(0.5)

print("=== run complete ===")
emit("ingestion_summary", {"total_documents": 27, "static_documents": 6, "dynamic_documents": 21, "total_observations": 84})
emit("run_complete", {"stats": {"plant_memory": {"total_assets": 5}, "knowledge_graph": {"nodes": 40}}})

print("\nDone. Check the browser.")
