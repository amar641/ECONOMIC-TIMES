"""
IndustrialMind — end-to-end demo.

Generates the ARK-25 synthetic refinery dataset, ingests every document
through the full pipeline (extraction -> classification -> knowledge graph
/ HMNN -> RAG index), prints the resulting asset health dashboard, then
runs a few sample RAG queries.

Requires GEMINI_API_KEY in a .env file at the project root (see
.env.example) — extraction, embeddings, and the explanation layer all call
the real Gemini API.

Usage:
    python demo.py
"""

import sys

from industrialmind.brain import IndustrialBrain
from industrialmind.dataset.ark25_generator import generate, write_to_disk, as_ingest_batch

SAMPLE_QUERIES = [
    ("Why is pump P-101 flagged as high risk?", "P-101"),
    ("What is the compliance status of storage tank T-501 and why?", "T-501"),
    ("Is compressor C-102 healthy? There seemed to be conflicting reports.", "C-102"),
]


def main():
    print("=" * 70)
    print("IndustrialMind - Industrial Cognitive Architecture demo")
    print("=" * 70)

    print("\n[1/4] Generating ARK-25 synthetic dataset...")
    docs = generate()
    out_dir = write_to_disk(docs)
    print(f"    {len(docs)} documents generated -> {out_dir}/")

    print("\n[2/4] Initialising Industrial Brain (Gemini extraction + RAG)...")
    try:
        brain = IndustrialBrain()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[3/4] Ingesting documents through the pipeline "
          "(extraction -> classification -> knowledge graph / HMNN)...")
    batch = as_ingest_batch(docs)
    results = brain.ingest_batch(batch)

    static_n = sum(1 for r in results if r.doc_class == "static")
    dynamic_n = sum(1 for r in results if r.doc_class == "dynamic")
    total_obs = sum(r.observations_extracted for r in results)
    print(f"    {len(results)} documents ingested "
          f"({static_n} static -> knowledge graph, {dynamic_n} dynamic -> HMNN)")
    print(f"    {total_obs} observations extracted")

    warned = [r for r in results if r.warnings]
    if warned:
        print(f"    {len(warned)} document(s) produced warnings, e.g.:")
        for r in warned[:3]:
            print(f"      - {r.filename}: {r.warnings}")

    print("\n[4/4] Asset health dashboard (from HMNN Asset Memory):")
    print("-" * 70)
    for asset in brain.asset_dashboard():
        print(
            f"  {asset['asset_id']:8s} {asset['asset_name']:38s} "
            f"status={asset['status']:9s} risk={asset['risk_label']:9s} "
            f"health={asset['health_score']:.2f} confidence={asset['confidence']:.2f} "
            f"n_obs={asset['n_observations']}"
        )
    print("-" * 70)

    print("\nSample RAG queries (Gemini explanation layer):")
    for query, asset_id in SAMPLE_QUERIES:
        print("\n" + "=" * 70)
        print(f"Q [{asset_id}]: {query}")
        print("-" * 70)
        answer = brain.ask(query, asset_id=asset_id)
        print(answer["answer"])
        print(f"\n  (belief-state confidence: {answer['belief_state_confidence']}, "
              f"{len(answer['citations'])} citations)")

    print("\n" + "=" * 70)
    print("Pipeline stats:", brain.stats())
    print("=" * 70)


if __name__ == "__main__":
    main()
