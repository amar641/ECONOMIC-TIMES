"""
Static Knowledge Graph
========================
Holds reference knowledge (OEM manuals, SOPs, regulations, vendor docs) as
a graph linking assets, asset types, documents, and the individual
reference observations extracted from them.

Static documents never update HMNN's Asset Memory (see hmnn_engine.py —
"Static Knowledge ... never directly update the plant's belief"). Instead
they populate this graph, which RAG queries for context alongside dynamic
evidence.

Node types: asset, asset_type, document, reference
Edge types: HAS_TYPE (asset->asset_type), DESCRIBED_IN (asset->document),
            CONTAINS (document->reference), ABOUT (reference->asset)
"""

from typing import Dict, List, Optional

import networkx as nx

from industrialmind.hmnn_engine import Observation


class KnowledgeGraph:
    """In-memory static knowledge graph, networkx-backed."""

    def __init__(self):
        self._g = nx.MultiDiGraph()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add_static_observations(self, observations: List[Observation]) -> Dict:
        """
        Fold a batch of static Observations (doc_class == "static") into the
        graph. Each observation becomes a `reference` node linked to its
        asset and source document.
        """
        assets_touched = set()
        for obs in observations:
            asset_node = f"asset:{obs.asset_id}"
            type_node = f"type:{obs.asset_type}"
            doc_node = f"doc:{obs.document_id}"
            ref_node = f"ref:{obs.observation_id}"

            self._g.add_node(asset_node, kind="asset", asset_id=obs.asset_id, name=obs.asset_name)
            self._g.add_node(type_node, kind="asset_type", name=obs.asset_type)
            self._g.add_node(doc_node, kind="document", source_type=obs.source_type,
                              authority=obs.authority)
            self._g.add_node(
                ref_node, kind="reference",
                attribute=obs.attribute, claim=obs.claim,
                raw_excerpt=obs.raw_excerpt, source_type=obs.source_type,
                authority=obs.authority, document_id=obs.document_id,
                asset_id=obs.asset_id,
            )

            self._g.add_edge(asset_node, type_node, relation="HAS_TYPE")
            self._g.add_edge(asset_node, doc_node, relation="DESCRIBED_IN")
            self._g.add_edge(doc_node, ref_node, relation="CONTAINS")
            self._g.add_edge(ref_node, asset_node, relation="ABOUT")

            assets_touched.add(obs.asset_id)

        return {"status": "ok", "assets_touched": sorted(assets_touched), "references_added": len(observations)}

    # ── Query ────────────────────────────────────────────────────────────────

    def references_for_asset(self, asset_id: str, source_type: Optional[str] = None) -> List[Dict]:
        """All static reference facts linked to an asset (optionally filtered by source_type)."""
        asset_node = f"asset:{asset_id}"
        if asset_node not in self._g:
            return []

        results = []
        # asset -> DESCRIBED_IN -> doc -> CONTAINS -> ref
        for _, doc_node, edata in self._g.out_edges(asset_node, data=True):
            if edata.get("relation") != "DESCRIBED_IN":
                continue
            for _, ref_node, redata in self._g.out_edges(doc_node, data=True):
                if redata.get("relation") != "CONTAINS":
                    continue
                ref_data = self._g.nodes[ref_node]
                if source_type and ref_data.get("source_type") != source_type:
                    continue
                results.append(dict(ref_data))
        return results

    def asset_type_for(self, asset_id: str) -> Optional[str]:
        asset_node = f"asset:{asset_id}"
        for _, type_node, edata in self._g.out_edges(asset_node, data=True):
            if edata.get("relation") == "HAS_TYPE":
                return self._g.nodes[type_node].get("name")
        return None

    def search_references(self, query_terms: List[str], asset_id: Optional[str] = None,
                           top_k: int = 10) -> List[Dict]:
        """
        Cheap keyword search over reference claims/excerpts — used as a
        fallback / complement to the RAG vector store for exact term hits
        (e.g. a regulation code like "OISD-181").
        """
        terms = [t.lower() for t in query_terms if t]
        scored = []
        for node, data in self._g.nodes(data=True):
            if data.get("kind") != "reference":
                continue
            if asset_id and data.get("asset_id") != asset_id:
                continue
            haystack = f"{data.get('claim', '')} {data.get('raw_excerpt', '')}".lower()
            score = sum(1 for t in terms if t in haystack)
            if score > 0:
                scored.append((score, data))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def stats(self) -> Dict:
        kinds = {}
        for _, data in self._g.nodes(data=True):
            kinds[data.get("kind", "unknown")] = kinds.get(data.get("kind", "unknown"), 0) + 1
        return {"nodes": self._g.number_of_nodes(), "edges": self._g.number_of_edges(), "by_kind": kinds}


# Singleton — shared across the application, mirrors hmnn_engine.plant_memory
knowledge_graph = KnowledgeGraph()
