"""Local RAG retrieval and Ollama explanation layer.

Retrieval is deterministic token-overlap ranking; no cloud embedding API or
separate embedding model is required. Ollama is used only to explain the
already-computed HMNN belief state and retrieved evidence.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from industrialmind.config import get_config
from industrialmind.hmnn_engine import PlantMemory
from industrialmind.knowledge_graph import KnowledgeGraph
from industrialmind.ollama_client import OllamaClient

_EXPLANATION_SYSTEM_PROMPT = """You explain an Industrial Knowledge Intelligence system's computed state.
Never contradict the provided belief state or invent facts. Base every factual
claim on the supplied evidence and cite it as [source_type, document_id,
timestamp]. If the evidence is insufficient, say so. Be concise and useful to
a field technician."""


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class RetrievedItem:
    text: str
    metadata: Dict
    score: float = 0.0


@dataclass
class _LocalStore:
    items: List[RetrievedItem] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.items

    def add(self, item_id: str, text: str, metadata: Dict) -> None:
        if any(item.metadata["item_id"] == item_id for item in self.items):
            return
        self.items.append(RetrievedItem(text, {**metadata, "item_id": item_id}))

    def search(self, query: str, top_k: int, asset_id: Optional[str]) -> List[RetrievedItem]:
        query_terms = _tokens(query)
        scored = []
        for item in self.items:
            if asset_id and item.metadata.get("asset_id") != asset_id:
                continue
            item_terms = _tokens(item.text + " " + item.metadata.get("raw_excerpt", ""))
            score = len(query_terms & item_terms) / max(len(query_terms), 1)
            scored.append(RetrievedItem(item.text, item.metadata, score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class RAGEngine:
    """Deterministic local retrieval plus a qwen2.5:7b explanation call."""

    def __init__(self):
        cfg = get_config()
        self._client = OllamaClient(cfg)
        self._store = _LocalStore()

    def index_observations(self, observations, doc_class: str) -> None:
        for obs in observations:
            self._store.add(obs.observation_id, f"{obs.attribute}: {obs.claim}", {
                "asset_id": obs.asset_id,
                "asset_name": obs.asset_name,
                "source_type": obs.source_type,
                "document_id": obs.document_id,
                "timestamp": obs.timestamp,
                "doc_class": doc_class,
                "raw_excerpt": obs.raw_excerpt,
            })

    def ask(self, query: str, plant_memory: PlantMemory, knowledge_graph: KnowledgeGraph,
            asset_id: Optional[str] = None, top_k: int = 8) -> Dict:
        belief_state = plant_memory.get_asset_detail(asset_id) if asset_id else None
        static_refs = knowledge_graph.references_for_asset(asset_id) if asset_id else []
        retrieved = self._store.search(query, top_k, asset_id) if not self._store.is_empty() else []
        context = self._build_context(belief_state, static_refs, retrieved)
        answer = self._client.generate(
            f"--- Context ---\n{context}\n\n--- User question ---\n{query}",
            system=_EXPLANATION_SYSTEM_PROMPT,
        )
        citations = [{key: item.metadata[key] for key in ("source_type", "document_id", "timestamp", "asset_id")}
                     for item in retrieved]
        citations.extend({"source_type": ref["source_type"], "document_id": ref["document_id"],
                          "timestamp": None, "asset_id": ref["asset_id"]} for ref in static_refs)
        return {"answer": answer, "citations": citations,
                "belief_state_confidence": belief_state["confidence"] if belief_state else None,
                "asset_status": belief_state["status"] if belief_state else None}

    @staticmethod
    def _build_context(belief_state: Optional[Dict], static_refs: List[Dict], retrieved: List[RetrievedItem]) -> str:
        parts = []
        if belief_state:
            parts.append(f"Current Belief State (final):\n  Asset: {belief_state['asset_name']} ({belief_state['asset_id']})\n"
                         f"  Status: {belief_state['status']} | Risk: {belief_state['risk_label']}\n"
                         f"  Health: {belief_state['health_score']} | Risk score: {belief_state['risk_score']}\n"
                         f"  Maintenance: {belief_state['maintenance_state']} | Compliance: {belief_state['compliance_state']}\n"
                         f"  Confidence: {belief_state['confidence']}\n  HMNN: phi={belief_state['hmnn_state']['phi']}, "
                         f"mu={belief_state['hmnn_state']['mu']}, eta={belief_state['hmnn_state']['eta']}")
        if retrieved:
            parts.append("Retrieved evidence:\n" + "\n".join(
                f"  - [{item.metadata['source_type']}, {item.metadata['document_id']}, {item.metadata['timestamp']}] {item.text}"
                for item in retrieved))
        if static_refs:
            parts.append("Static reference knowledge:\n" + "\n".join(
                f"  - [{ref['source_type']}, {ref['document_id']}] {ref['attribute']}: {ref['claim']}" for ref in static_refs))
        return "\n".join(parts) if parts else "No evidence retrieved for this query."


rag_engine_singleton: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global rag_engine_singleton
    if rag_engine_singleton is None:
        rag_engine_singleton = RAGEngine()
    return rag_engine_singleton
