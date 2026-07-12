"""
RAG + Gemini Explanation Layer
=================================
This is the ONLY layer that talks to an LLM for answers, and it never
decides asset health/risk — that belief already lives in AssetMemory,
computed purely by hmnn_engine's analytical update rule. This layer
retrieves:

  1. The asset's current Belief State (health, risk, HMNN signals) from
     PlantMemory — already computed, not re-derived here.
  2. Supporting dynamic observations (evidence) from PlantMemory.
  3. Relevant static knowledge (OEM/SOP/regulatory) from KnowledgeGraph.

...embeds all candidate text with Gemini embeddings, ranks by cosine
similarity to the query, and asks Gemini to explain — with citations back
to document_id/source_type/timestamp — never to invent a verdict.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from google import genai
from google.genai import types

from industrialmind.config import get_config
from industrialmind.hmnn_engine import PlantMemory
from industrialmind.knowledge_graph import KnowledgeGraph

_EXPLANATION_SYSTEM_PROMPT = """You are the explanation layer of an Industrial Knowledge Intelligence system.

You do NOT decide asset health, risk, or compliance state — those are already
computed by a separate evidence-aggregation engine (HMNN) and given to you as
facts. Your job is only to explain the current belief state in plain language,
grounded strictly in the evidence provided below, and to answer the user's
question.

Rules:
- Never state a health/risk/compliance conclusion that contradicts the given
  Belief State.
- Every factual claim in your answer must cite its source using the format
  [source_type, document_id, timestamp] immediately after the claim.
- If the evidence doesn't support an answer, say so explicitly — do not guess.
- Be concise and operational: this may be read by a field technician on a phone.
"""


@dataclass
class RetrievedItem:
    text: str
    metadata: Dict
    score: float = 0.0


@dataclass
class _VectorStore:
    """Minimal in-memory embedding index — numpy cosine similarity, no external DB."""
    ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)
    _vectors: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return len(self.ids) == 0

    def add(self, item_id: str, text: str, metadata: Dict, vector: List[float]):
        if item_id in self.ids:
            return
        self.ids.append(item_id)
        self.texts.append(text)
        self.metadatas.append(metadata)
        vec = np.array(vector, dtype="float32")
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

    def search(self, query_vector: List[float], top_k: int = 8,
               asset_id: Optional[str] = None) -> List[RetrievedItem]:
        if self.is_empty():
            return []
        q = np.array(query_vector, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._vectors @ q  # cosine similarity (vectors already normalised)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            if asset_id and self.metadatas[i].get("asset_id") != asset_id:
                continue
            results.append(RetrievedItem(text=self.texts[i], metadata=self.metadatas[i], score=float(scores[i])))
            if len(results) >= top_k:
                break
        return results


class RAGEngine:
    """RAG retrieval + Gemini explanation over Asset Memory, observations, and static knowledge."""

    def __init__(self):
        cfg = get_config()
        self._cfg = cfg
        self._client = genai.Client(api_key=cfg.gemini_api_key)
        self._store = _VectorStore()

    # ── Indexing ─────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.models.embed_content(
            model=self._cfg.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]

    def index_observations(self, observations, doc_class: str):
        """Embed and index a batch of Observation objects (static or dynamic)."""
        new_items = [
            (o.observation_id, f"{o.attribute}: {o.claim}", {
                "asset_id": o.asset_id,
                "asset_name": o.asset_name,
                "source_type": o.source_type,
                "document_id": o.document_id,
                "timestamp": o.timestamp,
                "doc_class": doc_class,
                "raw_excerpt": o.raw_excerpt,
            })
            for o in observations
            if o.observation_id not in self._store.ids
        ]
        if not new_items:
            return
        vectors = self._embed([t for _, t, _ in new_items], task_type="RETRIEVAL_DOCUMENT")
        for (item_id, text, metadata), vector in zip(new_items, vectors):
            self._store.add(item_id, text, metadata, vector)

    # ── Query ────────────────────────────────────────────────────────────────

    def ask(self, query: str, plant_memory: PlantMemory, knowledge_graph: KnowledgeGraph,
             asset_id: Optional[str] = None, top_k: int = 8) -> Dict:
        """
        Answer a natural-language query, grounded in Asset Memory + retrieved
        evidence. Returns answer text, citations, and the belief-state
        confidence score (never an LLM-invented confidence).
        """
        belief_state = None
        static_refs: List[Dict] = []
        if asset_id:
            belief_state = plant_memory.get_asset_detail(asset_id)
            static_refs = knowledge_graph.references_for_asset(asset_id)

        retrieved: List[RetrievedItem] = []
        if not self._store.is_empty():
            [query_vector] = self._embed([query], task_type="RETRIEVAL_QUERY")
            retrieved = self._store.search(query_vector, top_k=top_k, asset_id=asset_id)

        context = self._build_context(belief_state, static_refs, retrieved)

        response = self._client.models.generate_content(
            model=self._cfg.explanation_model,
            contents=(
                f"{_EXPLANATION_SYSTEM_PROMPT}\n\n"
                f"--- Context ---\n{context}\n\n"
                f"--- User question ---\n{query}"
            ),
            config=types.GenerateContentConfig(temperature=0.2),
        )

        citations = [
            {
                "source_type": r.metadata["source_type"],
                "document_id": r.metadata["document_id"],
                "timestamp": r.metadata["timestamp"],
                "asset_id": r.metadata["asset_id"],
            }
            for r in retrieved
        ]
        for ref in static_refs:
            citations.append({
                "source_type": ref["source_type"],
                "document_id": ref["document_id"],
                "timestamp": None,
                "asset_id": ref["asset_id"],
            })

        return {
            "answer": response.text,
            "citations": citations,
            "belief_state_confidence": belief_state["confidence"] if belief_state else None,
            "asset_status": belief_state["status"] if belief_state else None,
        }

    @staticmethod
    def _build_context(belief_state: Optional[Dict], static_refs: List[Dict],
                        retrieved: List[RetrievedItem]) -> str:
        parts = []

        if belief_state:
            parts.append(
                "Current Belief State (computed by HMNN, already final — do not contradict):\n"
                f"  Asset: {belief_state['asset_name']} ({belief_state['asset_id']})\n"
                f"  Status: {belief_state['status']} | Risk: {belief_state['risk_label']}\n"
                f"  Health score: {belief_state['health_score']} | Risk score: {belief_state['risk_score']}\n"
                f"  Maintenance state: {belief_state['maintenance_state']} | "
                f"Compliance state: {belief_state['compliance_state']}\n"
                f"  System confidence: {belief_state['confidence']}\n"
                f"  HMNN signals: consensus(phi)={belief_state['hmnn_state']['phi']}, "
                f"momentum(mu)={belief_state['hmnn_state']['mu']}, "
                f"entropy(eta)={belief_state['hmnn_state']['eta']}\n"
            )

        if retrieved:
            parts.append("Retrieved evidence (ranked by relevance):")
            for r in retrieved:
                m = r.metadata
                parts.append(
                    f"  - [{m['source_type']}, {m['document_id']}, {m['timestamp']}] {r.text}"
                    + (f" (excerpt: \"{m['raw_excerpt']}\")" if m.get("raw_excerpt") else "")
                )

        if static_refs:
            parts.append("Static reference knowledge:")
            for ref in static_refs:
                parts.append(
                    f"  - [{ref['source_type']}, {ref['document_id']}] "
                    f"{ref['attribute']}: {ref['claim']}"
                )

        return "\n".join(parts) if parts else "No evidence retrieved for this query."


rag_engine_singleton: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global rag_engine_singleton
    if rag_engine_singleton is None:
        rag_engine_singleton = RAGEngine()
    return rag_engine_singleton
