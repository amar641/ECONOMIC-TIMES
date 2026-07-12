"""
Industrial Brain — orchestrator
==================================
Wires the full pipeline from the architecture diagram into one entry point:

  Documents -> Extraction -> Knowledge Classification
       -> {Static -> Knowledge Graph, Dynamic -> HMNN Evidence Engine}
       -> Industrial Brain (Asset Memory)
       -> RAG + Gemini (explanation layer)

Two public methods: ingest_document() and ask().
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from industrialmind.classification import classify_document
from industrialmind.extraction import ExtractionEngine
from industrialmind.hmnn_engine import PlantMemory
from industrialmind.knowledge_graph import KnowledgeGraph
from industrialmind.rag import RAGEngine


@dataclass
class IngestResult:
    filename: str
    document_id: str
    doc_class: str
    source_type: str
    observations_extracted: int
    engine_result: Dict
    warnings: List[str] = field(default_factory=list)


class IndustrialBrain:
    """
    Single entry point for the Industrial Knowledge Intelligence platform.
    Holds one PlantMemory (dynamic asset belief) and one KnowledgeGraph
    (static reference knowledge), and routes every ingested document through
    classification before either engine sees it.
    """

    def __init__(self):
        self.plant_memory = PlantMemory()
        self.knowledge_graph = KnowledgeGraph()
        self._extractor = ExtractionEngine()
        self._rag = RAGEngine()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_document(self, filename: str, document_text: str, document_id: str) -> IngestResult:
        """
        Process one document end to end:
          1. Extract observations (Gemini).
          2. Classify the document (static vs dynamic, source_type/authority).
          3. Route: static -> Knowledge Graph, dynamic -> HMNN via PlantMemory.
          4. Index everything into the RAG vector store.
        """
        warnings: List[str] = []

        observations = self._extractor.extract(document_text, document_id, filename)
        if not observations:
            warnings.append("No observations extracted from this document.")

        classification = classify_document(
            filename, document_text,
            extractor_hint=observations[0].source_type if observations else "",
        )

        for obs in observations:
            obs.source_type = classification.source_type
            obs.authority = classification.authority
            obs.doc_class = classification.doc_class

        if classification.doc_class == "static":
            engine_result = self.knowledge_graph.add_static_observations(observations)
            self.plant_memory.ingest_observations(observations, doc_class="static")
        else:
            engine_result = self.plant_memory.ingest_observations(observations, doc_class="dynamic")

        if observations:
            self._rag.index_observations(observations, doc_class=classification.doc_class)

        return IngestResult(
            filename=filename,
            document_id=document_id,
            doc_class=classification.doc_class,
            source_type=classification.source_type,
            observations_extracted=len(observations),
            engine_result=engine_result,
            warnings=warnings,
        )

    def ingest_batch(self, documents: List[Dict]) -> List[IngestResult]:
        """documents: list of {"filename", "text", "document_id"}."""
        return [self.ingest_document(d["filename"], d["text"], d["document_id"]) for d in documents]

    # ── Query ────────────────────────────────────────────────────────────────

    def ask(self, query: str, asset_id: Optional[str] = None, top_k: int = 8) -> Dict:
        return self._rag.ask(query, self.plant_memory, self.knowledge_graph, asset_id=asset_id, top_k=top_k)

    # ── Dashboards ───────────────────────────────────────────────────────────

    def asset_dashboard(self) -> List[Dict]:
        return self.plant_memory.all_assets_summary()

    def asset_detail(self, asset_id: str) -> Optional[Dict]:
        return self.plant_memory.get_asset_detail(asset_id)

    def stats(self) -> Dict:
        return {
            "plant_memory": self.plant_memory.stats(),
            "knowledge_graph": self.knowledge_graph.stats(),
        }
