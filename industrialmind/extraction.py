"""
OCR / Entity Extraction + Observation Engine
==============================================
Turns raw document text (already OCR'd/parsed upstream from PDFs, scanned
forms, spreadsheets, email archives, P&IDs) into atomic `Observation`
records — the schema HMNN consumes (hmnn_engine.Observation).

The LLM (Gemini) is used ONLY for extraction: pulling structured claims out
of unstructured text. It never sets `authority` or `momentum` (see
hmnn_engine.py header) and its `source_type` guess is treated as a hint —
classification.classify_document() makes the deterministic call.
"""

import json
import uuid
from datetime import datetime, date
from typing import List

from google import genai
from google.genai import types

from industrialmind.config import get_config
from industrialmind.hmnn_engine import Observation

_OBSERVATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "asset_id": {
            "type": "string",
            "description": "Equipment tag as written in the document, e.g. 'P-101'. "
                            "Use 'PLANT' if the observation is plant-wide, not asset-specific.",
        },
        "asset_name": {"type": "string", "description": "Human-readable asset name, e.g. 'Centrifugal Pump P-101'."},
        "asset_type": {"type": "string", "description": "Equipment class, e.g. 'Centrifugal Pump', 'Pressure Vessel'."},
        "event_type": {
            "type": "string",
            "description": "One of: Inspection, Maintenance, Incident, RCA, Compliance Audit, "
                            "Regulatory Check, Operator Note, Reference.",
        },
        "attribute": {"type": "string", "description": "The specific parameter/condition observed, e.g. 'Seal Condition'."},
        "claim": {"type": "string", "description": "One-sentence factual claim extracted verbatim in meaning, e.g. 'Seal wear observed'."},
        "polarity": {
            "type": "number",
            "description": "-1.0 (critical/severe concern) to +1.0 (fully healthy/compliant). "
                            "0.0 is neutral/informational.",
        },
        "confidence": {"type": "number", "description": "Extractor's confidence in this extraction, 0.0-1.0."},
        "source_type_hint": {
            "type": "string",
            "description": "Best guess at document type: safety_regulation, oem_manual, rca_report, "
                            "inspection_report, maintenance_log, sensor_data, sop, incident_report, "
                            "compliance_audit, operator_log, vendor_document, email, general.",
        },
        "event_date": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD) if stated in the document, else empty string."},
        "raw_excerpt": {"type": "string", "description": "The exact sentence/phrase from the source text this was extracted from."},
    },
    "required": ["asset_id", "asset_name", "event_type", "attribute", "claim", "polarity", "confidence", "raw_excerpt"],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {"type": "array", "items": _OBSERVATION_ITEM_SCHEMA},
    },
    "required": ["observations"],
}

_EXTRACTION_SYSTEM_PROMPT = """You are an industrial document extraction engine.

Read the document text and extract every discrete, verifiable observation about
plant equipment, procedures, or regulatory content. Each observation must be
atomic — one equipment tag, one attribute, one claim.

Rules:
- Do NOT invent data. If an asset tag isn't stated, use "PLANT" for plant-wide claims.
- polarity reflects severity/health: critical problems near -1.0, healthy/compliant
  findings near +1.0, purely informational content near 0.0.
- Extract reference material (OEM specs, SOP steps, regulatory clauses) too —
  use event_type "Reference" and polarity 0.0 for these.
- raw_excerpt must be a real substring/paraphrase from the document, not fabricated.
- If the document contains no extractable claims, return an empty observations list.
"""


class ExtractionEngine:
    """Gemini-backed OCR/Entity Extraction stage of the pipeline."""

    def __init__(self):
        cfg = get_config()
        self._cfg = cfg
        self._client = genai.Client(api_key=cfg.gemini_api_key)

    def extract(self, document_text: str, document_id: str,
                filename: str = "") -> List[Observation]:
        """
        Extract structured Observations from raw document text.

        Args:
            document_text: full extracted text of the document (already OCR'd
                            upstream if it was a scan/image).
            document_id:   stable id for the source document, used for citations.
            filename:      original filename, passed through for context only.
        Returns:
            List of Observation objects with authority/momentum left at their
            defaults — those are set later by classification + MomentumEngine.
        """
        if not document_text.strip():
            return []

        prompt = (
            f"{_EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Filename: {filename or document_id}\n\n"
            f"Document text:\n\"\"\"\n{document_text}\n\"\"\""
        )

        response = self._client.models.generate_content(
            model=self._cfg.extraction_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.1,
            ),
        )

        payload = json.loads(response.text)
        raw_items = payload.get("observations", [])

        observations = []
        for item in raw_items:
            observations.append(self._to_observation(item, document_id))
        return observations

    @staticmethod
    def _to_observation(item: dict, document_id: str) -> Observation:
        event_date = item.get("event_date") or ""
        timestamp = _coerce_timestamp(event_date)
        source_type_hint = item.get("source_type_hint", "").strip() or "general"

        return Observation(
            observation_id=str(uuid.uuid4()),
            asset_id=item["asset_id"].strip() or "PLANT",
            asset_name=item.get("asset_name", "").strip() or item["asset_id"],
            asset_type=item.get("asset_type", "").strip() or "Unknown",
            event_type=item.get("event_type", "Reference").strip(),
            attribute=item.get("attribute", "").strip() or "General",
            claim=item["claim"].strip(),
            polarity=float(max(-1.0, min(1.0, item.get("polarity", 0.0)))),
            confidence=float(max(0.0, min(1.0, item.get("confidence", 0.9)))),
            timestamp=timestamp,
            source_type=source_type_hint,
            document_id=document_id,
            raw_excerpt=item.get("raw_excerpt", "").strip(),
    )


def _coerce_timestamp(event_date: str) -> str:
    if event_date:
        try:
            return datetime.fromisoformat(event_date).isoformat()
        except ValueError:
            pass
    return datetime.combine(date.today(), datetime.min.time()).isoformat()
