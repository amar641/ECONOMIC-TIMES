"""
Knowledge Classification
=========================
Decides, for each incoming document, two things the extractor is never
trusted to decide on its own:

  1. source_type  — which entry in hmnn_engine.AUTHORITY_POLICY applies.
  2. doc_class    — "static" (reference knowledge -> Knowledge Graph only)
                     or "dynamic" (plant events -> HMNN Evidence Engine).

This mirrors the Industrial Spec: authority and static/dynamic routing are
deterministic policy decisions, not LLM outputs, so they live here as a
rule-based classifier the extractor's source_type guess is checked against.
"""

import os
import re
from dataclasses import dataclass

from industrialmind.hmnn_engine import AUTHORITY_POLICY

# source_type values that describe plant events (feed HMNN) vs reference
# material (feed the Knowledge Graph only).
STATIC_SOURCE_TYPES = {
    "safety_regulation",
    "oem_manual",
    "sop",
    "vendor_document",
}
DYNAMIC_SOURCE_TYPES = {
    "inspection_report",
    "maintenance_log",
    "incident_report",
    "rca_report",
    "compliance_audit",
    "operator_log",
    "sensor_data",
}

# Ordered (pattern, source_type) rules, checked against the FILENAME ONLY,
# after underscores/hyphens are normalised to spaces (so "\b" boundaries work
# the same as they would on natural-language text — "_" is a \w character,
# so "P-101_Inspection" would otherwise never satisfy \binspection\b).
# A dynamic event report (an inspection, an incident...) routinely *mentions*
# OEM thresholds or regulation clause numbers in its body text without being
# a reference document itself, so filename — not body content — is the
# authoritative signal for "what kind of document is this". First match wins.
_FILENAME_RULES = [
    (r"\brca\b|root cause analysis", "rca_report"),
    (r"\binspection\b", "inspection_report"),
    (r"\b(maintenance log|work order|maintenance record)\b", "maintenance_log"),
    (r"\b(incident|near ?miss)\b", "incident_report"),
    (r"\b(compliance audit|regulatory audit)\b", "compliance_audit"),
    (r"\b(shift log|operator log|operator note)\b", "operator_log"),
    (r"\bsensor|telemetry|instrumentation reading\b", "sensor_data"),
    (r"\b(oisd|peso|factory act|environmental norms?|regulat\w*)\b", "safety_regulation"),
    (r"\b(oem|manufacturer specification|equipment manual)\b", "oem_manual"),
    (r"\b(sop|standard operating procedure)\b", "sop"),
    (r"\bvendor\b", "vendor_document"),
    (r"\bemail\b", "email"),
]

# Fallback rules applied to the document BODY, only used when the filename
# gives no signal at all (e.g. "notes.txt", an email export with no
# descriptive name).
_BODY_FALLBACK_RULES = [
    (r"\b(oisd|peso|factory act|environmental norms?)\b", "safety_regulation"),
    (r"\b(oem manufacturer specification|equipment manual)\b", "oem_manual"),
    (r"\bstandard operating procedure\b", "sop"),
    (r"\broot cause analysis\b", "rca_report"),
    (r"\binspection report\b", "inspection_report"),
    (r"\b(maintenance log|work order)\b", "maintenance_log"),
    (r"\b(incident report|near[- ]miss)\b", "incident_report"),
    (r"\b(compliance audit|regulatory audit)\b", "compliance_audit"),
    (r"\b(shift log|operator log)\b", "operator_log"),
    (r"\bemail\b", "email"),
]


@dataclass
class ClassificationResult:
    source_type: str
    doc_class: str          # "static" | "dynamic"
    authority: float
    matched_rule: str       # which rule fired, or "fallback"


def classify_document(filename: str, text_snippet: str,
                       extractor_hint: str = "") -> ClassificationResult:
    """
    Classify a document into a source_type + static/dynamic doc_class.

    Args:
        filename:       original filename (e.g. "P-101_Inspection_Mar2026.pdf")
        text_snippet:   first ~2000 chars of extracted document text
        extractor_hint: source_type the LLM extractor guessed (optional,
                         used only as a tie-breaker fallback — never trusted
                         outright, since authority must be deterministic)
    """
    fname = re.sub(r"[_\-]+", " ", os.path.basename(filename).lower())
    for pattern, source_type in _FILENAME_RULES:
        if re.search(pattern, fname):
            return _build_result(source_type, matched_rule=f"filename:{pattern}")

    body = text_snippet[:2000].lower()
    for pattern, source_type in _BODY_FALLBACK_RULES:
        if re.search(pattern, body):
            return _build_result(source_type, matched_rule=f"body:{pattern}")

    if extractor_hint in AUTHORITY_POLICY:
        return _build_result(extractor_hint, matched_rule="extractor_hint")

    return _build_result("general", matched_rule="fallback")


def _build_result(source_type: str, matched_rule: str) -> ClassificationResult:
    doc_class = "static" if source_type in STATIC_SOURCE_TYPES else "dynamic"
    authority = AUTHORITY_POLICY.get(source_type, 0.50)
    return ClassificationResult(
        source_type=source_type,
        doc_class=doc_class,
        authority=authority,
        matched_rule=matched_rule,
    )
