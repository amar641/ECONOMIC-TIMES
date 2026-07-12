"""
ARK-25 — Alpha Refinery Knowledge Benchmark
==============================================
Synthetic document set for a fictional refinery ("Alpha Refinery"), all
documents referencing the same fixed set of assets over a Jan-Jun 2026
timeline so the HMNN Evidence Engine has real trajectories to aggregate:
a degrading asset, a stable asset, a dip-then-recover asset, an asset with
contradictory evidence (tests entropy), and a slow-building concern (tests
momentum).

Includes both static reference documents (OEM manuals, SOPs, regulations)
and dynamic evidence documents (inspections, maintenance, incidents, RCAs,
operator logs, compliance audits) — filenames are written to match
classification.py's routing rules.

generate() returns documents in memory; write_to_disk() also drops them as
.txt files under data/ark25/ for human/judge inspection.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

ASSETS = {
    "P-101": {"name": "Centrifugal Pump P-101", "type": "Centrifugal Pump"},
    "V-204": {"name": "Pressure Vessel V-204", "type": "Pressure Vessel"},
    "E-305": {"name": "Shell & Tube Heat Exchanger E-305", "type": "Heat Exchanger"},
    "C-102": {"name": "Reciprocating Compressor C-102", "type": "Reciprocating Compressor"},
    "T-501": {"name": "Crude Storage Tank T-501", "type": "Storage Tank"},
}


@dataclass
class ArkDocument:
    filename: str
    text: str
    document_id: str


def _doc(docs: List[ArkDocument], filename: str, text: str):
    document_id = f"ARK25-{len(docs) + 1:03d}"
    docs.append(ArkDocument(filename=filename, text=text.strip(), document_id=document_id))


def _static_documents(docs: List[ArkDocument]):
    _doc(docs, "OEM_Manual_Centrifugal_Pump.txt", f"""
OEM Manufacturer Specification — Centrifugal Pump Class (covers {ASSETS['P-101']['name']})
Manufacturer: Alpha Rotating Equipment Co.

Seal: API 682 mechanical seal, expected service life 18-24 months under
normal operating conditions. Seal wear beyond 0.5mm face clearance requires
scheduled replacement within 30 days.

Vibration limits: sustained vibration above 4.5 mm/s RMS indicates bearing
or alignment degradation and should trigger inspection within 7 days.

Recommended inspection interval: quarterly visual, annual full teardown.
""")

    _doc(docs, "OEM_Manual_Heat_Exchanger.txt", f"""
OEM Manufacturer Specification — Shell & Tube Heat Exchanger Class (covers {ASSETS['E-305']['name']})
Manufacturer: Alpha Thermal Systems Ltd.

Fouling factor threshold: a drop in thermal efficiency exceeding 15% from
baseline indicates tube-side fouling and warrants chemical cleaning.
Recommended cleaning interval: every 6 months or on efficiency threshold
breach, whichever comes first.
""")

    _doc(docs, "SOP_Pump_Seal_Maintenance.txt", f"""
Standard Operating Procedure — Mechanical Seal Inspection and Replacement
Applies to: Centrifugal Pump class, including {ASSETS['P-101']['name']}.

1. Isolate and lock out pump per LOTO procedure.
2. Inspect seal faces for scoring, wear, or leakage residue.
3. Measure face clearance; record against OEM threshold (0.5mm).
4. If clearance exceeds threshold, replace seal assembly using OEM part kit.
5. Log all findings in the maintenance record with asset tag and date.
""")

    _doc(docs, "SOP_Compressor_Vibration_Monitoring.txt", f"""
Standard Operating Procedure — Reciprocating Compressor Vibration Monitoring
Applies to: {ASSETS['C-102']['name']} and class.

1. Record vibration readings at each monitoring point during routine rounds.
2. Compare against baseline; flag any reading more than 20% above baseline.
3. Cross-check flagged readings against operator shift observations before
   escalating to a maintenance work order.
""")

    _doc(docs, "OISD_118_Storage_Tank_Safety.txt", f"""
OISD-118 — Layout and Safety of Oil and Gas Installations: Storage Tanks
Applicable to atmospheric storage tanks including {ASSETS['T-501']['name']}.

Clause 6.3: Tank shell corrosion exceeding 10% of nominal wall thickness
requires immediate engineering assessment and, where confirmed, tank
re-rating or repair before continued service. Corrosion monitoring
inspections are mandatory at least every 6 months for tanks in crude
service.
""")

    _doc(docs, "PESO_Pressure_Vessel_Regulation.txt", f"""
PESO Regulation — Pressure Vessel Compliance Requirements
Applicable to pressure vessels including {ASSETS['V-204']['name']}.

All pressure vessels must undergo statutory hydrostatic testing and
compliance audit on an annual basis. A vessel is deemed compliant when the
audit finds no deviation from the approved design pressure, safety valve
certification, and inspection record completeness.
""")


def _dynamic_documents(docs: List[ArkDocument]):
    p = ASSETS["P-101"]["name"]
    _doc(docs, "P-101_Inspection_Report_2026-01-15.txt", f"""
Inspection Report — {p} — 2026-01-15
Inspector: Field Inspection Team, Alpha Refinery

Routine quarterly inspection. Seal face clearance measured at 0.2mm, well
within OEM threshold. No leakage observed. Pump running smoothly. Overall
condition: healthy.
""")

    _doc(docs, "P-101_Maintenance_Log_2026-02-10.txt", f"""
Maintenance Log — {p} — 2026-02-10
Technician: Maintenance Crew B

Routine lubrication and bearing check performed per schedule. No anomalies
found. Seal not inspected this cycle (not due).
""")

    _doc(docs, "P-101_Inspection_Report_2026-03-12.txt", f"""
Inspection Report — {p} — 2026-03-12
Inspector: Field Inspection Team, Alpha Refinery

Quarterly inspection. Seal face clearance now measured at 0.45mm, approaching
the OEM 0.5mm replacement threshold. Minor seal wear observed, slight
weeping at seal housing. Recommend close monitoring and scheduling seal
replacement within the next cycle.
""")

    _doc(docs, "P-101_Operator_Log_2026-03-20.txt", f"""
Shift Log — {p} — 2026-03-20
Operator: Night Shift, Unit 3

Noted minor dripping near pump seal housing during rounds. Wiped down and
continued monitoring, no immediate action taken. Will flag to maintenance
if it worsens.
""")

    _doc(docs, "P-101_Inspection_Report_2026-04-18.txt", f"""
Inspection Report — {p} — 2026-04-18
Inspector: Field Inspection Team, Alpha Refinery

Follow-up inspection. Seal face clearance now at 0.65mm, exceeding OEM
threshold. Visible leakage at seal housing confirmed. Seal wear has
progressed since March. Condition: concerning, replacement overdue.
""")

    _doc(docs, "P-101_Incident_Report_2026-05-02.txt", f"""
Incident Report — {p} — 2026-05-02
Reported by: Unit 3 Operations

Mechanical seal failure resulted in process fluid leak at pump housing.
Unit was isolated and shut down per emergency procedure. No injuries. Minor
environmental containment required. Root cause investigation initiated.
""")

    _doc(docs, "P-101_RCA_2026-05-10.txt", f"""
Root Cause Analysis — {p} Seal Failure — 2026-05-10
Prepared by: Reliability Engineering, Alpha Refinery

Finding: Seal replacement was not performed after the 2026-03-12 inspection
flagged clearance approaching threshold, and the 2026-04-18 inspection
confirmed threshold exceedance. The gap between detection and corrective
action exceeded the SOP-recommended 30-day replacement window, resulting in
seal failure. Corrective action: replace seal, and enforce automatic work
order generation when OEM thresholds are exceeded.
""")

    _doc(docs, "P-101_Maintenance_Log_2026-05-12.txt", f"""
Maintenance Log — {p} — 2026-05-12
Technician: Maintenance Crew A

Mechanical seal assembly replaced with new OEM API 682 kit per RCA
corrective action. Face clearance re-measured post-install at 0.05mm.
Pump returned to service, running normally.
""")

    v = ASSETS["V-204"]["name"]
    _doc(docs, "V-204_Inspection_Report_2026-01-20.txt", f"""
Inspection Report — {v} — 2026-01-20
Inspector: Field Inspection Team, Alpha Refinery

Routine inspection. Vessel shell, nozzles, and safety valve certification
all within normal parameters. No deviations found. Condition: healthy.
""")

    _doc(docs, "V-204_Compliance_Audit_2026-03-05.txt", f"""
Compliance Audit — {v} — 2026-03-05
Auditor: Regulatory Compliance Team

Annual PESO compliance audit completed. Hydrostatic test passed, safety
valve certification current, inspection records complete. No deviations
from approved design pressure. Compliance state: Compliant.
""")

    _doc(docs, "V-204_Inspection_Report_2026-05-22.txt", f"""
Inspection Report — {v} — 2026-05-22
Inspector: Field Inspection Team, Alpha Refinery

Routine inspection. Vessel in good condition, no corrosion or deviation
noted. Condition: healthy.
""")

    e = ASSETS["E-305"]["name"]
    _doc(docs, "E-305_Inspection_Report_2026-01-25.txt", f"""
Inspection Report — {e} — 2026-01-25
Inspector: Field Inspection Team, Alpha Refinery

Routine inspection. Thermal efficiency at 96% of baseline. No fouling
indicators. Condition: healthy.
""")

    _doc(docs, "E-305_Inspection_Report_2026-03-28.txt", f"""
Inspection Report — {e} — 2026-03-28
Inspector: Field Inspection Team, Alpha Refinery

Thermal efficiency dropped to 79% of baseline, exceeding the 15% fouling
threshold in the OEM manual. Tube-side fouling suspected. Recommend
chemical cleaning per OEM schedule.
""")

    _doc(docs, "E-305_Maintenance_Log_2026-04-08.txt", f"""
Maintenance Log — {e} — 2026-04-08
Technician: Maintenance Crew B

Chemical cleaning of tube bundle performed per OEM recommendation following
fouling finding. Post-cleaning efficiency check scheduled for next
inspection cycle.
""")

    _doc(docs, "E-305_Inspection_Report_2026-05-15.txt", f"""
Inspection Report — {e} — 2026-05-15
Inspector: Field Inspection Team, Alpha Refinery

Follow-up inspection post-cleaning. Thermal efficiency recovered to 94% of
baseline. Fouling resolved. Condition: healthy.
""")

    c = ASSETS["C-102"]["name"]
    _doc(docs, "C-102_Inspection_Report_2026-02-14.txt", f"""
Inspection Report — {c} — 2026-02-14
Inspector: Field Inspection Team, Alpha Refinery

Vibration reading at monitoring point 2 measured 24% above baseline,
exceeding SOP flag threshold. Recommend cross-check against operator
observations before escalation.
""")

    _doc(docs, "C-102_Operator_Log_2026-02-15.txt", f"""
Shift Log — {c} — 2026-02-15
Operator: Day Shift, Unit 2

Compressor running smoothly, no unusual noise or vibration felt during
rounds. No concerns to report this shift.
""")

    _doc(docs, "C-102_Operator_Log_2026-02-16.txt", f"""
Shift Log — {c} — 2026-02-16
Operator: Night Shift, Unit 2

Compressor sounds normal, all gauges within range. Nothing notable this
shift.
""")

    _doc(docs, "C-102_Maintenance_Log_2026-02-20.txt", f"""
Maintenance Log — {c} — 2026-02-20
Technician: Maintenance Crew A

Investigated elevated vibration reading from 2026-02-14 inspection. Found
minor mounting bolt looseness at monitoring point 2, torqued to spec.
Vibration reading re-checked and within normal range after correction.
""")

    t = ASSETS["T-501"]["name"]
    _doc(docs, "T-501_Inspection_Report_2026-01-30.txt", f"""
Inspection Report — {t} — 2026-01-30
Inspector: Field Inspection Team, Alpha Refinery

Routine corrosion monitoring inspection per OISD-118. Shell wall thickness
loss measured at 4% of nominal. Within acceptable range. Condition: healthy,
continue monitoring.
""")

    _doc(docs, "T-501_Inspection_Report_2026-04-02.txt", f"""
Inspection Report — {t} — 2026-04-02
Inspector: Field Inspection Team, Alpha Refinery

Corrosion monitoring inspection per OISD-118. Shell wall thickness loss now
measured at 7% of nominal, up from 4% in January. Trend is worsening.
Continue close monitoring; approaching the 10% clause 6.3 threshold.
""")

    _doc(docs, "T-501_Inspection_Report_2026-06-10.txt", f"""
Inspection Report — {t} — 2026-06-10
Inspector: Field Inspection Team, Alpha Refinery

Corrosion monitoring inspection per OISD-118. Shell wall thickness loss now
measured at 9.5% of nominal, approaching the clause 6.3 threshold of 10%.
Recommend engineering assessment be scheduled proactively before the
mandatory threshold is reached. Compliance state: gap risk if untreated.
""")

    _doc(docs, "T-501_Compliance_Audit_2026-06-15.txt", f"""
Compliance Audit — {t} — 2026-06-15
Auditor: Regulatory Compliance Team

OISD-118 clause 6.3 review triggered by the 2026-06-10 corrosion finding.
Engineering assessment not yet completed at time of audit. Compliance gap
identified pending assessment and, if required, tank re-rating or repair.
""")


def generate() -> List[ArkDocument]:
    """Build the full ARK-25 document set in memory, in a fixed narrative order."""
    docs: List[ArkDocument] = []
    _static_documents(docs)
    _dynamic_documents(docs)
    return docs


def write_to_disk(docs: List[ArkDocument], output_dir: str = "data/ark25") -> str:
    os.makedirs(output_dir, exist_ok=True)
    for d in docs:
        path = os.path.join(output_dir, d.filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(d.text)
    return output_dir


def as_ingest_batch(docs: List[ArkDocument]) -> List[Dict]:
    """Format for IndustrialBrain.ingest_batch()."""
    return [{"filename": d.filename, "text": d.text, "document_id": d.document_id} for d in docs]
