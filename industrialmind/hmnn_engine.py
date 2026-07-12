"""
HMNN Evidence Engine — IndustrialMind v2
=========================================
Authors: Amaresh Singh, Adarsh Sirswal (AIT Pune)

This file contains ZERO external API calls.
ZERO machine learning frameworks.
ZERO database dependencies.

It is pure mathematics — a direct translation of the HMNN paper
into the industrial knowledge domain.

Paper → Industrial mapping (Section 5 of HMNN Industrial Spec v1.0):
  d_i  (decision)         → evidence polarity    (0=concern, 1=healthy)
  e_i  (emotion)          → operational momentum (computed from history)
  s_i  (social influence) → knowledge authority  (from Authority Policy)
  φ(t) (consensus)        → evidence consensus   (do sources agree?)
  µ(t) (emotion EMA)      → operational momentum (persistence of concern)
  η(t) (vote entropy)     → evidence entropy     (are sources conflicting?)
  h(t) (hive memory)      → asset knowledge state vector

Learned scalars carried forward from paper:
  γ = 0.73  (consensus sensitivity)
  ρ = 0.80  (momentum decay / EMA weight)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

# ── Constants from HMNN paper ─────────────────────────────────────────────────
EPS   = 1e-9
GAMMA = 0.73   # learned consensus sensitivity  (paper Section 7.3)
RHO   = 0.80   # learned emotional/momentum decay (paper Section 7.3)
H_DIM = 16     # hive memory dimension

# ── Document Authority Policy ─────────────────────────────────────────────────
# Replaces social influence s_i from paper.
# These are deterministic lookups — never set by an LLM.
# Source: HMNN Industrial Spec v1.0, Section 2.2
AUTHORITY_POLICY: Dict[str, float] = {
    "safety_regulation":  1.00,   # OISD, PESO, Factory Act
    "oem_manual":         0.98,   # Manufacturer specifications
    "rca_report":         0.90,   # Root Cause Analysis
    "inspection_report":  0.92,   # Certified inspection findings
    "maintenance_log":    0.82,   # Maintenance work records
    "sensor_data":        0.85,   # Instrumentation readings
    "sop":                0.80,   # Standard Operating Procedures
    "incident_report":    0.88,   # Incident / near-miss records
    "compliance_audit":   0.91,   # Regulatory audit findings
    "operator_log":       0.70,   # Shift / operator notes
    "vendor_document":    0.55,   # Vendor communications
    "email":              0.40,   # Informal communications
    "general":            0.50,   # Unclassified
}

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Observation:
    """
    The atomic unit of knowledge the HMNN engine consumes.
    Corresponds to one evidence item extracted from one document.
    Schema: HMNN Industrial Spec v1.0, Section 3.
    """
    observation_id: str
    asset_id:       str           # e.g. "P-101"
    asset_name:     str           # e.g. "Centrifugal Pump P-101"
    asset_type:     str           # e.g. "Centrifugal Pump"
    event_type:     str           # Inspection | Maintenance | Incident | ...
    attribute:      str           # e.g. "Seal Condition"
    claim:          str           # e.g. "Seal wear observed"

    # ── HMNN Inputs ──────────────────────────────────────────────────────────
    # polarity: set by extractor from document text. Range -1.0 to +1.0
    # Normalised to [0,1] before HMNN by: p = (polarity + 1) / 2
    polarity:       float         # -1.0 (critical) to +1.0 (healthy)

    # momentum: NEVER set by extractor. Computed by MomentumEngine.
    momentum:       float = 0.0   # 0.0 (isolated) to 1.0 (persistent)

    # authority: NEVER set by extractor. Looked up from AUTHORITY_POLICY.
    authority:      float = 0.0   # set during ingest, not extraction

    # ── Metadata ─────────────────────────────────────────────────────────────
    confidence:     float = 0.90  # extraction confidence
    timestamp:      str   = ""    # ISO 8601
    source_type:    str   = "general"
    document_id:    str   = ""
    doc_class:      str   = "dynamic"  # static | dynamic
    raw_excerpt:    str   = ""

    def polarity_normalised(self) -> float:
        """Map polarity from [-1, +1] to [0, 1] for HMNN consumption."""
        return float(np.clip((self.polarity + 1.0) / 2.0, 0.0, 1.0))


@dataclass
class AssetMemory:
    """
    The Industrial Brain's evolving belief about one asset.
    h(t) from the HMNN paper — per asset instance.
    Schema: HMNN Industrial Spec v1.0, Section 2.5.
    """
    asset_id:   str
    asset_name: str
    asset_type: str = "Unknown"

    # ── Hive memory vector (raw HMNN state) ──────────────────────────────────
    h: np.ndarray = field(default_factory=lambda: np.zeros(H_DIM, dtype="float32"))

    # ── Derived belief dimensions (computed from h after each update) ─────────
    health_score:      float = 0.5    # overall operational health   [0,1]
    risk_score:        float = 0.5    # current risk level           [0,1]
    maintenance_state: str   = "Unknown"  # OK | Due | Overdue | Unknown
    compliance_state:  str   = "Unknown"  # Compliant | Gap | Unknown

    # ── HMNN signal history ───────────────────────────────────────────────────
    consensus:    float = 0.0   # φ — agreement across sources
    momentum:     float = 0.0   # µ — operational persistence
    certainty:    float = 0.0   # 1-η — inverse of evidence conflict
    last_scale:   float = 0.0   # most recent scale(t)
    confidence:   float = 0.0   # system confidence in current state

    # ── Evidence and audit trail ──────────────────────────────────────────────
    observations: List[Observation] = field(default_factory=list)
    update_log:   List[Dict]        = field(default_factory=list)

    def status_label(self) -> str:
        h = self.health_score
        if h >= 0.75: return "Healthy"
        if h >= 0.50: return "Monitor"
        if h >= 0.30: return "Warning"
        return "Critical"

    def risk_label(self) -> str:
        r = self.risk_score
        if r <= 0.25: return "Low"
        if r <= 0.50: return "Moderate"
        if r <= 0.75: return "High"
        return "Critical"


# ── Fixed projection weights ──────────────────────────────────────────────────
# Wc projects the 2D authority-diffused signal D(t) into H_DIM candidate space.
# Wo projects H_DIM hive state to 2 output classes (concern vs healthy).
# Fixed seed for reproducibility across sessions.
_rng = np.random.default_rng(seed=42)
_Wc  = _rng.normal(0, 0.1, (2, H_DIM)).astype("float32")
_bc  = np.zeros(H_DIM, dtype="float32")
_Wo  = _rng.normal(0, 0.1, (H_DIM, 2)).astype("float32")
_bo  = np.zeros(2, dtype="float32")


# ── HMNN Signal Functions (unchanged from paper, industrial semantics) ─────────

def consensus_pressure(polarities_norm: np.ndarray) -> float:
    """
    φ(t) — Consensus Pressure Gate.
    Paper Section 4.2, equation:
        φ(t) = (|r(t) - g(t)| / N + ε)^γ

    Industrial meaning:
        High φ → multiple independent evidence sources agree on asset state.
        Low φ  → sources are split (some say healthy, some say concern).

    Args:
        polarities_norm: array of normalised polarities in [0,1]
                         values > 0.5 count as "healthy vote"
                         values ≤ 0.5 count as "concern vote"
    """
    N   = len(polarities_norm)
    g   = (polarities_norm > 0.5).sum()   # "healthy" count (green votes)
    r   = N - g                            # "concern" count (red votes)
    phi = (abs(r - g) / N + EPS) ** GAMMA
    return float(phi)


def operational_momentum(polarities_norm: np.ndarray,
                         prev_mu: float) -> float:
    """
    µ(t) — Operational Momentum (replaces Emotional Momentum from paper).
    Paper Section 4.3, equation:
        µ(t) = ρ · µ(t-1) + (1-ρ) · ē(t)

    Industrial meaning:
        High µ → concern signals have been persistent over time.
        Low µ  → this is an isolated / one-off observation.

    Note: concern_level = 1 - polarity_norm
    High concern (low polarity) drives momentum upward.
    ρ = 0.80 means 80% of past momentum persists (sticky, from paper).

    Args:
        polarities_norm: normalised polarities for current observation batch
        prev_mu:         previous momentum value for this asset
    """
    concern_level = 1.0 - polarities_norm.mean()   # high concern → high momentum
    mu = RHO * prev_mu + (1 - RHO) * concern_level
    return float(mu)


def evidence_entropy(polarities_norm: np.ndarray) -> float:
    """
    η(t) — Evidence Entropy Dampener (replaces Vote Entropy from paper).
    Paper Section 4.4, Shannon entropy equation.

    Industrial meaning:
        High η → sources are contradicting each other (uncertain state).
        Low η  → sources agree, low uncertainty.
        η = 1.0 at perfect 50/50 split.
        η = 0.0 when all sources agree.

    When η is high, scale(t) is suppressed — the brain does not
    strongly revise its belief when evidence is contradictory.
    """
    N  = len(polarities_norm)
    pg = (polarities_norm > 0.5).sum() / N   # fraction "healthy"
    pr = 1.0 - pg                             # fraction "concern"
    eta = -(pr * np.log2(pr + EPS) + pg * np.log2(pg + EPS))
    return float(np.clip(eta, 0.0, 1.0))


def authority_diffusion(polarities_norm: np.ndarray,
                        momentums:       np.ndarray,
                        authorities:     np.ndarray) -> np.ndarray:
    """
    D(t) — Authority-Weighted Evidence Centroid.
    Replaces Influence Diffusion (social influence) from paper Section 4.1.

    Industrial meaning:
        High-authority sources (OEM, OISD) pull the signal centroid harder.
        Low-authority sources (emails, informal notes) have less weight.

    Returns a 2D vector [polarity_centroid, momentum_centroid]
    which is the input to the candidate projection.
    """
    weights = np.exp(authorities)
    weights = weights / (weights.sum() + EPS)
    d_pol = (weights * polarities_norm).sum()
    d_mom = (weights * momentums).sum()
    return np.array([d_pol, d_mom], dtype="float32")


def compute_scale(phi: float, mu: float, eta: float) -> float:
    """
    scale(t) = φ(t) · µ(t) · (1 - η(t))
    Paper Section 4.5.

    This is the memory update coefficient.
    Large when: sources agree (high φ) AND concern is persistent (high µ)
                AND sources don't contradict each other (low η).
    Small when: sources split, isolated event, or conflicting evidence.
    """
    return float(np.clip(phi * mu * (1.0 - eta), 0.0, 1.0))


# ── Core HMNN Update ───────────────────────────────────────────────────────────

def update_asset_memory(memory: AssetMemory,
                        new_observations: List[Observation]) -> AssetMemory:
    """
    Process a batch of new dynamic observations through HMNN.
    Updates the asset's hive memory h(t) using the convex interpolation rule.

    Paper Section 4.5:
        c(t) = ReLU(D(t) · Wc + bc)
        h(t) = h(t-1) + scale(t) · (c(t) - h(t-1))

    This is the only function that mutates AssetMemory.
    Called every time new dynamic documents are ingested about this asset.

    Args:
        memory:           current AssetMemory for this asset
        new_observations: batch of dynamic Observations (already have
                          authority and momentum set by their engines)
    Returns:
        updated AssetMemory (same object, mutated in-place)
    """
    if not new_observations:
        return memory

    # ── Extract signal arrays ────────────────────────────────────────────────
    polarities_raw  = np.array([o.polarity   for o in new_observations], dtype="float32")
    polarities_norm = np.array([o.polarity_normalised() for o in new_observations], dtype="float32")
    momentums       = np.array([o.momentum   for o in new_observations], dtype="float32")
    authorities     = np.array([o.authority  for o in new_observations], dtype="float32")

    # ── Compute HMNN signals ─────────────────────────────────────────────────
    phi  = consensus_pressure(polarities_norm)
    mu   = operational_momentum(polarities_norm, memory.momentum)
    # Entropy across ALL observations for this asset (not just current batch)
    # This captures cross-document contradiction, not just within-batch
    all_asset_pols = np.array(
        [o.polarity_normalised() for o in memory.observations] +
        list(polarities_norm),
        dtype="float32"
    )
    eta  = evidence_entropy(all_asset_pols) if len(all_asset_pols) > 1 else 0.0
    cert = 1.0 - eta
    scale = compute_scale(phi, mu, eta)

    # ── Authority diffusion → candidate state ────────────────────────────────
    D = authority_diffusion(polarities_norm, momentums, authorities)
    z = D @ _Wc + _bc
    c = np.maximum(z, 0.0)   # ReLU — paper Section 4.5

    # ── Hive memory update (convex interpolation) ────────────────────────────
    h_new = memory.h + scale * (c - memory.h)

    # ── Derive belief dimensions from updated h ──────────────────────────────
    # Health = authority-weighted mean of normalised polarity across ALL observations
    # for this asset (current batch + history already in memory.observations)
    all_asset_obs = memory.observations + new_observations
    if all_asset_obs:
        all_pol  = np.array([o.polarity_normalised() for o in all_asset_obs], dtype="float32")
        all_auth = np.array([o.authority for o in all_asset_obs], dtype="float32")
        weights  = all_auth / (all_auth.sum() + EPS)
        health   = float((weights * all_pol).sum())
    else:
        health = 0.5
    health = float(np.clip(health, 0.0, 1.0))
    risk   = 1.0 - health

    # Maintenance state heuristic from recent event types
    recent_events = [o.event_type for o in new_observations]
    if "Maintenance" in recent_events:
        maint_state = "OK"
    elif health < 0.35:
        maint_state = "Overdue"
    elif health < 0.55:
        maint_state = "Due"
    else:
        maint_state = memory.maintenance_state if memory.maintenance_state != "Unknown" else "OK"

    # Compliance state from audit observations
    if any(o.event_type in ("Compliance Audit", "Regulatory Check") for o in new_observations):
        avg_polarity = polarities_norm.mean()
        compliance   = "Compliant" if avg_polarity > 0.65 else "Gap"
    else:
        compliance = memory.compliance_state

    # Confidence grows with number of consistent, high-scale updates
    new_confidence = float(np.clip(
        memory.confidence * 0.75 + scale * 0.20 + len(new_observations) * 0.01,
        0.0, 1.0
    ))

    # ── Build audit log entry ────────────────────────────────────────────────
    log_entry = {
        "timestamp":       datetime.now().isoformat(),
        "n_observations":  len(new_observations),
        "sources":         list({o.source_type for o in new_observations}),
        "phi":             round(phi,   4),
        "mu":              round(mu,    4),
        "eta":             round(eta,   4),
        "certainty":       round(cert,  4),
        "scale":           round(scale, 4),
        "health_before":   round(memory.health_score, 4),
        "health_after":    round(health, 4),
        "health_delta":    round(health - memory.health_score, 4),
        "mean_polarity":   round(float(polarities_norm.mean()), 4),
        "mean_authority":  round(float(authorities.mean()), 4),
    }

    # ── Commit all updates ───────────────────────────────────────────────────
    memory.h                 = h_new
    memory.health_score      = health
    memory.risk_score        = risk
    memory.maintenance_state = maint_state
    memory.compliance_state  = compliance
    memory.consensus         = phi
    memory.momentum          = mu
    memory.certainty         = cert
    memory.last_scale        = scale
    memory.confidence        = new_confidence
    memory.observations.extend(new_observations)
    memory.update_log.append(log_entry)

    return memory


# ── Momentum Engine ───────────────────────────────────────────────────────────

class MomentumEngine:
    """
    Computes operational momentum µ for each (asset_id, attribute) pair
    from the full observation history.

    CRITICAL: This is the only source of momentum values.
    The extractor (Gemini) must NEVER set momentum.
    Momentum is a property of the evidence stream over time, not a document.
    """

    def compute(self,
                all_observations: List[Observation],
                asset_id: str,
                attribute: str) -> float:
        """
        µ(t) from the full history for this (asset, attribute) pair.
        Uses EMA with ρ=0.80 (from paper).
        Concern level = 1 - polarity_normalised (more concern → higher momentum).
        """
        relevant = [
            o for o in all_observations
            if o.asset_id == asset_id and o.attribute == attribute
        ]
        if not relevant:
            return 0.0

        relevant_sorted = sorted(relevant, key=lambda o: o.timestamp)
        mu = 0.0
        for obs in relevant_sorted:
            concern = 1.0 - obs.polarity_normalised()
            mu = RHO * mu + (1 - RHO) * concern

        return float(mu)

    def batch_compute(self,
                      new_observations: List[Observation],
                      history: List[Observation]) -> List[Observation]:
        """
        Assign computed momentum to each new observation before HMNN ingestion.
        Combines history + new observations for the EMA computation.
        Returns the new_observations list with momentum fields populated.
        """
        all_obs = history + new_observations
        for obs in new_observations:
            obs.momentum = self.compute(all_obs, obs.asset_id, obs.attribute)
        return new_observations


# ── Plant Memory Store ─────────────────────────────────────────────────────────

class PlantMemory:
    """
    The Industrial Brain.
    Stores one AssetMemory per asset. Orchestrates HMNN updates.
    Single source of truth for all asset belief states.
    """

    def __init__(self):
        self._assets:  Dict[str, AssetMemory] = {}
        self._all_obs: List[Observation]       = []
        self._momentum = MomentumEngine()
        self._doc_count = 0

    # ── Asset registry ────────────────────────────────────────────────────────

    def get_or_create(self, asset_id: str,
                      asset_name: str,
                      asset_type: str = "Unknown") -> AssetMemory:
        if asset_id not in self._assets:
            self._assets[asset_id] = AssetMemory(
                asset_id=asset_id,
                asset_name=asset_name,
                asset_type=asset_type
            )
        return self._assets[asset_id]

    # ── Main ingestion entry point ────────────────────────────────────────────

    def ingest_observations(self,
                            observations: List[Observation],
                            doc_class: str = "dynamic") -> Dict:
        """
        Process extracted observations into the plant brain.

        Static observations → stored for RAG context only (no HMNN update).
        Dynamic observations → momentum computed → HMNN update → memory revised.

        Args:
            observations: list of Observation objects from extractor
            doc_class:    "static" or "dynamic"
        Returns:
            dict with ingestion summary
        """
        if not observations:
            return {"status": "ok", "assets_updated": [], "observations": 0}

        # Assign authority from policy (never from LLM)
        for obs in observations:
            obs.authority = AUTHORITY_POLICY.get(obs.source_type, 0.50)
            obs.doc_class = doc_class

        self._doc_count += 1

        if doc_class == "static":
            # Static docs populate knowledge graph & RAG — no HMNN update
            self._all_obs.extend(observations)
            return {
                "status":          "ok",
                "doc_class":       "static",
                "message":         "Static document stored for knowledge graph and RAG context. HMNN not updated.",
                "assets_found":    list({o.asset_id for o in observations}),
                "observations":    len(observations),
            }

        # Dynamic: compute momentum from history, then update HMNN
        enriched = self._momentum.batch_compute(observations, self._all_obs)
        self._all_obs.extend(enriched)

        # Group by asset and run HMNN update per asset
        by_asset: Dict[str, List[Observation]] = {}
        for obs in enriched:
            by_asset.setdefault(obs.asset_id, []).append(obs)

        updated_summaries = []
        for asset_id, asset_obs in by_asset.items():
            name  = asset_obs[0].asset_name
            atype = asset_obs[0].asset_type
            mem   = self.get_or_create(asset_id, name, atype)
            before_health = mem.health_score
            update_asset_memory(mem, asset_obs)
            updated_summaries.append({
                "asset_id":      asset_id,
                "asset_name":    name,
                "health_before": round(before_health, 3),
                "health_after":  round(mem.health_score, 3),
                "health_delta":  round(mem.health_score - before_health, 3),
                "status":        mem.status_label(),
                "scale":         round(mem.last_scale, 4),
                "phi":           round(mem.consensus, 4),
                "mu":            round(mem.momentum, 4),
                "eta":           round(1 - mem.certainty, 4),
                "n_obs":         len(asset_obs),
            })

        return {
            "status":          "ok",
            "doc_class":       "dynamic",
            "assets_updated":  updated_summaries,
            "observations":    len(enriched),
        }

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_asset(self, asset_id: str) -> Optional[AssetMemory]:
        return self._assets.get(asset_id)

    def all_assets_summary(self) -> List[Dict]:
        result = []
        for m in self._assets.values():
            result.append({
                "asset_id":          m.asset_id,
                "asset_name":        m.asset_name,
                "asset_type":        m.asset_type,
                "health_score":      round(m.health_score, 3),
                "risk_score":        round(m.risk_score, 3),
                "status":            m.status_label(),
                "risk_label":        m.risk_label(),
                "maintenance_state": m.maintenance_state,
                "compliance_state":  m.compliance_state,
                "consensus":         round(m.consensus, 3),
                "momentum":          round(m.momentum, 3),
                "certainty":         round(m.certainty, 3),
                "last_scale":        round(m.last_scale, 3),
                "confidence":        round(m.confidence, 3),
                "n_observations":    len(m.observations),
                "last_updated":      m.update_log[-1]["timestamp"] if m.update_log else None,
            })
        return sorted(result, key=lambda x: x["health_score"])

    def get_asset_detail(self, asset_id: str) -> Optional[Dict]:
        m = self._assets.get(asset_id)
        if not m:
            return None
        return {
            "asset_id":          m.asset_id,
            "asset_name":        m.asset_name,
            "asset_type":        m.asset_type,
            "health_score":      round(m.health_score, 3),
            "risk_score":        round(m.risk_score, 3),
            "status":            m.status_label(),
            "risk_label":        m.risk_label(),
            "maintenance_state": m.maintenance_state,
            "compliance_state":  m.compliance_state,
            "hmnn_state": {
                "phi":       round(m.consensus,  4),
                "mu":        round(m.momentum,   4),
                "eta":       round(1 - m.certainty, 4),
                "certainty": round(m.certainty,  4),
                "scale":     round(m.last_scale, 4),
                "gamma":     GAMMA,
                "rho":       RHO,
                "interpretation": {
                    "phi":  "High = independent sources agree on asset state",
                    "mu":   "High = concern signals have persisted over time",
                    "eta":  "High = sources are contradicting each other",
                    "scale":"How strongly the last evidence batch shifted the belief",
                }
            },
            "confidence":      round(m.confidence, 3),
            "n_observations":  len(m.observations),
            "update_log":      m.update_log[-5:],  # last 5 updates
            "recent_observations": [
                {
                    "observation_id": o.observation_id,
                    "event_type":     o.event_type,
                    "attribute":      o.attribute,
                    "claim":          o.claim,
                    "polarity":       round(o.polarity, 3),
                    "momentum":       round(o.momentum, 3),
                    "authority":      round(o.authority, 3),
                    "source_type":    o.source_type,
                    "document_id":    o.document_id,
                    "timestamp":      o.timestamp,
                }
                for o in m.observations[-10:]   # last 10 observations
            ],
        }

    def stats(self) -> Dict:
        return {
            "total_assets":         len(self._assets),
            "total_observations":   len(self._all_obs),
            "dynamic_observations": sum(1 for o in self._all_obs if o.doc_class == "dynamic"),
            "static_observations":  sum(1 for o in self._all_obs if o.doc_class == "static"),
            "documents_processed":  self._doc_count,
        }

    def get_observations_for_rag(self,
                                  asset_id: Optional[str] = None,
                                  top_k: int = 10) -> List[Dict]:
        """Return recent dynamic observations for RAG context assembly."""
        obs = [o for o in self._all_obs if o.doc_class == "dynamic"]
        if asset_id:
            obs = [o for o in obs if o.asset_id == asset_id]
        obs_sorted = sorted(obs, key=lambda o: o.timestamp, reverse=True)
        return [
            {
                "asset_name":  o.asset_name,
                "event_type":  o.event_type,
                "attribute":   o.attribute,
                "claim":       o.claim,
                "polarity":    o.polarity,
                "momentum":    round(o.momentum, 3),
                "authority":   round(o.authority, 3),
                "source_type": o.source_type,
                "document_id": o.document_id,
                "timestamp":   o.timestamp,
                "excerpt":     o.raw_excerpt,
            }
            for o in obs_sorted[:top_k]
        ]


# Singleton — shared across the entire application
plant_memory = PlantMemory()