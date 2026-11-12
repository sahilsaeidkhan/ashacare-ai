"""FastAPI backend for AshaCare AI.

Endpoints:
  GET  /health   — model status + held-out test metrics.
  POST /triage   — risk prediction for one case (6 vitals).
  POST /triage/batch — risk predictions for a roster (list of cases).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ashacare.model import (
    FEATURES,
    RISK_LABELS,
    URGENCY_TIERS,
    VITAL_RANGES,
    TriageModel,
    validate_vitals,
)

app = FastAPI(
    title="AshaCare AI",
    description=(
        "Maternal-health triage copilot for frontline (ASHA) health workers. "
        "An XGBoost model predicts Low / Mid / High maternal risk from 6 "
        "basic vitals and explains each prediction in plain language."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_model() -> TriageModel:
    return TriageModel.load(Path(__file__).resolve().parent.parent)


class Vitals(BaseModel):
    """Six vitals an ASHA worker can measure with a BP cuff, thermometer
    and glucometer. Bounds come from VITAL_RANGES (ashacare.model) — the
    same ranges the batch validator uses; anything outside is rejected
    with 422, never silently clipped."""

    Age: float = Field(..., ge=VITAL_RANGES["Age"][0], le=VITAL_RANGES["Age"][1], description="Age in years")
    SystolicBP: float = Field(..., ge=VITAL_RANGES["SystolicBP"][0], le=VITAL_RANGES["SystolicBP"][1], description="Systolic BP (mm Hg)")
    DiastolicBP: float = Field(..., ge=VITAL_RANGES["DiastolicBP"][0], le=VITAL_RANGES["DiastolicBP"][1], description="Diastolic BP (mm Hg)")
    BS: float = Field(..., ge=VITAL_RANGES["BS"][0], le=VITAL_RANGES["BS"][1], description="Blood sugar (mmol/L)")
    BodyTemp: float = Field(..., ge=VITAL_RANGES["BodyTemp"][0], le=VITAL_RANGES["BodyTemp"][1], description="Body temperature (°F)")
    HeartRate: float = Field(..., ge=VITAL_RANGES["HeartRate"][0], le=VITAL_RANGES["HeartRate"][1], description="Heart rate (bpm)")

    def as_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in FEATURES}


class BatchRequest(BaseModel):
    # Raw dicts on purpose: each row is validated individually so one bad
    # row is skipped and reported instead of failing the whole roster.
    cases: list[dict]


@app.get("/health")
def health():
    model = get_model()
    meta = model.meta
    return {
        "status": "ok",
        "model": "XGBoost 3-class classifier (maternal health risk)",
        "features": FEATURES,
        "classes": ["Low", "Mid", "High"],
        "dataset": meta.get("dataset_source"),
        "dataset_rows": meta.get("dataset_rows"),
        "test_metrics": meta.get("test_metrics"),
        "trained_on": meta.get("trained_on"),
        # The documented, rule-based referral-urgency tiers (not model output).
        "urgency_tiers": {
            RISK_LABELS[risk_class]: {"tier": tier["tier"], "window": tier["window"]}
            for risk_class, tier in URGENCY_TIERS.items()
        },
    }


@app.post("/triage")
def triage(vitals: Vitals):
    return get_model().predict(vitals.as_dict())


@app.post("/triage/batch")
def triage_batch(request: BatchRequest):
    """Score a roster and return it ranked by risk (High first), so the
    worker sees who to visit first. A bad row (missing / non-numeric /
    out-of-range vital) is SKIPPED and reported in `skipped` — it never
    fails the whole roster, and it is never scored on guessed values."""
    model = get_model()
    rank = {"High": 0, "Mid": 1, "Low": 2}
    scored = []
    skipped = []
    for i, raw in enumerate(request.cases):
        if not isinstance(raw, dict):
            skipped.append({"index": i, "name": "", "errors": ["not a case object"]})
            continue
        name = raw.get("name", "") or ""
        vitals, errors = validate_vitals(raw)
        if errors:
            skipped.append({"index": i, "name": name, "errors": errors})
            continue
        result = model.predict(vitals)
        scored.append({"name": name, "vitals": vitals, **result})
    scored.sort(key=lambda r: (rank[r["risk_level"]], -r["confidence"]))
    counts = {level: sum(1 for r in scored if r["risk_level"] == level)
              for level in ("High", "Mid", "Low")}
    return {
        "count": len(scored),
        "counts": counts,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "priority_list": scored,
    }
