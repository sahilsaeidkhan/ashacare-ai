"""Model loading, prediction and plain-language explanation.

Explanations are measured from the model, not invented: for each vital we
replace it with the training-set median and measure how much the predicted
class probability drops. The vitals whose replacement drops the probability
most are the ones driving this prediction. Referral urgency is a separate,
explicitly rule-based recommendation (labelled as such in the API response),
derived from the predicted risk level plus standard clinical thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
CLASSES = ["low risk", "mid risk", "high risk"]

# Plain-language labels for each vital, used in explanations.
FEATURE_LABELS = {
    "Age": "age",
    "SystolicBP": "systolic blood pressure (upper number)",
    "DiastolicBP": "diastolic blood pressure (lower number)",
    "BS": "blood sugar",
    "BodyTemp": "body temperature",
    "HeartRate": "heart rate",
}

# Usual (healthy) reference range for each vital, used to turn a measured
# model driver into a plain-language sentence an ASHA worker can read
# ("Systolic BP is 160 mm Hg — above the usual 120 mm Hg"). Values are
# standard clinical reference points, NOT model outputs: systolic ~120 and
# diastolic ~80 mm Hg; blood sugar 3.9–7.8 mmol/L (fasting to after-meal);
# body temperature 97–99 °F; resting heart rate 60–100 bpm. Age has no
# healthy reference — its sentence just states the value.
REFERENCE_RANGES = {
    "Age": None,
    "SystolicBP": (120, 120, "120 mm Hg"),
    "DiastolicBP": (80, 80, "80 mm Hg"),
    "BS": (3.9, 7.8, "3.9–7.8 mmol/L"),
    "BodyTemp": (97.0, 99.0, "97–99 °F"),
    "HeartRate": (60, 100, "60–100 bpm"),
}


def driver_sentence(feature: str, value: float) -> str:
    """One plain-language sentence for a risk driver, e.g.
    'Systolic blood pressure (upper number) is 160 mm Hg — above the usual
    120 mm Hg'. Compares the reading against the usual reference range;
    Age just states the value, since age has no healthy reference."""
    label = FEATURE_LABELS[feature]
    sentence_label = label[0].upper() + label[1:]
    ref = REFERENCE_RANGES[feature]
    if ref is None:
        return f"{sentence_label} is {value:g} years"
    lo, hi, ref_text = ref
    unit = VITAL_RANGES[feature][2]
    if value > hi:
        relation = f"above the usual {ref_text}"
    elif value < lo:
        relation = f"below the usual {ref_text}"
    else:
        relation = f"within the usual {ref_text}"
    return f"{sentence_label} is {value:g} {unit} — {relation}"


# Plausible physiological ranges for each vital. Single source of truth:
# the API (pydantic) and the batch/dashboard validators both use these, so a
# range change in one place applies everywhere. Bounds are generous
# (dataset ranges widened) — anything outside is rejected, not silently
# clipped, because a mistyped vital must never produce a confident-looking
# risk score.
VITAL_RANGES = {
    # field: (min, max, unit)
    "Age": (10, 70, "years"),
    "SystolicBP": (70, 220, "mm Hg"),
    "DiastolicBP": (40, 140, "mm Hg"),
    "BS": (3, 30, "mmol/L"),
    "BodyTemp": (95, 106, "°F"),
    "HeartRate": (40, 140, "bpm"),
}


def validate_vitals(raw: dict) -> tuple[dict[str, float], list[str]]:
    """Validate one case's raw vitals (e.g. one CSV row or one batch entry).

    Returns (cleaned vitals, errors). `cleaned` is only complete when
    `errors` is empty. Checks, in order: missing field, non-numeric value,
    out-of-range value. Error strings are plain-language so they can be
    shown to a health worker as-is.
    """
    cleaned: dict[str, float] = {}
    errors: list[str] = []
    for feature in FEATURES:
        if feature not in raw or raw[feature] is None or (
            isinstance(raw[feature], str) and raw[feature].strip() == ""
        ):
            errors.append(f"missing {FEATURE_LABELS[feature]}")
            continue
        try:
            value = float(raw[feature])
        except (TypeError, ValueError):
            errors.append(
                f"{FEATURE_LABELS[feature]} is not a number: {raw[feature]!r}"
            )
            continue
        lo, hi, unit = VITAL_RANGES[feature]
        if not (lo <= value <= hi):
            errors.append(
                f"{FEATURE_LABELS[feature]} {value:g} {unit} is outside the "
                f"plausible range {lo:g}–{hi:g} {unit} — check the reading"
            )
            continue
        cleaned[feature] = value
    return cleaned, errors

# Human-readable risk labels.
RISK_LABELS = {"low risk": "Low", "mid risk": "Mid", "high risk": "High"}

# Referral-urgency tiers — explicit, documented, RULE-BASED guidance.
# The model only predicts the risk level; the urgency tier, the time window
# and the action are looked up from the table below (plus the escalation
# rules underneath). They are labelled as rules in every response, never
# presented as model output.
#
#   risk level  tier      window                    action
#   High        URGENT    within 24 hours           refer to PHC / hospital
#   Mid         SOON      within 7 days             follow-up visit
#   Low         ROUTINE   at next scheduled visit   routine antenatal care
URGENCY_TIERS = {
    "high risk": {
        "tier": "URGENT",
        "window": "within 24 hours",
        "action": (
            "Refer to the PHC / hospital within 24 hours and re-check "
            "vitals the same day."
        ),
    },
    "mid risk": {
        "tier": "SOON",
        "window": "within 7 days",
        "action": (
            "Schedule a follow-up visit within 1 week and re-check blood "
            "pressure and blood sugar."
        ),
    },
    "low risk": {
        "tier": "ROUTINE",
        "window": "at the next scheduled visit",
        "action": (
            "Routine antenatal care; re-check vitals at the next scheduled "
            "visit."
        ),
    },
}

# Escalation rules: a single extreme vital overrides the tier to IMMEDIATE,
# whatever the model predicted. Thresholds are standard clinical red flags
# (hypertensive crisis, severe hyperglycaemia, high fever). Also rule-based.
ESCALATION_RULES = [
    ("SystolicBP", 180, None,
     "Systolic BP {value:g} mm Hg — hypertensive crisis (≥ 180 mm Hg)"),
    ("DiastolicBP", 120, None,
     "Diastolic BP {value:g} mm Hg — hypertensive crisis (≥ 120 mm Hg)"),
    ("BS", 20, None,
     "Blood sugar {value:g} mmol/L — severe hyperglycaemia (≥ 20 mmol/L)"),
    ("BodyTemp", 103.5, None,
     "Body temperature {value:g} °F — high fever (≥ 103.5 °F)"),
]

URGENCY_BASIS = (
    "Rule-based guidance from the predicted risk level and standard "
    "clinical thresholds; not a model output."
)

# Backwards-compatible one-line referral text, kept for existing clients.
REFERRAL_RULES = {
    risk: tier["action"] for risk, tier in URGENCY_TIERS.items()
}

IMMEDIATE_ACTION = (
    "Emergency referral TODAY — do not wait. Take the mother to the PHC / "
    "hospital now and re-check vitals on arrival."
)

# One-click demo cases for the dashboard — real rows from the UCI Maternal
# Health Risk dataset, verified against the trained model (confidence
# > 0.99 each). A judge or a health worker can load one instead of typing
# six vitals by hand, and the demo video uses them. The expected risk is
# asserted in tests, so a future model retrain that changes any of these
# fails loudly instead of silently demoing the wrong answer.
EXAMPLE_CASES = {
    "high": {
        "label": "High-risk example",
        "description": (
            "Age 25, BP 140/100, blood sugar 15 — high risk, refer "
            "within 24 hours."
        ),
        "vitals": {
            "Age": 25, "SystolicBP": 140, "DiastolicBP": 100,
            "BS": 15.0, "BodyTemp": 98.6, "HeartRate": 70,
        },
        "expected_risk": "High",
    },
    "mid": {
        "label": "Mid-risk example",
        "description": (
            "Age 29, BP 130/70, blood sugar 6.7 — mid risk, follow up "
            "within a week."
        ),
        "vitals": {
            "Age": 29, "SystolicBP": 130, "DiastolicBP": 70,
            "BS": 6.7, "BodyTemp": 98.0, "HeartRate": 78,
        },
        "expected_risk": "Mid",
    },
    "low": {
        "label": "Low-risk example",
        "description": (
            "Age 21, BP 75/50, blood sugar 7.5 — low risk, routine care."
        ),
        "vitals": {
            "Age": 21, "SystolicBP": 75, "DiastolicBP": 50,
            "BS": 7.5, "BodyTemp": 98.0, "HeartRate": 60,
        },
        "expected_risk": "Low",
    },
}


def urgency_for(risk_class: str, vitals: dict[str, float]) -> dict:
    """Urgency tier for a case: base tier from the risk level, escalated to
    IMMEDIATE if any single vital crosses a clinical red-flag threshold."""
    tier = dict(URGENCY_TIERS[risk_class])
    escalation_reason = None
    for feature, threshold, _upper, template in ESCALATION_RULES:
        if vitals[feature] >= threshold:
            escalation_reason = template.format(value=vitals[feature])
            break
    if escalation_reason is not None:
        return {
            "tier": "IMMEDIATE",
            "window": "today — emergency referral",
            "action": IMMEDIATE_ACTION,
            "escalated": True,
            "escalation_reason": escalation_reason,
            "basis": URGENCY_BASIS,
        }
    return {**tier, "escalated": False, "escalation_reason": None,
            "basis": URGENCY_BASIS}


class TriageModel:
    """Wraps the trained XGBoost model plus its metadata."""

    def __init__(self, model_path: Path, meta_path: Path):
        self.model = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text())
        self.medians = np.array(self.meta["feature_medians"], dtype=float)
        # Class order of predict_proba columns is the explicit encoding in
        # train.py (low=0, mid=1, high=2); read it from meta, not from the
        # model's integer classes_.
        self.classes = list(self.meta["classes"])

    @classmethod
    def load(cls, base_dir: Path | None = None) -> "TriageModel":
        base = base_dir or Path(__file__).resolve().parent.parent
        return cls(base / "model.joblib", base / "model_meta.json")

    def predict(self, vitals: dict[str, float]) -> dict:
        """Predict risk for one case; returns level, probabilities, drivers."""
        x = np.array([[float(vitals[f]) for f in FEATURES]], dtype=float)
        proba = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = self.classes[pred_idx]

        probabilities = {
            RISK_LABELS[c]: round(float(p), 4)
            for c, p in zip(self.classes, proba)
        }

        drivers = self._top_drivers(x[0], pred_idx, proba[pred_idx])
        urgency = urgency_for(pred_class, vitals)
        referral = (
            IMMEDIATE_ACTION if urgency["escalated"]
            else REFERRAL_RULES[pred_class]
        )
        # One plain-language paragraph a health worker can read as-is:
        # risk level, why (each driver in words, compared with the usual
        # range), and what to do. Falls back to a generic reason when no
        # single vital stood out.
        if drivers:
            reasons = "; ".join(d["sentence"] for d in drivers)
            reasons = reasons[0].lower() + reasons[1:]
            why = f"Mainly because {reasons}."
        else:
            why = ("No single vital stood out — the overall pattern of the "
                   "six readings points to this risk level.")
        plain_summary = (
            f"{RISK_LABELS[pred_class]} risk. {why} {urgency['action']}"
        )
        return {
            "risk_level": RISK_LABELS[pred_class],
            "risk_class": pred_class,
            "probabilities": probabilities,
            "confidence": round(float(proba[pred_idx]), 4),
            "top_drivers": drivers,
            "plain_summary": plain_summary,
            "referral": referral,
            "referral_basis": URGENCY_BASIS,
            "urgency": urgency,
        }

    def _top_drivers(self, x: np.ndarray, pred_idx: int, base_proba: float) -> list[dict]:
        """Baseline-replacement feature contributions, measured from the model.

        For each vital, replace it with the training median and see how much
        the predicted class probability falls. A big fall means that vital
        was pushing the prediction toward this risk level.
        """
        contributions = []
        for i, feature in enumerate(FEATURES):
            x_perturbed = x.copy()
            x_perturbed[i] = self.medians[i]
            p = float(self.model.predict_proba(x_perturbed.reshape(1, -1))[0][pred_idx])
            drop = base_proba - p
            contributions.append(
                {"feature": feature, "label": FEATURE_LABELS[feature],
                 "value": float(x[i]), "contribution": round(float(drop), 4),
                 "sentence": driver_sentence(feature, float(x[i]))}
            )
        # Only keep vitals that actually pushed toward this prediction.
        positive = [c for c in contributions if c["contribution"] > 0.005]
        positive.sort(key=lambda c: c["contribution"], reverse=True)
        return positive[:3]
