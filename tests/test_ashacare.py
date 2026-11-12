"""Tests for AshaCare AI — model + API."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ashacare.api import app
from ashacare.model import EXAMPLE_CASES, FEATURES, TriageModel, validate_vitals

BASE = Path(__file__).resolve().parent.parent

HIGH_RISK_CASE = {
    "Age": 40, "SystolicBP": 160, "DiastolicBP": 100,
    "BS": 15.0, "BodyTemp": 101.0, "HeartRate": 90,
}
LOW_RISK_CASE = {
    "Age": 25, "SystolicBP": 110, "DiastolicBP": 70,
    "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 72,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def model():
    return TriageModel.load(BASE)


def test_model_loads_and_predicts_three_classes(model):
    assert model.model is not None
    result = model.predict(HIGH_RISK_CASE)
    assert result["risk_level"] in ("Low", "Mid", "High")
    assert set(result["probabilities"]) == {"Low", "Mid", "High"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
    assert "referral" in result


def test_high_risk_case_is_high(model):
    result = model.predict(HIGH_RISK_CASE)
    assert result["risk_level"] == "High", result
    assert result["probabilities"]["High"] > 0.5


def test_low_risk_case_is_low(model):
    result = model.predict(LOW_RISK_CASE)
    assert result["risk_level"] == "Low", result


def test_health_returns_metrics(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["test_metrics"]["accuracy"] > 0.7
    assert body["dataset_rows"] == 1014
    assert body["features"] == FEATURES


def test_triage_valid_high_risk(client):
    r = client.post("/triage", json=HIGH_RISK_CASE)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] == "High"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert body["top_drivers"], "expected at least one risk driver"
    assert "referral" in body


def test_triage_rejects_invalid_input(client):
    # Systolic BP far outside any plausible range.
    bad = dict(LOW_RISK_CASE, SystolicBP=999)
    assert client.post("/triage", json=bad).status_code == 422
    # Missing a vital.
    missing = {k: v for k, v in LOW_RISK_CASE.items() if k != "BS"}
    assert client.post("/triage", json=missing).status_code == 422
    # Wrong type.
    wrong = dict(LOW_RISK_CASE, Age="twenty-five")
    assert client.post("/triage", json=wrong).status_code == 422


def test_batch_ranks_high_risk_first(client):
    r = client.post("/triage/batch", json={"cases": [
        {**LOW_RISK_CASE, "name": "A"},
        {**HIGH_RISK_CASE, "name": "B"},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["priority_list"][0]["name"] == "B"
    assert body["priority_list"][0]["risk_level"] == "High"
    assert body["priority_list"][1]["risk_level"] == "Low"


# --- Referral-urgency tiers (rule-based, not model output) ---------------

def test_urgency_tier_matches_risk_level(model):
    high = model.predict(HIGH_RISK_CASE)
    assert high["urgency"]["tier"] == "URGENT"
    assert high["urgency"]["window"] == "within 24 hours"
    assert high["urgency"]["escalated"] is False
    assert "not a model output" in high["urgency"]["basis"]

    low = model.predict(LOW_RISK_CASE)
    assert low["urgency"]["tier"] == "ROUTINE"
    assert low["urgency"]["window"] == "at the next scheduled visit"
    assert low["urgency"]["escalated"] is False


def test_urgency_tier_for_each_risk_level():
    from ashacare.model import urgency_for

    assert urgency_for("high risk", LOW_RISK_CASE)["tier"] == "URGENT"
    mid = urgency_for("mid risk", LOW_RISK_CASE)
    assert mid["tier"] == "SOON" and mid["window"] == "within 7 days"
    low = urgency_for("low risk", LOW_RISK_CASE)
    assert low["tier"] == "ROUTINE"


def test_urgency_escalates_on_extreme_vitals(model):
    crisis = model.predict({**LOW_RISK_CASE, "SystolicBP": 190})
    assert crisis["urgency"]["tier"] == "IMMEDIATE"
    assert crisis["urgency"]["escalated"] is True
    assert "Systolic BP" in crisis["urgency"]["escalation_reason"]
    assert "Emergency referral" in crisis["referral"]

    sugar = model.predict({**LOW_RISK_CASE, "BS": 22.0})
    assert sugar["urgency"]["tier"] == "IMMEDIATE"
    assert "Blood sugar" in sugar["urgency"]["escalation_reason"]

    fever = model.predict({**LOW_RISK_CASE, "BodyTemp": 104.0})
    assert fever["urgency"]["tier"] == "IMMEDIATE"

    # Just under the thresholds: no escalation.
    assert model.predict({**LOW_RISK_CASE, "SystolicBP": 179})["urgency"]["escalated"] is False


def test_triage_response_carries_urgency(client):
    r = client.post("/triage", json=HIGH_RISK_CASE)
    assert r.status_code == 200
    body = r.json()
    assert body["urgency"]["tier"] == "URGENT"
    assert body["urgency"]["window"] == "within 24 hours"


def test_health_lists_urgency_tiers(client):
    body = client.get("/health").json()
    assert body["urgency_tiers"] == {
        "High": {"tier": "URGENT", "window": "within 24 hours"},
        "Mid": {"tier": "SOON", "window": "within 7 days"},
        "Low": {"tier": "ROUTINE", "window": "at the next scheduled visit"},
    }


# --- Input validation ----------------------------------------------------

def test_validate_vitals_accepts_boundaries():
    from ashacare.model import validate_vitals

    edge = {"Age": 10, "SystolicBP": 220, "DiastolicBP": 40,
            "BS": 3.0, "BodyTemp": 106.0, "HeartRate": 140}
    cleaned, errors = validate_vitals(edge)
    assert errors == [] and len(cleaned) == 6


def test_validate_vitals_reports_each_problem():
    from ashacare.model import validate_vitals

    _, errors = validate_vitals({"Age": 25})  # 5 fields missing
    assert len(errors) == 5 and any("missing" in e for e in errors)

    _, errors = validate_vitals({**LOW_RISK_CASE, "BS": "high"})
    assert any("not a number" in e for e in errors)

    _, errors = validate_vitals({**LOW_RISK_CASE, "HeartRate": 300})
    assert any("outside the plausible range" in e for e in errors)

    _, errors = validate_vitals({**LOW_RISK_CASE, "BodyTemp": None})
    assert any("missing body temperature" in e for e in errors)


@pytest.mark.parametrize("field,bad", [
    ("Age", 5), ("Age", 90),
    ("SystolicBP", 60), ("SystolicBP", 250),
    ("DiastolicBP", 30), ("DiastolicBP", 150),
    ("BS", 1.0), ("BS", 40.0),
    ("BodyTemp", 90.0), ("BodyTemp", 110.0),
    ("HeartRate", 30), ("HeartRate", 200),
])
def test_triage_rejects_each_vital_out_of_range(client, field, bad):
    r = client.post("/triage", json={**LOW_RISK_CASE, field: bad})
    assert r.status_code == 422, (field, bad, r.text)


def test_batch_skips_bad_rows_instead_of_failing(client):
    r = client.post("/triage/batch", json={"cases": [
        {**LOW_RISK_CASE, "name": "Good"},
        {**HIGH_RISK_CASE, "name": "BadBP", "SystolicBP": 999},
        {**LOW_RISK_CASE, "name": "BadType", "BS": "high"},
        {**LOW_RISK_CASE, "name": "MissingHR", "HeartRate": None},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["priority_list"][0]["name"] == "Good"
    assert body["skipped_count"] == 3
    assert [s["index"] for s in body["skipped"]] == [1, 2, 3]
    assert body["skipped"][0]["name"] == "BadBP"
    assert any("outside the plausible range" in e for e in body["skipped"][0]["errors"])
    assert any("not a number" in e for e in body["skipped"][1]["errors"])
    assert any("missing" in e for e in body["skipped"][2]["errors"])
    assert body["counts"] == {"High": 0, "Mid": 0, "Low": 1}


def test_batch_all_bad_rows_returns_empty_list(client):
    r = client.post("/triage/batch", json={"cases": [
        {**LOW_RISK_CASE, "SystolicBP": 999},
        {**LOW_RISK_CASE, "Age": "x"},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["skipped_count"] == 2
    assert body["priority_list"] == []


# --- Plain-language explanations (v3) -------------------------------------

def test_driver_sentences_compare_with_usual_range():
    from ashacare.model import driver_sentence

    assert driver_sentence("SystolicBP", 160) == (
        "Systolic blood pressure (upper number) is 160 mm Hg — "
        "above the usual 120 mm Hg"
    )
    assert "above the usual 3.9–7.8 mmol/L" in driver_sentence("BS", 15.0)
    assert "below the usual 120 mm Hg" in driver_sentence("SystolicBP", 100)
    assert "within the usual 97–99 °F" in driver_sentence("BodyTemp", 98.0)
    assert "within the usual 60–100 bpm" in driver_sentence("HeartRate", 72)
    assert driver_sentence("Age", 40) == "Age is 40 years"


def test_top_drivers_carry_sentences(model):
    result = model.predict(HIGH_RISK_CASE)
    assert result["top_drivers"], "expected drivers for a high-risk case"
    for d in result["top_drivers"]:
        assert "sentence" in d and d["label"].lower() in d["sentence"].lower()
        assert str(int(d["value"])) in d["sentence"] or f"{d['value']:g}" in d["sentence"]


def test_plain_summary_high_risk(model):
    result = model.predict(HIGH_RISK_CASE)
    summary = result["plain_summary"]
    assert summary.startswith("High risk.")
    assert "systolic blood pressure" in summary
    assert "above the usual" in summary
    assert "within 24 hours" in summary


def test_plain_summary_low_risk_mentions_usual_range(model):
    result = model.predict(LOW_RISK_CASE)
    summary = result["plain_summary"]
    assert summary.startswith("Low risk.")
    assert "within the usual" in summary or "below the usual" in summary
    assert "next scheduled visit" in summary


def test_plain_summary_escalated_case(model):
    result = model.predict({**LOW_RISK_CASE, "SystolicBP": 190})
    assert "Emergency referral TODAY" in result["plain_summary"]
    assert "IMMEDIATE" in result["plain_summary"] or "today" in result["plain_summary"].lower()


def test_triage_response_carries_plain_summary(client):
    body = client.post("/triage", json=HIGH_RISK_CASE).json()
    assert "plain_summary" in body
    assert body["plain_summary"].startswith("High risk.")
    assert all("sentence" in d for d in body["top_drivers"])
    batch = client.post("/triage/batch", json={"cases": [
        {**HIGH_RISK_CASE, "name": "B"}, {**LOW_RISK_CASE, "name": "A"},
    ]}).json()
    assert batch["priority_list"][0]["plain_summary"].startswith("High risk.")


# --- Printable visit summary (v3) -----------------------------------------

def test_printable_summary_ranks_and_counts(client):
    from ashacare.summary import build_printable_summary

    batch = client.post("/triage/batch", json={"cases": [
        {**LOW_RISK_CASE, "name": "Asha"},
        {**HIGH_RISK_CASE, "name": "Meena"},
        {**LOW_RISK_CASE, "name": "BadBP", "SystolicBP": 999},
    ]}).json()
    text = build_printable_summary(
        batch["priority_list"], counts=batch["counts"],
        skipped=batch["skipped"], date_str="2026-11-08",
    )
    assert "Village visit priority list" in text
    assert "Date: 2026-11-08" in text
    assert "1 high risk (refer within 24 hours)" in text
    assert "1 low risk (routine)" in text
    assert "Skipped: 1 row(s)" in text and "BadBP" in text
    # Highest risk first, with action and reason in plain words.
    assert text.index("1. Meena") < text.index("2. Asha")
    assert "Meena — High risk, URGENT (within 24 hours)" in text
    assert "Action: Refer to the PHC" in text
    assert "Why: systolic blood pressure" in text
    assert "not a diagnosis" in text


def test_printable_summary_empty_roster():
    from ashacare.summary import build_printable_summary

    text = build_printable_summary([], counts={"High": 0, "Mid": 0, "Low": 0})
    assert "Scored: 0 mothers" in text
    assert "Visit in this order" in text


def test_example_cases_predict_expected_risk(model):
    # The dashboard's one-click examples must demo the risk level they claim.
    for key, case in EXAMPLE_CASES.items():
        vitals, errors = validate_vitals(case["vitals"])
        assert errors == [], (key, errors)
        result = model.predict(vitals)
        assert result["risk_level"] == case["expected_risk"], (key, result)
        assert result["confidence"] > 0.9, (key, result)


def test_example_cases_cover_all_risk_levels():
    assert {c["expected_risk"] for c in EXAMPLE_CASES.values()} == {
        "Low", "Mid", "High",
    }
    for case in EXAMPLE_CASES.values():
        assert set(case["vitals"]) == set(FEATURES)
        assert case["label"] and case["description"]


def _load_sample_roster():
    import pandas as pd

    df = pd.read_csv(BASE / "data" / "sample_roster.csv")
    assert list(df.columns) == ["name"] + FEATURES
    return df


def test_sample_roster_scores_all_risk_levels(model):
    # The sample roster is the judge's one-click batch demo: real dataset
    # rows, one per risk level at least, all predicted at high confidence.
    df = _load_sample_roster()
    assert len(df) == 8
    good, bad = [], []
    for _, row in df.iterrows():
        raw = {f: (None if pd.isna(row[f]) else row[f]) for f in FEATURES}
        vitals, errors = validate_vitals(raw)
        if errors:
            bad.append((row["name"], errors))
        else:
            good.append(model.predict(vitals))
    assert len(good) == 7, bad
    assert len(bad) == 1 and bad[0][0] == "Geeta"  # missing blood sugar
    assert {r["risk_level"] for r in good} == {"Low", "Mid", "High"}
    assert all(r["confidence"] > 0.9 for r in good)


def test_sample_roster_batch_ranks_high_first(client):
    import pandas as pd

    df = _load_sample_roster()
    cases = []
    for _, row in df.iterrows():
        raw = {f: (None if pd.isna(row[f]) else row[f]) for f in FEATURES}
        vitals, errors = validate_vitals(raw)
        if not errors:
            cases.append(vitals)
    r = client.post("/triage/batch", json={"cases": cases})
    assert r.status_code == 200
    body = r.json()
    assert body["priority_list"][0]["risk_level"] == "High"
    assert body["priority_list"][-1]["risk_level"] == "Low"
    assert body["counts"]["High"] == 3
    assert body["counts"]["Mid"] == 2
    assert body["counts"]["Low"] == 2
