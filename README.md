# AshaCare AI

**Maternal-health triage copilot for frontline (ASHA) health workers** — built for
[GenHacks 2026](https://genhacks-open.devpost.com/) (Healthcare & Environmental track).

An ASHA worker in rural India visits pregnant women with only a BP cuff, a
thermometer and a glucometer. AshaCare AI tells her, on the spot, which mothers
are high-risk and need urgent referral — and why, in plain language.

## Demo video

🎥 **Watch the demo (68 seconds):** https://www.youtube.com/watch?v=RUKevNL949M
(unlisted) — high-risk case → low-risk contrast → batch roster → printable
summary, with narration and captions. Every screen shown is the real running
app; every case is a real row from the UCI dataset.

## What it does

1. Worker enters **6 basic vitals** — age, systolic BP, diastolic BP, blood
sugar, body temperature, heart rate.
2. An **XGBoost classifier** (trained on the UCI Maternal Health Risk dataset,
1,014 cases) outputs a risk level — **Low / Mid / High** — with class
probabilities.
3. Each prediction comes with **plain-language risk drivers** — the vitals
actually pushing this case toward its risk level, measured from the model by
baseline replacement (replace a vital with the training median, see how much
the predicted probability falls) — each one written as a sentence against
the usual range ("Systolic blood pressure is 160 mm Hg — above the usual
120 mm Hg"), plus a one-paragraph `plain_summary` the worker can read
as-is, and **rule-based referral urgency** (see below).
4. **Batch mode:** upload a whole village roster as CSV and get a risk-ranked
priority list — who to visit first. Bad rows (missing / non-numeric /
out-of-range vitals) are skipped and listed, never scored on guessed
values and never failing the whole roster. Try it instantly with the
included sample roster — `data/sample_roster.csv` (8 real dataset rows,
3 High / 2 Mid / 2 Low plus 1 row with a missing blood-sugar reading
that gets skipped and listed; names are illustrative) — downloadable
from the dashboard's Batch tab.
5. **Printable visit summary:** the ranked priority list downloads as plain
text (.txt) — counts by urgency, one plain-language line per mother, the
skipped rows — so it works on paper or a basic phone with no signal.

## Referral urgency (rule-based — not a model output)

The model only predicts the risk level. The urgency tier, time window and
action are looked up from an explicit table, and a single extreme vital
escalates the case to emergency referral whatever the model said:

| Risk level | Urgency tier | Window | Action |
|---|---|---|---|
| High | 🔴 URGENT | within 24 hours | Refer to the PHC / hospital within 24 hours; re-check vitals the same day |
| Mid | 🟡 SOON | within 7 days | Follow-up visit within 1 week; re-check BP and blood sugar |
| Low | 🟢 ROUTINE | at next scheduled visit | Routine antenatal care |

Escalation to 🚨 **IMMEDIATE — emergency referral today**, regardless of the
predicted risk level: systolic BP ≥ 180 mm Hg or diastolic BP ≥ 120 mm Hg
(hypertensive crisis), blood sugar ≥ 20 mmol/L (severe hyperglycaemia), or
body temperature ≥ 103.5 °F (high fever). Every response carries
`urgency.basis` stating this guidance is rule-based.

## Input validation

All 6 vitals are checked against plausible physiological ranges (age 10–70,
systolic BP 70–220, diastolic BP 40–140, blood sugar 3–30 mmol/L, body
temperature 95–106 °F, heart rate 40–140 bpm). Out-of-range, missing or
non-numeric input is rejected — `422` from `POST /triage`, skipped-and-listed
from `POST /triage/batch` and the dashboard CSV upload — never silently
clipped or guessed.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py            # trains the model, saves model.joblib + model_meta.json
uvicorn ashacare.api:app --reload          # API on http://localhost:8000
streamlit run dashboard.py                 # dashboard on http://localhost:8501
pytest                                     # tests
```

## API

- `GET /health` — model status + held-out test metrics.
- `POST /triage` — one case, 6 vitals in, risk level + probabilities + top
risk drivers (each with a plain-language `sentence`) + a one-paragraph
`plain_summary` + referral guidance out.
- `POST /triage/batch` — a roster of cases, returned ranked High → Low,
with per-tier `counts` and any bad rows listed under `skipped`.
- Printable visit summary: `ashacare.summary.build_printable_summary`
turns a scored roster into plain text (dashboard download button).

Example:

```bash
curl -X POST http://localhost:8000/triage -H 'Content-Type: application/json' \
  -d '{"Age": 40, "SystolicBP": 160, "DiastolicBP": 100, "BS": 15, "BodyTemp": 101, "HeartRate": 90}'
```

```json
{
  "risk_level": "High",
  "probabilities": {"Low": 0.0001, "Mid": 0.0005, "High": 0.9994},
  "confidence": 0.9994,
  "top_drivers": [
    {"feature": "SystolicBP", "label": "systolic blood pressure (upper number)", "value": 160.0, "contribution": 0.0277, "sentence": "Systolic blood pressure (upper number) is 160 mm Hg — above the usual 120 mm Hg"}
  ],
  "plain_summary": "High risk. Mainly because systolic blood pressure (upper number) is 160 mm Hg — above the usual 120 mm Hg. Refer to the PHC / hospital within 24 hours and re-check vitals the same day.",
  "referral": "Refer to the PHC / hospital within 24 hours and re-check vitals the same day.",
  "urgency": {
    "tier": "URGENT",
    "window": "within 24 hours",
    "action": "Refer to the PHC / hospital within 24 hours and re-check vitals the same day.",
    "escalated": false,
    "escalation_reason": null,
    "basis": "Rule-based guidance from the predicted risk level and standard clinical thresholds; not a model output."
  }
}
```

## Dataset & model

- **Dataset:** UCI Maternal Health Risk dataset — 1,014 cases, 6 vitals,
  target `RiskLevel` (low/mid/high risk). Source:
  https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip
  (copy in `data/`). Class balance: low 406 (40.0%), mid 336 (33.1%),
  high 272 (26.8%). No missing values.
- **Model:** XGBoost, 200 trees, max depth 4, stratified 80/20 split.
  Held-out test accuracy 0.8522 (high-risk recall 0.964 — see `model_meta.json`
  for per-class precision/recall and the confusion matrix). Retrain anytime with
  `python train.py`.
- **Disclaimer:** screening aid only, not a diagnosis. High-risk cases must
  be referred to a doctor.

## How it maps to the GenHacks judging criteria

- **Impact:** maternal mortality and delayed referral are a documented,
severe public-health problem in India; the users are ~1M ASHA frontline
workers who today have no triage tool at all. The product acts where the
problem is — at the doorstep, with 6 vitals a worker can already measure.
- **Technical execution:** a real XGBoost model trained on a real public
dataset (1,014 cases, held-out accuracy 0.8522, high-risk recall 0.964),
a working FastAPI backend + Streamlit dashboard, batch roster triage with
bad-row handling, a ready-to-upload sample roster, and 40 passing tests.
- **Creativity:** plain-language, model-measured explanations plus
rule-based referral urgency with clinical red-flag escalation — and a
printable paper summary for villages with no signal — not just a score.
- **Presentation:** the demo video shows the full loop in 68 seconds —
  https://www.youtube.com/watch?v=RUKevNL949M — enter vitals → risk +
  explanation → priority list → printable summary.

## Status

- **v8 (Nov 13, 2026):** demo video recorded and uploaded —
  https://www.youtube.com/watch?v=RUKevNL949M (68.5s, 1080p, unlisted,
  verified playing) — real dashboard captures (high-risk example, low-risk
  contrast, batch roster with the skipped row, the actual printable summary
  file), narration + burned-in captions; built with the proven PayGuard
  video pipeline (Playwright screenshots + ffmpeg). 40/40 tests pass, API +
  dashboard live-verified before recording.
- **v7 (Nov 12, 2026):** public GitHub repo live —
  https://github.com/sahilsaeidkhan/ashacare-ai (all 15 files, MIT license
  detected, README renders) — and clean-checkout re-verification of
  saved-v6 from a fresh archive (fresh venv, `train.py` reproduces
  accuracy 0.8522 with the identical confusion matrix, 40/40 tests,
  API `/health` + `/triage` + `/triage/batch` live, dashboard HTTP 200).
  The submission's repo link and "functional demo" claim now both hold
  from a cold start. 40/40 tests pass.
- **v6 (Nov 11, 2026):** sample roster for instant batch demo —
`data/sample_roster.csv` (8 real UCI rows: 3 High / 2 Mid / 2 Low, all
model-verified at confidence > 0.9, plus 1 row with a missing blood-sugar
reading that is skipped and listed), downloadable from the dashboard's
Batch tab; 2 new tests (roster rows validate + cover all risk levels,
batch endpoint ranks High first). 40/40 tests pass, API + dashboard
live-verified.
- **v5 (Nov 10, 2026):** clean-checkout re-verification of saved-v4 from a
fresh clone (fresh venv, `train.py` reproduces accuracy 0.8522 with the
same confusion matrix, 38/38 tests, API `/health` + `/triage` live,
dashboard HTTP 200) — the submission's "functional demo" claim now holds
from a cold start, not just from the dev machine. Stale "36 passing
tests" in the judging-criteria map corrected to 38.
- **v4 (Nov 9, 2026):** one-click demo examples — High / Mid / Low real
dataset rows (model-verified, confidence > 0.99) load into the single-case
form with one click, so a judge or the video can demo any risk level
instantly; demo-video script written. 38/38 tests pass.
- **v3 (Nov 8, 2026):** plain-language explanation sentences (each driver
compared with the usual range), one-paragraph `plain_summary` per case,
printable visit summary (.txt) for single cases and rosters, submission
writeup draft. 36/36 tests pass.
- **v2 (Nov 7, 2026):** referral-urgency tiers + clinical red-flag
escalation to IMMEDIATE, shared `VITAL_RANGES` validation, batch CSV
skips bad rows with reasons, dashboard urgency banner + tier counts.
28/28 tests pass.
- **v1 (Nov 6, 2026):** XGBoost triage end to end — dataset, model,
FastAPI + Streamlit, batch ranking. 7/7 tests pass.

## Project layout

```
train.py            # training + evaluation; writes model.joblib, model_meta.json
ashacare/
  model.py          # model loading, prediction, measured explanations
  api.py            # FastAPI app (/health, /triage, /triage/batch)
  summary.py        # printable plain-text visit summary for a scored roster
dashboard.py        # Streamlit dashboard (single case + batch CSV roster)
data/               # UCI dataset CSV
tests/              # pytest suite (model + API)
```

## License

MIT — see `LICENSE`.
