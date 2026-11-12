"""AshaCare AI — Streamlit dashboard for ASHA workers.

Single-case triage form + batch CSV upload with a risk-ranked priority list.
Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ashacare.model import EXAMPLE_CASES, FEATURES, TriageModel, validate_vitals  # noqa: E402
from ashacare.summary import build_printable_summary  # noqa: E402

st.set_page_config(page_title="AshaCare AI — Maternal Health Triage", layout="wide")
st.title("AshaCare AI")
st.write(
    "Maternal-health triage copilot for frontline (ASHA) health workers. "
    "Enter 6 basic vitals and get a risk level, a plain-language explanation, "
    "and referral guidance — or upload a whole village roster and see who to "
    "visit first."
)


@st.cache_resource
def load_model() -> TriageModel:
    return TriageModel.load(Path(__file__).resolve().parent)


model = load_model()
RISK_EMOJI = {"High": "🔴", "Mid": "🟡", "Low": "🟢"}
URGENCY_EMOJI = {"IMMEDIATE": "🚨", "URGENT": "🔴", "SOON": "🟡", "ROUTINE": "🟢"}


def show_urgency(result: dict) -> None:
    """Referral-urgency banner. Rule-based guidance (from the risk level +
    clinical red-flag thresholds) — labelled as such, never shown as a
    model output."""
    u = result["urgency"]
    text = (
        f"{URGENCY_EMOJI[u['tier']]} **{u['tier']} — {u['window']}**: "
        f"{u['action']}"
    )
    if u["escalated"]:
        text += f"\n\nEscalated because: {u['escalation_reason']}."
    if u["tier"] in ("IMMEDIATE", "URGENT"):
        st.error(text)
    elif u["tier"] == "SOON":
        st.warning(text)
    else:
        st.info(text)
    st.caption("Referral urgency is rule-based guidance, not a model output.")


def show_result(result: dict, name: str = "") -> None:
    level = result["risk_level"]
    st.subheader(f"{RISK_EMOJI[level]} {name + ' — ' if name else ''}{level} risk "
                 f"({result['confidence'] * 100:.0f}% confidence)")
    probs = result["probabilities"]
    st.write(
        f"Low: {probs['Low'] * 100:.0f}% · Mid: {probs['Mid'] * 100:.0f}% · "
        f"High: {probs['High'] * 100:.0f}%"
    )
    st.info(result["plain_summary"])
    if result["top_drivers"]:
        st.write("**Why this risk level:**")
        for d in result["top_drivers"]:
            st.write(f"- {d['sentence']}")
    show_urgency(result)


tab_single, tab_batch = st.tabs(["Single case", "Batch roster (CSV)"])

with tab_single:
    st.write(
        "No readings handy? Load a real example case from the dataset — "
        "one click fills all six vitals."
    )
    ex_cols = st.columns(3)
    for (key, case), col in zip(EXAMPLE_CASES.items(), ex_cols):
        with col:
            if st.button(case["label"], key=f"example_{key}",
                         use_container_width=True,
                         help=case["description"]):
                for f, v in case["vitals"].items():
                    st.session_state[f"vital_{f}"] = v
                st.session_state["example_loaded"] = case["label"]
                st.rerun()
    if st.session_state.get("example_loaded"):
        st.caption(f"Loaded: {st.session_state['example_loaded']} — "
                   "press Check risk to score it.")
    _defaults = {"Age": 25, "SystolicBP": 120, "DiastolicBP": 80,
                 "BS": 7.5, "BodyTemp": 98.0, "HeartRate": 76}
    for f, d in _defaults.items():
        st.session_state.setdefault(f"vital_{f}", d)
    with st.form("triage_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=10, max_value=70, value=st.session_state["vital_Age"], key="vital_Age")
            systolic = st.number_input("Systolic BP (mm Hg)", min_value=70, max_value=220, value=st.session_state["vital_SystolicBP"], key="vital_SystolicBP")
        with col2:
            diastolic = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=140, value=st.session_state["vital_DiastolicBP"], key="vital_DiastolicBP")
            bs = st.number_input("Blood sugar (mmol/L)", min_value=3.0, max_value=30.0, value=st.session_state["vital_BS"], key="vital_BS")
        with col3:
            temp = st.number_input("Body temperature (°F)", min_value=95.0, max_value=106.0, value=st.session_state["vital_BodyTemp"], key="vital_BodyTemp")
            hr = st.number_input("Heart rate (bpm)", min_value=40, max_value=140, value=st.session_state["vital_HeartRate"], key="vital_HeartRate")
        submitted = st.form_submit_button("Check risk")

    if submitted:
        vitals = {
            "Age": age, "SystolicBP": systolic, "DiastolicBP": diastolic,
            "BS": bs, "BodyTemp": temp, "HeartRate": hr,
        }
        result = model.predict(vitals)
        show_result(result)
        st.download_button(
            "Download this case as a printable summary (.txt)",
            build_printable_summary(
                [{"name": "", **result}],
                counts={result["risk_level"]: 1},
            ).encode(),
            file_name="ashacare-case-summary.txt",
            mime="text/plain",
        )

with tab_batch:
    st.write(
        "Upload a CSV with columns: `name` (optional), `Age`, `SystolicBP`, "
        "`DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`. The roster is scored and "
        "ranked — highest risk first."
    )
    uploaded = st.file_uploader("Roster CSV", type=["csv"])
    st.download_button(
        "Download the sample roster (8 mothers, 1 with a missing reading)",
        (Path(__file__).resolve().parent / "data" / "sample_roster.csv")
        .read_bytes(),
        file_name="sample_roster.csv",
        mime="text/csv",
        help="Real rows from the UCI dataset (names are illustrative). "
             "Upload it above to see the ranked priority list — including "
             "how one bad row is skipped and listed instead of breaking "
             "the whole roster.",
    )
    if uploaded is not None:
        df = pd.read_csv(io.BytesIO(uploaded.read()))
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            rows = []
            skipped = []
            scored_results = []  # API-shaped results, for the printable summary
            skipped_api = []
            for idx, row in df.iterrows():
                raw = {f: row[f] for f in FEATURES}
                # NaN (empty cell) counts as missing — pandas reads it as
                # float('nan'), so normalise it before validating.
                raw = {f: (None if pd.isna(v) else v) for f, v in raw.items()}
                name = "" if pd.isna(row.get("name", "")) else str(row.get("name", ""))
                vitals, errors = validate_vitals(raw)
                if errors:
                    skipped.append(f"Row {idx + 1}" + (f" ({name})" if name else "")
                                   + ": " + "; ".join(errors))
                    skipped_api.append({"index": int(idx), "name": name, "errors": errors})
                    continue
                r = model.predict(vitals)
                u = r["urgency"]
                scored_results.append({"name": name, **r})
                rows.append({
                    "Name": name,
                    "Risk": f"{RISK_EMOJI[r['risk_level']]} {r['risk_level']}",
                    "Urgency": f"{URGENCY_EMOJI[u['tier']]} {u['tier']} — {u['window']}",
                    "Confidence": round(r["confidence"], 2),
                    "P(High)": round(r["probabilities"]["High"], 2),
                    "Referral": r["referral"],
                    **{f: vitals[f] for f in FEATURES},
                })
            if skipped:
                st.warning(
                    f"⚠️ {len(skipped)} row(s) skipped — bad or missing vitals. "
                    "Fix these readings and re-upload; the rest of the roster "
                    "was still scored:\n\n" + "\n".join(f"- {s}" for s in skipped)
                )
            if not rows:
                st.error("No rows could be scored — every row had bad or "
                         "missing vitals (see the list above).")
            else:
                out = pd.DataFrame(rows)
                rank = {"High": 0, "Mid": 1, "Low": 2}
                out["_risk"] = out["Risk"].str.replace(r"^[^\s]+\s+", "", regex=True)
                out["_rank"] = out["_risk"].map(rank)
                out = out.sort_values(["_rank", "P(High)"], ascending=[True, False])
                out = out.drop(columns=["_rank", "_risk"])
                n_high = int((out["Risk"].str.endswith("High")).sum())
                n_mid = int((out["Risk"].str.endswith("Mid")).sum())
                n_low = int((out["Risk"].str.endswith("Low")).sum())
                n_imm = int(out["Urgency"].str.startswith("🚨").sum())
                st.write(
                    f"**{len(out)} mothers scored** — "
                    f"🚨 {n_imm} need emergency referral today · "
                    f"🔴 {n_high} high risk (refer within 24h) · "
                    f"🟡 {n_mid} mid risk (follow up within a week) · "
                    f"🟢 {n_low} low risk (routine)."
                )
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download priority list (CSV)",
                    out.to_csv(index=False).encode(),
                    file_name="ashacare-priority-list.csv",
                    mime="text/csv",
                )
                rank_order = {"High": 0, "Mid": 1, "Low": 2}
                ranked = sorted(
                    scored_results,
                    key=lambda r: (rank_order[r["risk_level"]],
                                   -r["probabilities"]["High"]),
                )
                st.download_button(
                    "Download printable visit summary (.txt)",
                    build_printable_summary(
                        ranked,
                        counts={"High": n_high, "Mid": n_mid, "Low": n_low},
                        skipped=skipped_api,
                    ).encode(),
                    file_name="ashacare-visit-summary.txt",
                    mime="text/plain",
                    help="Plain-text priority list for printing — works offline, "
                         "on paper, on a basic phone.",
                )

st.caption(
    "Screening aid only — not a diagnosis. High-risk cases must be referred "
    "to a doctor. Model: XGBoost trained on the UCI Maternal Health Risk "
    "dataset (1,014 cases)."
)
