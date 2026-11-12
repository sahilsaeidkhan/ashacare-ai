"""Printable visit summary for a scored roster.

An ASHA worker may have no signal in the village and a basic phone, so the
priority list must also work on paper: this builds a plain-text report from
the scored cases (the same dicts `POST /triage/batch` returns in
`priority_list`) — counts by urgency, the ranked list with one plain-language
line per mother, and the skipped rows. Pure function, no I/O, so the
dashboard and the tests can use it directly.
"""

from __future__ import annotations

from datetime import date


def build_printable_summary(
    scored: list[dict],
    counts: dict[str, int] | None = None,
    skipped: list[dict] | None = None,
    date_str: str | None = None,
) -> str:
    """Render the ranked priority list as plain text.

    `scored` is the risk-ranked list (High first), each entry a predict()
    result with an optional "name". `counts` maps Low/Mid/High to totals;
    `skipped` lists the rows that could not be scored, with reasons.
    """
    counts = counts or {
        level: sum(1 for r in scored if r["risk_level"] == level)
        for level in ("High", "Mid", "Low")
    }
    skipped = skipped or []
    n_emergency = sum(
        1 for r in scored if r["urgency"]["tier"] == "IMMEDIATE"
    )

    lines = [
        "AshaCare AI — Village visit priority list",
        f"Date: {date_str or date.today().isoformat()}",
        "",
        f"Scored: {len(scored)} mothers — {n_emergency} need emergency "
        f"referral TODAY, {counts.get('High', 0)} high risk (refer within "
        f"24 hours), {counts.get('Mid', 0)} mid risk (follow up within "
        f"7 days), {counts.get('Low', 0)} low risk (routine).",
    ]
    if skipped:
        lines.append(
            f"Skipped: {len(skipped)} row(s) with bad or missing vitals — "
            "re-check those readings and re-upload; they were not scored."
        )
        for s in skipped:
            name = f" ({s['name']})" if s.get("name") else ""
            lines.append(
                f"  - Row {s['index'] + 1}{name}: "
                + "; ".join(s["errors"])
            )
    lines.append("")
    lines.append("Visit in this order (highest risk first):")
    for i, r in enumerate(scored, 1):
        name = r.get("name") or f"Case {i}"
        u = r["urgency"]
        lines.append(
            f"{i}. {name} — {r['risk_level']} risk, "
            f"{u['tier']} ({u['window']})"
        )
        lines.append(f"   Action: {u['action']}")
        drivers = r.get("top_drivers") or []
        if drivers:
            why = "; ".join(d.get("sentence", d["label"]) for d in drivers)
            lines.append(f"   Why: {why[0].lower() + why[1:]}")
        lines.append("")
    lines.append("Screening aid only — not a diagnosis. High-risk cases must "
                 "be referred to a doctor.")
    lines.append(
        "AshaCare AI — GenHacks 2026. Model: XGBoost trained on the UCI "
        "Maternal Health Risk dataset (1,014 cases)."
    )
    return "\n".join(lines)
