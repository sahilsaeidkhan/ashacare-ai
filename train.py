"""Train the AshaCare AI XGBoost classifier on the UCI Maternal Health Risk
dataset and save model.joblib + model_meta.json.

Run:  python train.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Maternal Health Risk Data Set.csv"
DATASET_SOURCE = "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
CLASSES = ["low risk", "mid risk", "high risk"]


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    assert list(df.columns) == FEATURES + [TARGET], f"unexpected columns: {df.columns.tolist()}"
    assert len(df) == 1014, f"expected 1014 rows, got {len(df)}"

    print("=== EDA ===")
    print(f"Rows: {len(df)}, columns: {len(df.columns)}, missing values: {int(df.isnull().sum().sum())}")
    print("Class balance:")
    for cls in CLASSES:
        n = int((df[TARGET] == cls).sum())
        print(f"  {cls}: {n} ({100 * n / len(df):.1f}%)")
    print("Feature ranges:")
    for f in FEATURES:
        print(f"  {f}: {df[f].min()} – {df[f].max()} (median {df[f].median()})")

    X = df[FEATURES].astype(float)
    # XGBoost's sklearn API needs integer labels; encode explicitly so the
    # column order of predict_proba matches CLASSES (low=0, mid=1, high=2).
    label_to_int = {c: i for i, c in enumerate(CLASSES)}
    y = df[TARGET].map(label_to_int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n=== Evaluation (held-out test, n=203) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Per-class precision / recall / F1:")
    per_class = {}
    for cls in CLASSES:
        r = report[cls]
        per_class[cls] = {
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        }
        print(f"  {cls}: P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")
    print("Confusion matrix (rows = true, cols = predicted; low/mid/high):")
    print(cm)

    joblib.dump(model, BASE / "model.joblib")
    meta = {
        "model": "XGBoost (n_estimators=200, max_depth=4, lr=0.1)",
        "features": FEATURES,
        "classes": CLASSES,
        "feature_medians": [float(X[f].median()) for f in FEATURES],
        "dataset_source": DATASET_SOURCE,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        },
        "trained_on": date.today().isoformat(),
    }
    (BASE / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model.joblib + model_meta.json (accuracy {acc:.4f})")
    return meta


if __name__ == "__main__":
    main()
