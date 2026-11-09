# -*- coding: utf-8 -*-
"""Stratified train/test split benchmark for the tone classifier.

Runs an honest stratified random split and writes static/benchmarks_tone.json
(consumed by the Benchmarks page). Reports accuracy, macro-F1, majority
baseline, chance, and a per-class table for each dataset.

Usage:
    python scripts/bench_tone.py
"""
import csv
import datetime
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support,
                             classification_report)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_tone import FEATURE_KEYS  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("label"):
                continue
            try:
                feat = [float(r[k]) for k in FEATURE_KEYS]
            except (KeyError, ValueError):
                continue
            rows.append((feat, r["label"].strip()))
    if not rows:
        return None, None
    X = np.array([x[0] for x in rows])
    y = np.array([x[1] for x in rows])
    return X, y


def eval_split(name, path, test_size=0.25, trees=200):
    try:
        X, y = load_rows(path)
        if X is None:
            return {"dataset": name, "error": "no valid labeled rows"}
        if len(np.unique(y)) < 2:
            return {"dataset": name, "error": "fewer than 2 classes"}
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        clf = RandomForestClassifier(n_estimators=trees, random_state=42,
                                     n_jobs=-1)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        acc = accuracy_score(yte, yp)
        macro_f1 = f1_score(yte, yp, average="macro", zero_division=0)
        labels = sorted(set(y.tolist()))
        p, r, f, s = precision_recall_fscore_support(yte, yp, labels=labels,
                                                     zero_division=0)
        per_class = {}
        for idx, lab in enumerate(labels):
            per_class[lab] = {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f[idx]), 4),
                "support": int(s[idx]),
            }
        counts = np.bincount([labels.index(v) for v in yte],
                             minlength=len(labels))
        majority = float(counts.max()) / len(yte) if len(yte) else 0.0
        return {
            "dataset": name,
            "model": f"RandomForest({trees})",
            "rows_total": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "majority_baseline": round(majority, 4),
            "chance": round(1.0 / len(labels), 4),
            "per_class": per_class,
            "report": classification_report(yte, yp, labels=labels,
                                            zero_division=0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}


def main():
    out = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "note": ("Stratified random train/test split. Synthetic tone features "
                 "are auto-generated; the real set is a small hand-labeled "
                 "corpus (n=20). Treat these as relative signals, not "
                 "deployment guarantees."),
        "datasets": [
            eval_split("synthetic (features_synth.csv)",
                       os.path.join(BASE, "data", "features_synth.csv")),
            eval_split("real (features_real.csv)",
                       os.path.join(BASE, "data", "features_real.csv")),
        ],
    }
    dest = os.path.join(BASE, "static", "benchmarks_tone.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
