#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker-independent cross-validation benchmark for the CREMA-D RandomForest.

Trains the *same* RandomForest used by scripts/analyze_behavior.py on
data/crema_subset.csv but evaluates on held-out **speakers** (GroupKFold by
actor) to measure how the model generalizes to unseen actors.

Usage:
    python eval/baseline_cv.py
"""
import os
import sys
import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "crema_subset.csv")
METADATA = {"id", "actor", "label"}


def load_rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = load_rows()
    keys = [k for k in rows[0].keys() if k not in METADATA]
    X = np.array([[float(r[k]) for k in keys] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["actor"] for r in rows])

    gkf = GroupKFold(n_splits=5)
    accs = []
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = accuracy_score(y[te], pred)
        accs.append(acc)
        y_true_all += list(y[te])
        y_pred_all += list(pred)
        print(f"fold {fold}: n_train={len(tr)} n_test={len(te)} acc={acc:.4f}")

    print("\n=== Baseline (speaker-independent) CV ===")
    print(f"mean accuracy across folds : {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"pooled accuracy            : {accuracy_score(y_true_all, y_pred_all):.4f}")
    print("\nPer-emotion report:\n" + classification_report(y_true_all, y_pred_all))


if __name__ == "__main__":
    main()
