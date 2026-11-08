# -*- coding: utf-8 -*-
# Train a speech-emotion model on CREMA-D (speaker-independent split) and
# evaluate on RAVDESS (cross-corpus). Writes static/eval/summary.json.
import os, sys, csv, json, datetime, argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import feature_names

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crema", default="data/crema_subset.csv")
    ap.add_argument("--ravdess", default="data/ravdess_features.csv")
    ap.add_argument("--summary", default="static/eval/summary.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    np.random.seed(args.seed)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys = feature_names()
    # ---- load CREMA-D ----
    crema = load_csv(os.path.join(base, args.crema))
    print("CREMA-D rows", len(crema))
    X = np.array([[float(r[k]) for k in keys] for r in crema])
    y = np.array([r["label"] for r in crema])
    actors = np.array([r["actor"] for r in crema])
    # ---- speaker-independent split (by actor) ----
    uniq = np.unique(actors)
    np.random.shuffle(uniq)
    n_test_actors = max(1, int(round(len(uniq) * 0.25)))
    test_actors = set(uniq[:n_test_actors].tolist())
    train_actors = set(uniq[n_test_actors:].tolist())
    train_mask = np.array([a in train_actors for a in actors])
    test_mask = np.array([a in test_actors for a in actors])
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print("train actors", len(train_actors), "test actors", len(test_actors))
    print("train clips", X_tr.shape[0], "test clips", X_te.shape[0])
    # stricter guarantee: no actor in both
    assert train_actors.isdisjoint(test_actors), "actor overlap!"
    # ---- majority baseline & chance ----
    classes = sorted(np.unique(y_tr))
    majority_cls = max(classes, key=lambda c: int(np.sum(y_tr == c)))
    majority_baseline = float(np.mean(y_te == majority_cls))
    chance = 1.0 / len(classes)
    # ---- train ----
    clf = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    labels = sorted(set(y_te) | set(y_pred))
    prec, rec, f1z, supp = precision_recall_fscore_support(y_te, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1z[i]),
            "support": int(supp[i]),
        }
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    print()
    print("========== CREMA-D test (speaker-independent) ==========")
    print("accuracy", round(acc,4))
    print("macro_f1", round(macro_f1,4))
    print("majority_baseline", round(majority_baseline,4))
    print("chance", round(chance,4))
    print(classification_report(y_te, y_pred, zero_division=0))
    print("confusion labels", labels)
    print("confusion matrix"); print(cm)
    # ---- cross-corpus RAVDESS ----
    rav = load_csv(os.path.join(base, args.ravdess))
    cross = None
    if rav:
        Xr = np.array([[float(r[k]) for k in keys] for r in rav])
        yr = np.array([r["label"] for r in rav])
        yr_pred = clf.predict(Xr)
        cross_acc = float(accuracy_score(yr, yr_pred))
        cross = {
            "train": "CREMA-D",
            "test": "RAVDESS",
            "accuracy": cross_acc,
            "n": int(len(yr)),
        }
        print()
        print("========== Cross-corpus RAVDESS (no retraining) ==========")
        print("n", len(yr), "accuracy", round(cross_acc,4))
        print(classification_report(yr, yr_pred, zero_division=0))
    # ---- write summary.json ----
    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasets": [
            {"name": "CREMA-D", "clips_used": int(len(crema)),
             "speakers": int(len(uniq)), "role": "train"},
            {"name": "RAVDESS", "clips_used": int(len(rav)) if rav else 0,
             "speakers": int(len(set(r["actor"] for r in rav))) if rav else 0,
             "role": "cross-test"},
        ],
        "split": {
            "method": "speaker-independent (by actor)",
            "train_clips": int(X_tr.shape[0]),
            "test_clips": int(X_te.shape[0]),
            "train_speakers": int(len(train_actors)),
            "test_speakers": int(len(test_actors)),
        },
        "model": {
            "type": "RandomForestClassifier",
            "features": keys,
            "trained_on": "CREMA-D",
            "params": {"n_estimators": args.trees, "random_state": args.seed},
        },
        "results": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "majority_baseline": majority_baseline,
            "chance": chance,
            "per_class": per_class,
            "confusion": {"labels": labels, "matrix": cm.tolist()},
        },
        "cross_corpus": cross,
        "human_check": None,
        "limitations": [
            "RAVDESS calm and surprise clips excluded (no CREMA-D counterpart).",
            "Studio-recorded acted speech; real-world settings will score lower.",
            "Acoustic features only; no lexical/semantic information.",
        ],
    }
    summary_path = os.path.join(base, args.summary)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", summary_path)

if __name__ == "__main__":
    main()
