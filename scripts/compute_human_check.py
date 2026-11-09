# -*- coding: utf-8 -*-
# Read the human-filled labels, compute agreement vs the model, and write
# human_check into static/eval/summary.json.
import os, sys, csv, json, datetime, argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/human_check_clips.csv")
    ap.add_argument("--summary", default="static/eval/summary.json")
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_path = os.path.join(base, args.labels)
    summary_path = os.path.join(base, args.summary)

    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    missing = [r["id"] for r in rows if not (r.get("human_label") or "").strip()]
    if missing:
        print(f"ERROR: {len(missing)} clip(s) still missing human_label:")
        for m in missing:
            print("   ", m)
        sys.exit(1)

    valid = {"ANG","DIS","FEA","HAP","NEU","SAD"}
    bad = [r["id"] for r in rows if (r["human_label"].strip().upper() not in valid)]
    if bad:
        print("ERROR: invalid label(s):", bad)
        sys.exit(1)

    y_model = [r["model_prediction"] for r in rows]
    y_human = [r["human_label"].strip().upper() for r in rows]
    n = len(rows)
    acc = float(accuracy_score(y_human, y_model))
    macro = float(f1_score(y_human, y_model, average="macro", zero_division=0))
    labels = sorted(set(y_human) | set(y_model))
    cm = confusion_matrix(y_human, y_model, labels=labels).tolist()

    human_check = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_clips": n,
        "agreement_accuracy": acc,
        "agreement_macro_f1": macro,
        "per_class": {},
        "confusion": {"labels": labels, "matrix": cm},
        "sources": ["CREMA-D test clips (listened to by a person)"],
    }
    for lbl in labels:
        tp = sum(1 for h, m in zip(y_human, y_model) if h == m == lbl)
        n_pred = sum(1 for m in y_model if m == lbl)
        n_gt = sum(1 for h in y_human if h == lbl)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        human_check["per_class"][lbl] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(n_gt),
        }

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["human_check"] = human_check
    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"human_check written to {summary_path}")
    print(f"n clips: {n}")
    print(f"agreement accuracy: {acc:.4f}")
    print(f"agreement macro F1: {macro:.4f}")
    print(f"confusion ({len(labels)}):")
    print("labels:", labels)
    print("matrix:")
    for row in cm:
        print("  ", row)

if __name__ == "__main__":
    main()
