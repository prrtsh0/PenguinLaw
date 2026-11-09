# -*- coding: utf-8 -*-
# Independent automated cross-check: run a large pretrained speech-emotion
# model (SUPERB wav2vec2) on the human-check clips and compare its predictions
# to our acoustic-feature Random Forest. Records as model_cross_check
# (distinct from human_check). No human listening involved.
import os, sys, csv, json, datetime, argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

P2M = {"ang": "ANG", "hap": "HAP", "neu": "NEU", "sad": "SAD"}
OVERLAP = {"ANG", "HAP", "NEU", "SAD"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/human_check_clips.csv")
    ap.add_argument("--summary", default="static/eval/summary.json")
    ap.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_path = os.path.join(base, args.labels)
    summary_path = os.path.join(base, args.summary)

    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # keep rows whose ground-truth emotion (in filename) overlaps SUPERB's classes
    selected = []
    for r in rows:
        fn = os.path.basename(r["id"]).replace(".wav", "")
        parts = fn.split("_")
        gt = parts[2].upper() if len(parts) >= 3 else ""
        if gt in OVERLAP:
            selected.append((r, gt))
    # note: if a human label exists, use it; otherwise fall back to filename GT
    # For an independent automated check we compare SUPERB vs our model on the
    # same clips; we report agreement and also SUPERB's accuracy vs filename GT.

    import torch
    import librosa
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

    print("loading model:", args.model)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model.eval()

    def predict_superb(path):
        y, sr = librosa.load(path, sr=16000)
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(torch.argmax(logits, dim=1).item())
        label = model.config.id2label.get(idx, "?")
        return P2M.get(label, label)

    supers = []
    our_preds = []
    gt_list = []
    n_skipped_ours = 0
    for r, gt in selected:
        path = r["clip_path"]
        if not os.path.exists(path):
            continue
        sup = predict_superb(path)
        our = r["model_prediction"]
        if our not in OVERLAP:
            n_skipped_ours += 1
            continue
        supers.append(sup)
        our_preds.append(our)
        gt_list.append(gt)

    n = len(supers)
    print(f"clips evaluated: {n} (overlap emotions ANG/HAP/NEU/SAD)")
    print(f"  skipped (our model predicted DIS/FEA, not in SUPERB classes): {n_skipped_ours}")

    # agreement between our model and SUPERB
    acc_ours_vs_superb = float(accuracy_score(our_preds, supers))
    macro_ours_vs_superb = float(f1_score(our_preds, supers, average="macro", zero_division=0))
    labels3 = sorted(set(our_preds) | set(supers))
    cm = confusion_matrix(our_preds, supers, labels=labels3).tolist()

    # SUPERB accuracy vs ground-truth filename label
    acc_superb_gt = float(accuracy_score(gt_list, supers))

    # our model accuracy vs ground-truth filename label (on this subset)
    acc_ours_gt = float(accuracy_score(gt_list, our_preds))

    cross = {
        "kind": "model_cross_check",
        "reference_model": args.model,
        "n_clips": n,
        "our_model_vs_reference_agreement": acc_ours_vs_superb,
        "our_model_vs_reference_macro_f1": macro_ours_vs_superb,
        "reference_model_accuracy_vs_filename_gt": acc_superb_gt,
        "our_model_accuracy_vs_filename_gt": acc_ours_gt,
        "confusion_ours_vs_reference": {"labels": labels3, "matrix": cm},
        "note": "Automated model-vs-model cross-check; NOT a human check. "
                 "Limited to the 4 overlapping emotions: ANG/HAP/NEU/SAD.",
    }

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["model_cross_check"] = cross
    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== model cross-check written to summary.json ===")
    print(f"n clips: {n}")
    print(f"our model vs SUPERB agreement: {acc_ours_vs_superb:.4f}")
    print(f"our model vs SUPERB macro F1: {macro_ours_vs_superb:.4f}")
    print(f"SUPERB accuracy vs filename GT: {acc_superb_gt:.4f}")
    print(f"our model accuracy vs filename GT (subset): {acc_ours_gt:.4f}")
    print("confusion (rows=our model, cols=SUPERB):")
    print("labels:", labels3)
    for row in cm:
        print("  ", row)

if __name__ == "__main__":
    main()
