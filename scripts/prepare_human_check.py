# -*- coding: utf-8 -*-
# Prepare a human-check set: pick N test clips, run the trained model on each,
# and write a CSV the human fills in with their emotion labels.
import os, sys, csv, argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import feature_names

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crema", default="data/crema_subset.csv")
    ap.add_argument("--audio_dir", default="data/crema_audio")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="data/human_check_clips.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    np.random.seed(args.seed)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys = feature_names()

    crema = load_csv(os.path.join(base, args.crema))
    X = np.array([[float(r[k]) for k in keys] for r in crema])
    y = np.array([r["label"] for r in crema])
    actors = np.array([r["actor"] for r in crema])

    # same speaker-independent split as training
    uniq = np.unique(actors)
    np.random.shuffle(uniq)
    n_test_actors = max(1, int(round(len(uniq) * 0.25)))
    test_actors = set(uniq[:n_test_actors].tolist())
    test_mask = np.array([a in test_actors for a in actors])

    X_tr, y_tr = X[~test_mask], y[~test_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    crema_te = [crema[i] for i in range(len(crema)) if test_mask[i]]

    clf = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)

    # pick a deterministic spread across clips
    idx = np.arange(len(crema_te))
    rng = np.random.RandomState(args.seed + 1)
    rng.shuffle(idx)
    picked = idx[: args.n]

    audio_dir = os.path.join(base, args.audio_dir)
    out_path = os.path.join(base, args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "clip_path", "model_prediction", "human_label"])
        for i in picked:
            r = crema_te[i]
            clip_path = os.path.join(audio_dir, r["id"])
            w.writerow([r["id"], clip_path, preds[i], ""])

    print(f"wrote {out_path} with {len(picked)} clips")
    print()
    print("HOW TO USE")
    print("  1. Open each clip (any media player). File paths are in the CSV.")
    print("  2. For each row, set human_label to one of: ANG DIS FEA HAP NEU SAD")
    print("  3. Save the CSV, then run: python scripts/compute_human_check.py")
    print()
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(f"  {row['id']:40s} model={row['model_prediction']:4s} -> human_label")

if __name__ == "__main__":
    main()
