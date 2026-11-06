"""
Train the tone classifier from a labeled features CSV.

Supports both synthetic (offline) and real hand-labeled data.
Saves a versioned model (joblib) + metrics CSV for drift/shadow comparison.

Usage:
    python scripts/train_tone.py --data data/features_synth.csv --out models/tone_model_v1.joblib
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

FEATURE_KEYS = [
    'pitch_mean', 'pitch_std', 'pitch_range', 'energy_mean', 'energy_std',
    'activity_ratio', 'speech_rate_seg_per_sec', 'spectral_flux_mean',
    'duration_sec',
] + [f'mfcc_{i}' for i in range(13)]


def load_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if not r.get('label'):
                continue
            try:
                feat = [float(r[k]) for k in FEATURE_KEYS]
            except (KeyError, ValueError) as e:
                print(f'  skip row {r.get("id")}: {e}')
                continue
            rows.append((feat, r['label'].strip()))
    if not rows:
        sys.exit('No valid labeled rows found.')
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='models/tone_model_v1.joblib')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--trees', type=int, default=200)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, args.data)
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=args.trees, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    report = classification_report(y_test, clf.predict(X_test), zero_division=0)

    # versioned bundle
    bundle = {
        'model': clf,
        'features': FEATURE_KEYS,
        'classes': sorted(set(y.tolist())),
        'train_date': datetime.now(timezone.utc).isoformat(),
        'data_source': os.path.basename(data_path),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'test_accuracy': round(float(acc), 4),
        'git_flavor': 'tone-model',
    }
    joblib.dump(bundle, out_path)

    # metrics CSV for drift/shadow comparison
    metrics_path = out_path.replace('.joblib', '_metrics.csv')
    with open(metrics_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'train_date', 'n_train', 'n_test',
                    'test_accuracy', 'report'])
        w.writerow([os.path.basename(out_path), bundle['train_date'],
                    bundle['n_train'], bundle['n_test'],
                    bundle['test_accuracy'], report.replace('\n', ' | ')])

    print(f'Saved model -> {out_path}')
    print(f'Test accuracy: {acc:.3f}')
    print(report)


if __name__ == '__main__':
    main()
