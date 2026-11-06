"""
Generate a labeled features CSV so the pipeline can be trained & demoed
fully OFFLINE (no Gnani API key needed). This is a stand-in for real
hand-labeled clips; it exercises the whole train/predict path.

Usage:
    python scripts/make_synthetic_data.py --out data/features_synth.csv --n 300

Output row shape (matches features/extract.py):
    id, label, <22 feature columns>
Classes: bullish, bearish, neutral, hesitant, confident
"""
import argparse
import csv
import os
import random

FEATURE_KEYS = [
    'pitch_mean', 'pitch_std', 'pitch_range', 'energy_mean', 'energy_std',
    'activity_ratio', 'speech_rate_seg_per_sec', 'spectral_flux_mean',
    'duration_sec',
] + [f'mfcc_{i}' for i in range(13)]

LABELS = ['bullish', 'bearish', 'neutral', 'hesitant', 'confident']


def _base_row():
    row = {
        'pitch_mean': random.gauss(180, 30),
        'pitch_std': random.gauss(20, 8),
        'pitch_range': random.gauss(80, 30),
        'energy_mean': random.gauss(0.05, 0.02),
        'energy_std': random.gauss(0.02, 0.01),
        'activity_ratio': random.gauss(0.6, 0.15),
        'speech_rate_seg_per_sec': random.gauss(1.5, 0.5),
        'spectral_flux_mean': random.gauss(0.4, 0.15),
        'duration_sec': random.gauss(60, 20),
    }
    for i in range(13):
        row[f'mfcc_{i}'] = random.gauss(0, 20)
    return row


def _label_shift(row, label):
    """Bias features per class so the classifier can learn something."""
    # pitch variability up for emotional classes
    if label in ('bullish', 'confident'):
        row['pitch_std'] += random.gauss(8, 3)
        row['pitch_range'] += random.gauss(30, 10)
        row['energy_std'] += random.gauss(0.008, 0.003)
    elif label == 'bearish':
        row['pitch_std'] += random.gauss(6, 3)
        row['energy_mean'] -= random.gauss(0.01, 0.004)
    elif label == 'hesitant':
        row['speech_rate_seg_per_sec'] -= random.gauss(0.4, 0.15)
        row['activity_ratio'] -= random.gauss(0.15, 0.05)
    elif label == 'neutral':
        row['pitch_std'] -= random.gauss(4, 2)
    # confidence -> faster, steadier
    if label == 'confident':
        row['speech_rate_seg_per_sec'] += random.gauss(0.3, 0.1)
        row['activity_ratio'] += random.gauss(0.1, 0.04)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/features_synth.csv')
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for i in range(args.n):
        label = random.choice(LABELS)
        row = _base_row()
        row['id'] = f'synth_{i:04d}'
        row['label'] = label
        rows.append(_label_shift(row, label))

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['id', 'label'] + FEATURE_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    print(f'Wrote {len(rows)} rows -> {out_path}')
    print('Class counts:', counts)


if __name__ == '__main__':
    main()
