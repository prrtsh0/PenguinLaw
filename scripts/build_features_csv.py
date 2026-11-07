"""
Build a features CSV from labeled audio files (real data path).

Takes the labeling template (data/labels_template.csv: audio_path,label)
and produces the feature CSV that train_tone.py consumes
(columns: id, label, <22 features>).

Usage:
    python scripts/build_features_csv.py --labels data/labels_template.csv \
        --out data/features_real.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import extract_features, feature_names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True,
                    help='CSV with columns: audio_path,label[,notes]')
    ap.add_argument('--out', default='data/features_real.csv')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_path = os.path.join(base, args.labels)
    out_path = os.path.join(base, args.out)

    with open(labels_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        sys.exit('No labeled rows found.')

    keys = feature_names()
    out_rows = []
    skipped = []
    for r in rows:
        audio = r.get('audio_path', '').strip()
        label = (r.get('label') or '').strip()
        if not audio or not label:
            skipped.append((audio, 'missing audio_path or label'))
            continue
        full = audio if os.path.isabs(audio) else os.path.join(base, audio)
        if not os.path.exists(full):
            skipped.append((audio, 'file not found'))
            continue
        try:
            feats = extract_features(full)
        except Exception as e:
            skipped.append((audio, f'extract failed: {e}'))
            continue
        # ensure all feature keys present (0-fill any missing)
        row = {'id': os.path.basename(audio), 'label': label}
        for k in keys:
            row[k] = feats.get(k, 0.0)
        out_rows.append(row)

    if not out_rows:
        sys.exit('No features could be extracted.')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['id', 'label'] + keys)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f'Wrote {len(out_rows)} feature rows -> {out_path}')
    counts = {}
    for r in out_rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    print('Label counts:', counts)
    if skipped:
        print('\nSkipped:')
        for a, why in skipped:
            print(f'  {a}: {why}')


if __name__ == '__main__':
    main()
