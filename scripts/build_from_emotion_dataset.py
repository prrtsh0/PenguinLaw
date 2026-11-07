"""
Build a real training set from a public pre-labeled emotion speech dataset.

Sources:
  - CREMA-D (GitHub, Git LFS): 7,442 clips, 6 emotions (ANG/DIS/FEA/HAP/NEU/SAD)
  - RAVDESS (Zenodo): ~1,440 clips, 8 emotions (01..08)

Each clip already has an emotion label. We map emotion -> market tone via a
documented rubric, extract the 22 prosody features, and write the feature CSV
that train_tone.py consumes. Optionally trains the model immediately.

Usage examples:
    # auto-download a balanced subset of CREMA-D and train
    python scripts/build_from_emotion_dataset.py --source crema --n 150 \
        --train --out models/tone_model_v2.joblib

    # use a locally-downloaded CREMA-D audio folder
    python scripts/build_from_emotion_dataset.py --source local \
        --local_dir data/crema_audio --n 200 --out data/features_real.csv
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import extract_features, feature_names  # noqa: E402

# --- emotion -> market tone rubric ---------------------------------------
CREMA_EMOTION_TONE = {
    'ANG': 'confident',   # anger = assertive/aggressive
    'DIS': 'bearish',     # disgust = risk-off
    'FEA': 'hesitant',    # fear = hesitant/uncertain
    'HAP': 'bullish',     # happy = optimistic
    'NEU': 'neutral',     # neutral
    'SAD': 'bearish',     # sad = risk-off
}
RAVDESS_EMOTION_TONE = {
    '01': 'neutral',      # neutral
    '02': 'neutral',      # calm
    '03': 'bullish',      # happy
    '04': 'bearish',      # sad
    '05': 'confident',    # angry
    '06': 'hesitant',     # fearful
    '07': 'bearish',      # disgust
    '08': 'bullish',      # surprised
}

CREMA_BASE = ('https://media.githubusercontent.com/media/'
              'CheyneyComputerScience/CREMA-D/master/')
CREMA_TREE_API = ('https://api.github.com/repos/'
                  'CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1')


def crema_emotion(filename):
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


def ravdess_emotion(filename):
    parts = os.path.basename(filename).replace('.wav', '').split('-')
    if len(parts) >= 3:
        return parts[2]
    return None


def tone_for(source, emotion):
    table = CREMA_EMOTION_TONE if source == 'crema' else RAVDESS_EMOTION_TONE
    return table.get(emotion)


def list_crema_files():
    """Fetch all AudioWAV/*.wav paths from the GitHub tree API."""
    with urllib.request.urlopen(CREMA_TREE_API, timeout=30) as res:
        data = json.loads(res.read().decode())
    paths = [t['path'] for t in data.get('tree', [])
             if t['path'].startswith('AudioWAV/') and t['path'].endswith('.wav')]
    return paths


def download(url, dest):
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['crema', 'ravdess', 'local'],
                    required=True)
    ap.add_argument('--n', type=int, default=150,
                    help='number of clips to use')
    ap.add_argument('--local_dir', default='data/crema_audio',
                    help='for --source local: folder of wav files')
    ap.add_argument('--out', default='data/features_real.csv')
    ap.add_argument('--train', action='store_true',
                    help='also train the model after building features')
    ap.add_argument('--model-out', default='models/tone_model_v2.joblib')
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(base, 'data', 'audio', 'emotion')
    os.makedirs(audio_dir, exist_ok=True)

    # ---- stage 1: gather (audio_path, emotion) candidates ----------------
    candidates = []
    if args.source == 'local':
        for fn in os.listdir(args.local_dir):
            if fn.lower().endswith('.wav'):
                emotion = crema_emotion(fn) or ravdess_emotion(fn)
                if emotion:
                    candidates.append((os.path.join(args.local_dir, fn), emotion))
    elif args.source == 'crema':
        paths = list_crema_files()
        for p in paths:
            emotion = crema_emotion(p)
            if emotion and emotion in CREMA_EMOTION_TONE:
                candidates.append((p, emotion))
    elif args.source == 'ravdess':
        # RAVDESS single-file zip URL (actor 01..24); user can point --local_dir
        # at the extracted folder instead; here we support a local dir + naming.
        print('RAVDESS: use --source local --local_dir <extracted folder>')
        sys.exit(1)

    if not candidates:
        sys.exit('No candidate audio files found.')

    # ---- stage 2: balance by tone class ----------------------------------
    by_tone = {}
    for path, emotion in candidates:
        tone = tone_for(args.source, emotion)
        if tone:
            by_tone.setdefault(tone, []).append((path, emotion, tone))
    # sample evenly across tone classes up to --n total
    picked = []
    tones = sorted(by_tone)
    per_class = max(1, args.n // len(tones))
    for tone in tones:
        pool = by_tone[tone]
        random.shuffle(pool)
        picked.extend(pool[:per_class])
    if len(picked) > args.n:
        random.shuffle(picked)
        picked = picked[:args.n]
    print(f'Picked {len(picked)} clips:',
          {t: sum(1 for _, _, tt in picked if tt == t) for t in set(t for _, _, t in picked)})

    # ---- stage 3: download (if remote) + extract features ----------------
    keys = feature_names()
    out_rows = []
    skipped = []
    for path, emotion, tone in picked:
        local = path
        if not os.path.exists(local):
            dest = os.path.join(audio_dir, os.path.basename(path))
            try:
                download(CREMA_BASE + urllib.parse.quote(path), dest)
                local = dest
            except Exception as e:
                skipped.append((path, f'download failed: {e}'))
                continue
        try:
            feats = extract_features(local)
        except Exception as e:
            skipped.append((path, f'extract failed: {e}'))
            continue
        row = {'id': os.path.basename(path), 'label': tone}
        for k in keys:
            row[k] = feats.get(k, 0.0)
        out_rows.append(row)

    if not out_rows:
        sys.exit('No features extracted.')
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['id', 'label'] + keys)
        writer.writeheader()
        writer.writerows(out_rows)
    counts = {}
    for r in out_rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    print(f'Wrote {len(out_rows)} feature rows -> {args.out}')
    print('Label counts:', counts)
    if skipped:
        print(f'Skipped {len(skipped)}:')
        for a, w in skipped[:10]:
            print(f'  {a}: {w}')

    if args.train:
        # make test split viable for small datasets (stratified needs
        # test_size * n >= n_classes)
        n = len(out_rows)
        n_classes = len(set(r['label'] for r in out_rows))
        test_size = max(0.2, round(n_classes * 1.5 / n, 3))
        test_size = min(0.5, test_size)
        print('\nTraining...')
        subprocess.run([sys.executable,
                        os.path.join(base, 'scripts', 'train_tone.py'),
                        '--data', args.out, '--out', args.model_out,
                        '--test-size', str(test_size)],
                       check=True)


if __name__ == '__main__':
    main()
