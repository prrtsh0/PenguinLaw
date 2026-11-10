#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop in a manually-downloaded public-figure audio as a homepage example clip.

Usage:
    python scripts/add_public_clip.py <audio_file> <clip_id> "<Name>" "<topic>"

Example:
    python scripts/add_public_clip.py ~/Downloads/powell.mp3 clip4 "Jerome Powell" "FOMC press conference"

What it does:
  1. Converts/copies the audio to data/demo_clips/<clip_id>.wav (16k mono wav).
  2. Runs the full pipeline (transcribe -> voice emotion -> tone setter) to write
     data/results/<clip_id>.json.
  3. Registers a friendly name/topic so the UI shows it.

After that, the web app auto-discovers the clip (available_clips() scans
data/results/clipN.json) — just reload the homepage. Use "Live" mode or
re-run this script to refresh the analysis.
"""
import os
import sys
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DEMO_DIR = os.path.join(BASE, 'data', 'demo_clips')
RESULTS_DIR = os.path.join(BASE, 'data', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', help='path to downloaded audio (mp3/wav/ogg/…)')
    ap.add_argument('clip_id', help='e.g. clip4')
    ap.add_argument('--name', default=None, help='display name (e.g. "Jerome Powell")')
    ap.add_argument('--topic', default='Market mood', help='topic line')
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        sys.exit(f'audio not found: {args.audio}')

    # 1. normalise to a wav in demo_clips
    os.makedirs(DEMO_DIR, exist_ok=True)
    wav = os.path.join(DEMO_DIR, f'{args.clip_id}.wav')
    try:
        import soundfile as sf
        import numpy as np
        y, sr = sf.read(args.audio)
        if sr != 16000:
            import scipy.signal
            y = scipy.signal.resample_poly(y, 16000, sr)
            sr = 16000
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(wav, y, sr)
        print('wrote audio ->', wav)
    except Exception as e:
        print('audio conversion failed, falling back to copy:', e)
        import shutil
        shutil.copy2(args.audio, wav)

    # 2. run the full pipeline to produce the cached result
    from scripts.analyze_clip import analyze, CLIPS
    if args.clip_id not in CLIPS:
        # temporarily register so analyze() accepts the id; reload module dict
        CLIPS[args.clip_id] = wav
    result = analyze(args.clip_id, force_live=True, use_cache=False)
    print('analysis written ->', os.path.join(RESULTS_DIR, f'{args.clip_id}.json'))

    # 3. friendly name/topic
    from scripts import web_app
    web_app.SPECIAL_CLIP_NAME[args.clip_id] = (
        args.name or ('Clip ' + args.clip_id[len('clip'):]),
        args.topic,
    )
    print('done. reload the homepage — new clip appears in the library.')


if __name__ == '__main__':
    main()
