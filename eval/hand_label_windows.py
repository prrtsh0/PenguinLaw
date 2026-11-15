#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a hand-labelling sheet for every voice-emotion window.

By default this now includes ALL windows across the real podcast clips (10 per
clip x 5 clips = 50). If a sheet already exists, any previously filled
`human_label` values are carried over for matching (clip, start, end) rows.

Usage:
    python eval/hand_label_windows.py [--clips clip1 clip2 ...]
"""
import argparse
import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'data', 'results')
OUT = os.path.join(RESULTS, '_hand_label_sheet.csv')

DEFAULT_CLIPS = ['clip1', 'clip2', 'clip3', 'clip4', 'clip5']


def top_emotion(w):
    probs = w.get('probs') or {}
    if not probs:
        return ''
    return max(probs, key=probs.get)


def load_previous():
    old = {}
    if os.path.exists(OUT):
        with open(OUT, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                key = (r.get('clip'), r.get('start_sec'), r.get('end_sec'))
                old[key] = (r.get('human_label') or '').strip()
    return old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='*', default=DEFAULT_CLIPS)
    args = ap.parse_args()

    prev = load_previous()
    rows = []
    for cid in args.clips:
        path = os.path.join(RESULTS, f'{cid}.json')
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding='utf-8'))
        for w in data.get('voice_windows') or []:
            emo = top_emotion(w)
            conf = (w.get('probs') or {}).get(emo, 0.0)
            start = round(w.get('start', 0), 1)
            end = round(w.get('end', 0), 1)
            label = prev.get((cid, str(start), str(end)), '')
            rows.append({
                'clip': cid,
                'start_sec': start,
                'end_sec': end,
                'model_top_emotion': emo,
                'model_confidence': round(conf, 3),
                'human_label': label,
            })

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    filled = sum(1 for r in rows if r['human_label'])
    print(f"wrote {len(rows)} windows to {OUT}")
    print(f"already labelled: {filled} (carried over from previous sheet)")
    print("listen to each window, fill the 'human_label' column, then run eval/report_agreement.py")


if __name__ == '__main__':
    main()
