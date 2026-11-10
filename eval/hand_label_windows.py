#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a hand-labelling sheet for ~30 voice-emotion windows.

For a handful of windows across the real podcast clips this writes a CSV with
the clip, the window time span, and the model's top predicted emotion (so a
human can label them by listening). Fill the `human_label` column, then run
`report_agreement.py` to get a verifiable agreement number.

Usage:
    python eval/hand_label_windows.py                 # writes data/results/_hand_label_sheet.csv
"""
import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'data', 'results')
OUT = os.path.join(RESULTS, '_hand_label_sheet.csv')

CLIPS = ['clip1', 'clip2', 'clip3', 'clip4', 'clip5']
PER_CLIP = 6          # 6 * 5 = 30 windows


def top_emotion(w):
    probs = w.get('probs') or {}
    if not probs:
        return ''
    return max(probs, key=probs.get)


def main():
    rows = []
    for cid in CLIPS:
        path = os.path.join(RESULTS, f'{cid}.json')
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding='utf-8'))
        wins = data.get('voice_windows') or []
        # spread the sampling across the clip
        step = max(1, len(wins) // PER_CLIP)
        picked = wins[::step][:PER_CLIP]
        for w in picked:
            emo = top_emotion(w)
            conf = (w.get('probs') or {}).get(emo, 0.0)
            rows.append({
                'clip': cid,
                'start_sec': round(w.get('start', 0), 1),
                'end_sec': round(w.get('end', 0), 1),
                'model_top_emotion': emo,
                'model_confidence': round(conf, 3),
                'human_label': '',
            })

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} windows to {OUT}")
    print("listen to each window, fill the 'human_label' column, then run eval/report_agreement.py")


if __name__ == '__main__':
    main()
