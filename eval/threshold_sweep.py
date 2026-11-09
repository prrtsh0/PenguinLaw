#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the confidence/domain-guard threshold across the demo clips.

Shows the trade-off between the confidence threshold and how many non-neutral
(emotion) windows are retained, using the RAW pre-guard probabilities from the
data/results/_orig_clip*.json backups.

Usage:
    python eval/threshold_sweep.py
"""
import os
import sys
import glob
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "data", "results")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def raw_windows():
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "_orig_clip*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        out.extend(w["probs"] for w in d["voice_windows"])
    return out


def apply_guard(pm, thr):
    top = max(pm, key=pm.get)
    if top == "neutral" or pm[top] >= thr:
        return top
    return "neutral"


def main():
    wins = raw_windows()
    print(f"raw windows loaded: {len(wins)} (from _orig_clip*.json backups)")
    print(f"{'threshold':>9} {'neutral':>8} {'non-neutral':>11}  detailed")
    for thr in THRESHOLDS:
        counts = {}
        for pm in wins:
            lab = apply_guard(pm, thr)
            counts[lab] = counts.get(lab, 0) + 1
        nonneutral = sum(v for k, v in counts.items() if k != "neutral")
        detail = {k: v for k, v in sorted(counts.items())}
        print(f"{thr:9.2f} {counts.get('neutral', 0):8d} {nonneutral:11d}  {detail}")


if __name__ == "__main__":
    main()
