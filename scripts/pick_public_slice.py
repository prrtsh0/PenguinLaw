#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find the most emotionally charged N-second slice of an audio clip.

Scores every 3-second window with the same voice-emotion model the homepage
timeline uses (wav2vec2-emb -> real_ser_embeds.joblib), then slides a
`--win`-second window across the clip and picks the span whose contained
windows carry the most non-neutral/emphatic probability mass.

Usage:
    python scripts/pick_public_slice.py <audio> [--win 30] [--step 3]
        [--min-start 0] [--top 5]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np

from scripts import ser_pretrained as sp

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def window_probs(path, start, end):
    """Emotion probabilities for [start,end) using the production emb model."""
    pm = sp.classify_audio_emb(path, start=start, end=end)
    if not pm:
        return None
    # normalise + fill
    out = {e: float(pm.get(e, 0.0)) for e in EMOTIONS}
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}


def charge(probs):
    """How 'emotionally charged' one 3s window is (mass on non-neutral)."""
    p = probs or {}
    return 1.0 - p.get("neutral", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        sys.exit("audio not found: " + args.audio)

    y, sr = librosa.load(args.audio, sr=16000, mono=True)
    duration = len(y) / sr
    print(f"loaded {args.audio}: {duration:.1f}s @ {sr}Hz", flush=True)

    # precompute per-window probabilities
    wins = []  # (start,end,probs)
    t = 0.0
    while t < duration - 0.5:
        end = min(t + 3.0, duration)
        probs = window_probs(args.audio, t, end)
        wins.append((t, end, probs))
        t += 3.0

    charges = [charge(w[2]) for w in wins]
    print(f"windows: {len(wins)} (first charges: "
          f"{[round(c,3) for c in charges[:8]]})", flush=True)

    # slide the win-second window
    cands = []
    for i in range(len(wins)):
        start = wins[i][0]
        end = start + args.win
        if end > duration + 0.5:
            break
        # sum charge of windows fully/partially inside
        total = 0.0
        voiced = 0
        for j in range(i, len(wins)):
            if wins[j][0] >= end - 0.01:
                break
            total += charges[j]
            if charges[j] > 0.1:
                voiced += 1
        cands.append((total, voiced, start, min(end, duration)))

    cands.sort(key=lambda c: (-c[0], -c[1]))
    print("\nTop slices:")
    for total, voiced, start, end in cands[:args.top]:
        print(f"  start={start:6.1f}s -> {end:6.1f}s  charge={total:.2f} "
              f"voiced_windows={voiced}")

    best = cands[0]
    print(f"\nBEST SLICE: start={best[2]:.1f}s end={best[3]:.1f}s "
          f"(duration {best[3]-best[2]:.1f}s) charge={best[0]:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "audio": args.audio,
                "duration_sec": duration,
                "win_sec": args.win,
                "step_sec": args.step,
                "best": {"start": best[2], "end": best[3], "charge": best[0]},
                "candidates": [
                    {"start": s, "end": e, "charge": c, "voiced": v}
                    for c, v, s, e in cands[:args.top]
                ],
            }, fh, indent=2)
        print("report written ->", args.out)


if __name__ == "__main__":
    main()
