#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align transcript segments to the audio using a speech-energy warp.

Prisma returns no per-segment timestamps, so the pipeline falls back to evenly
spaced sentences, which drift from what is actually spoken. This tool re-times
the segments so each sentence's span is proportional to the speech energy
around it (a lightweight VAD proxy): a sentence gets more wall-clock time where
the speaker is actually talking, and almost none across silence.

Usage:
    python eval/align_segments.py clip4 clip5 [--window 0.05]
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'data', 'results')
DEMO = os.path.join(BASE, 'data', 'demo_clips')


def energy_cdf(audio_path, win=0.05):
    """Return (times, cumulative_weight) for the utterance, normalized to [0,1]."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    n = max(1, int(win * sr))
    hop = n // 2
    # frame RMS energy
    frames = librosa.util.frame(y, frame_length=n, hop_length=hop)
    rms = np.sqrt((frames ** 2).mean(axis=0) + 1e-9)
    # subtract a noise floor so silence gets ~zero weight
    floor = np.percentile(rms, 20)
    w = np.clip(rms - floor, 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    cum = np.concatenate(([0.0], np.cumsum(w)))
    cum = cum / cum[-1]
    # time at the start of each frame row
    times = np.arange(len(cum), dtype=float) * (hop / sr)
    return times, cum


def cdf_to_time(times, cum, frac):
    """Map a cumulative fraction in [0,1] to a time via the energy CDF."""
    frac = min(1.0, max(0.0, frac))
    return float(np.interp(frac, cum, times))


def align_clip(cid, win=0.05):
    path = os.path.join(RESULTS, f'{cid}.json')
    if not os.path.exists(path):
        print('MISSING', path); return False
    d = json.load(open(path, encoding='utf-8'))
    segs = d.get('segments') or []
    if len(segs) < 2:
        print(cid, 'too few segments'); return False

    audio_path = d.get('audio_path') or os.path.join(DEMO, f'{cid}.wav')
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(BASE, audio_path)
    if not os.path.exists(audio_path):
        print(cid, 'audio not found:', audio_path); return False

    dur = float(d.get('duration_sec') or 0.0)
    times, cum = energy_cdf(audio_path, win)

    # per-sentence weight = character length (more speech => more time)
    lengths = np.array([max(1, len((s.get('text') or '').strip())) for s in segs],
                       dtype=float)
    w = lengths / lengths.sum()
    # sentence boundaries by cumulative content fraction
    bounds = np.concatenate(([0.0], np.cumsum(w)))

    new_segs = []
    for i, s in enumerate(segs):
        start = cdf_to_time(times, cum, bounds[i]) if i > 0 else 0.0
        end = cdf_to_time(times, cum, bounds[i + 1]) if i < len(segs) - 1 else dur
        if end <= start:
            end = min(dur, start + 0.5)
        ns = dict(s)
        ns['start'] = round(start, 2)
        ns['end'] = round(end, 2)
        new_segs.append(ns)

    d['segments'] = new_segs
    json.dump(d, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    spans = [(round(s['start'],1), round(s['end'],1)) for s in new_segs]
    print(cid, 'aligned:', spans[:8], '...' if len(spans) > 8 else '')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--window', type=float, default=0.05)
    args = ap.parse_args()
    for c in args.clips:
        align_clip(c, args.window)


if __name__ == '__main__':
    main()
