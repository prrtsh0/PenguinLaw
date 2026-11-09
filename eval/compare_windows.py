#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare old argmax labels vs the new domain-guarded labels for the demo clips.

Loads the raw pre-guard windows (_orig_clip*.json) and the current guarded
results (clipN.json), then summarizes each window's old label (argmax) versus
its new label after the guard, plus how many windows changed.

Usage:
    python eval/compare_windows.py
"""
import os
import sys
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "data", "results")

CLIPS = ["clip1", "clip2", "clip3"]


def label_of(pm, guarded):
    top = max(pm, key=pm.get)
    return "neutral(guarded)" if guarded else top


def main():
    total = collections.Counter()
    changes = collections.Counter()
    for name in CLIPS:
        raw = json.load(open(os.path.join(RESULTS, f"_orig_{name}.json"), encoding="utf-8"))
        cur = json.load(open(os.path.join(RESULTS, f"{name}.json"), encoding="utf-8"))
        rw = raw["voice_windows"]
        cw = cur["voice_windows"]
        assert len(rw) == len(cw), f"window count mismatch in {name}"
        for r, c in zip(rw, cw):
            old = label_of(r["probs"], False)
            new = label_of(c["probs"], c.get("guarded", False))
            total[old] += 1
            if old != new:
                changes[(old, new)] += 1

    print("old argmax labels (raw, pre-guard):")
    for lab, n in sorted(total.items()):
        print(f"  {lab:>16}: {n}")
    print("\nchanges after guard:")
    for (old, new), n in changes.most_common():
        print(f"  {old:>16} -> {new:<16} x{n}")
    if not changes:
        print("  (no windows changed)")


if __name__ == "__main__":
    main()
