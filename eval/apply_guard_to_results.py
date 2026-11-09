#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-apply the domain guard to result JSONs (backing up originals).

Regenerates data/results/clipN.json so every voice window carries the guarded
flag and low-confidence windows are set to neutral. Original files are backed up
as _orig_clipN.json if not already present.

Usage:
    python eval/apply_guard_to_results.py [--threshold 0.7]
"""
import os
import sys
import argparse
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "data", "results")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
CLIPS = ["clip1", "clip2", "clip3"]


def apply_guard(pm, thr):
    top = max(pm, key=pm.get)
    if top == "neutral" or pm[top] >= thr:
        return pm, True
    guarded = {k: 0.0 for k in EMOTIONS}
    guarded["neutral"] = 1.0
    return guarded, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    for name in CLIPS:
        path = os.path.join(RESULTS, name + ".json")
        backup = os.path.join(RESULTS, f"_orig_{name}.json")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        d = json.load(open(path, encoding="utf-8"))
        guarded_count = 0
        for w in d.get("voice_windows", []):
            pm, kept = apply_guard(w.get("probs", {}), args.threshold)
            w["probs"] = {k: round(v, 4) for k, v in pm.items()}
            w["guarded"] = not kept
            guarded_count += 0 if kept else 1
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        print(f"re-guarded {name} at {args.threshold:.2f} "
              f"({guarded_count}/{len(d.get('voice_windows', []))} windows guarded)")


if __name__ == "__main__":
    main()
