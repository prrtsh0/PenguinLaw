#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report agreement between human labels and the model on the hand-label sheet.

Reads data/results/_hand_label_sheet.csv (produced by hand_label_windows.py).
Only rows where the human filled `human_label` are counted. Prints a verifiable
number like "agrees on 14/30 (47%)".

Usage:
    python eval/report_agreement.py [sheet.csv]
"""
import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(BASE, 'data', 'results', '_hand_label_sheet.csv')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        sys.exit(f'sheet not found: {path} (run hand_label_windows.py first)')

    total = 0
    agree = 0
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            human = (r.get('human_label') or '').strip().lower()
            if not human:
                continue
            total += 1
            if human == (r.get('model_top_emotion') or '').strip().lower():
                agree += 1

    if total == 0:
        print('no human labels yet — fill the human_label column, then re-run.')
        return

    pct = 100.0 * agree / total
    print(f'agrees on {agree}/{total} ({pct:.0f}%)')
    print('Agreement = fraction of labelled windows where the human label matches '
          'the model top emotion. This is the honest, checkable number.')


if __name__ == '__main__':
    main()
