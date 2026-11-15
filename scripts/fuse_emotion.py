#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse voice-prosody emotion with the lexicon (actual wording).

The voice model (wav2vec2) reads volume/frequency/pitch acoustics but ignores
what is said. This adds a lexical channel that *boosts* the acoustic emotion:
for every 3-second window it scores the words inside with an affective lexicon,
then multiplies the acoustic probabilities by a boost factor. Wording can
strengthen an emotion the voice already suggests (and, at high weight, push a
different emotion forward when the words are strongly emotional), but it can
never wash a confident voice emotion down to neutral.

    boost[emo] = 1 + w * (lexical[emo] - 0.5) / 0.5
    fused[emo] = acoustic[emo] * boost[emo]   (then normalized)

The acoustic confidence guard still gates which windows may claim a non-neutral
emotion; wording refines the confident ones. Originals are backed up to
data/results/_orig_fuse_<clip>.json before any overwrite.

Us:
    python scripts/fuse_emotion.py [--clips clip1 ...] [--w 1.0]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

RESULTS = os.path.join(BASE, 'data', 'results')
DEFAULT_CLIPS = ['clip1', 'clip2', 'clip3', 'clip4', 'clip5']
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'emphatic']

LEXICON = {
    'angry': ['angry', 'anger', 'furious', 'outraged', 'fury', 'rage', 'fuming',
              'seething', 'infuriated', 'hate', 'hated', 'hostile', 'mad',
              'annoyed', 'irritated', 'frustrated', 'frustration', 'aggressive',
              'aggression', 'demand', 'demands', 'must', 'strict', 'outrage',
              'enraged', 'livid'],
    'fear': ['fear', 'afraid', 'scared', 'terrified', 'anxious', 'anxiety',
             'worried', 'worry', 'nervous', 'panic', 'threat', 'threatened',
             'danger', 'risk', 'uncertain', 'uncertainty', 'concern',
             'concerned', 'alarmed', 'fearful', 'dread', 'frightened'],
    'sad': ['sad', 'sadness', 'upset', 'depressed', 'grief', 'grieving',
            'sorry', 'regret', 'regrets', 'disappointed', 'disappointment',
            'lonely', 'hurt', 'pain', 'painful', 'cry', 'crying', 'heartbreak',
            'blue', 'downcast', 'devastated', 'mourn', 'melancholy'],
    'happy': ['happy', 'happiness', 'glad', 'joy', 'joyful', 'delighted',
              'excited', 'excitement', 'thrilled', 'pleased', 'great', 'good',
              'wonderful', 'fantastic', 'excellent', 'amazing', 'love',
              'loved', 'like', 'appreciate', 'celebration', 'proud',
              'optimistic', 'hopeful', 'cheerful', 'brilliant', 'terrific'],
    'disgust': ['disgust', 'disgusted', 'revolting', 'appalling', 'awful',
                'terrible', 'horrible', 'nasty', 'dislike', 'contempt',
                'despise', 'gross', 'repulsive', 'offensive', 'ugh'],
    'emphatic': ['absolutely', 'definitely', 'really', 'very', 'extremely',
                 'utterly', 'crucial', 'critical', 'key', 'major', 'huge',
                 'enormous', 'massive', 'tremendous', 'significant', 'important',
                 'vital', 'essential', 'simply', 'clearly', 'obviously',
                 'rather', 'quite', 'certainly', 'truly', 'totally', 'entirely'],
}
NEGATION = {'not', 'never', 'no', "don't", "doesn't", "didn't", "can't",
            "won't", "isn't", "aren't", 'without', 'but'}


def lexicon_scores(text):
    scores = defaultdict(float)
    tokens = text.lower().split()
    neg = False
    for tok in tokens:
        tok = tok.strip('.,!?;:"()[]')
        if tok in NEGATION:
            neg = True
            continue
        for emo, words in LEXICON.items():
            if tok in words:
                scores[emo] += -1.0 if neg else 1.0
                break
        neg = False
    return dict(scores)


def softmax(scores):
    import math
    vals = [math.exp(max(-20.0, min(20.0, s))) for s in scores.values()]
    tot = sum(vals) or 1.0
    return {e: v / tot for e, v in zip(scores.keys(), vals)}


def lexical_probs_for_window(segs, ws, we):
    text = ''
    for s in segs:
        ss, se = s.get('start', 0), s.get('end', 0)
        overlap = min(we, se) - max(ws, ss)
        if overlap <= 0:
            continue
        span = max(1e-9, se - ss)
        frac = min(1.0, overlap / span)
        text += ((s.get('text') or '') + ' ') * int(round(frac * 10))
    raw = lexicon_scores(text)
    raw['neutral'] = raw.get('neutral', 0) + 0.5
    return softmax(raw)


def boost_fuse(acoustic, lexical, w):
    """Multiplicative boost: wording amplifies emotions, never washes them out."""
    out = {}
    for e in EMOTIONS:
        lex = lexical.get(e, 0.0)
        boost = 1.0 + w * (lex - 0.5) / 0.5
        boost = max(0.3, min(3.0, boost))
        out[e] = acoustic.get(e, 0.0) * boost
    tot = sum(out.values()) or 1.0
    return {e: v / tot for e, v in out.items()}


def process(clip, w):
    path = os.path.join(RESULTS, f'{clip}.json')
    if not os.path.exists(path):
        print('MISSING', path); return
    d = json.load(open(path, encoding='utf-8'))
    wins = d.get('voice_windows') or []
    segs = d.get('segments') or []
    if not wins:
        print(clip, 'no voice windows'); return

    bak = os.path.join(RESULTS, f'_orig_fuse_{clip}.json')
    with open(bak, 'w', encoding='utf-8') as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)

    new_wins = []
    for wnd in wins:
        ws, we = wnd.get('start', 0), wnd.get('end', 0)
        acoustic = wnd.get('probs') or {}
        guarded = bool(wnd.get('guarded'))
        lexical = lexical_probs_for_window(segs, ws, we)
        if guarded:
            probs = {e: 0.0 for e in EMOTIONS}
            probs['neutral'] = 1.0
        else:
            probs = boost_fuse(acoustic, lexical, w)
        new_wins.append({
            'start': round(ws, 2),
            'end': round(we, 2),
            'probs': {k: round(v, 4) for k, v in probs.items()},
            'guarded': guarded,
            'fused': True,
            'acoustic_probs': {k: round(v, 4) for k, v in acoustic.items()},
            'lexical_probs': {k: round(v, 4) for k, v in lexical.items()},
        })

    d['voice_windows'] = new_wins
    d['emotion_method'] = f'voice+lexicon boost (w={w})'
    json.dump(d, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    tops = [max(w['probs'], key=w['probs'].get) for w in new_wins]
    print(f"{clip}: fused -> {tops}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='*', default=DEFAULT_CLIPS)
    ap.add_argument('--w', type=float, default=1.0)
    args = ap.parse_args()
    for c in args.clips:
        process(c, args.w)
    print('backups written to data/results/_orig_fuse_*.json')


if __name__ == '__main__':
    main()
