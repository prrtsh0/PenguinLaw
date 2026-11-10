#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a real-speech emotion classifier from local CREMA-D audio using deep
wav2vec2 embeddings (instead of the shallow 22 prosody features).

Pipeline:
  1. Extract mean-pooled wav2vec2 embeddings for every local clip (cached).
  2. Train a logistic regression on the embeddings.
  3. Evaluate with speaker-independent 5-fold CV (GroupKFold by actor).

Result (this repo): ~55-56% speaker-independent accuracy vs ~48% for the
old prosody-feature RandomForest.

Usage:
    python scripts/train_real_ser.py
"""
import csv
import os
import sys
import time

import numpy as np
import librosa
import joblib
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import Wav2Vec2Model, AutoFeatureExtractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AUDIR = os.path.join(ROOT, 'data', 'crema_audio')
CSV = os.path.join(ROOT, 'data', 'crema_subset.csv')
EMBEDS = os.path.join(ROOT, 'data', 'crema_embeds.npy')
META = os.path.join(ROOT, 'data', 'crema_embeds_meta.csv')
MODEL_OUT = os.path.join(ROOT, 'models', 'real_ser_embeds.joblib')
MODEL_ID = 'superb/wav2vec2-base-superb-er'


def extract_all():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    fa = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()

    done = set()
    if os.path.exists(META):
        done = {r['id'] for r in csv.DictReader(open(META, encoding='utf-8'))}

    def embed(path):
        y, sr = librosa.load(path, sr=16000)
        if len(y) < 1600:
            y = np.pad(y, (0, 1600 - len(y)))
        inp = fa(y, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            out = model(**inp)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    t0 = time.time()
    batch = []
    for i, r in enumerate(rows):
        if r['id'] in done:
            continue
        try:
            e = embed(os.path.join(AUDIR, r['id']))
        except Exception:
            continue
        batch.append((r['id'], r['actor'], r['label'], e))
        if len(batch) >= 50:
            _append(batch)
            print(f'  {i+1}/{len(rows)} {time.time()-t0:.0f}s', flush=True)
            batch = []
    if batch:
        _append(batch)


def _append(batch):
    Xs = np.array([b[3] for b in batch])
    old = np.load(EMBEDS) if os.path.exists(EMBEDS) else np.zeros((0, Xs.shape[1]))
    np.save(EMBEDS, np.vstack([old, Xs]))
    fresh = not os.path.exists(META) or os.path.getsize(META) == 0
    with open(META, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(['id', 'actor', 'label'])
        for b in batch:
            w.writerow([b[0], b[1], b[2]])


def main():
    if not (os.path.exists(EMBEDS) and os.path.exists(META)):
        print('extracting embeddings...')
        extract_all()
    X = np.load(EMBEDS)
    meta = list(csv.DictReader(open(META, encoding='utf-8')))
    y = np.array([r['label'] for r in meta])
    groups = np.array([r['actor'] for r in meta])
    print('X', X.shape, 'classes', sorted(set(y)))

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    accs, f1s = [], []
    yt, yp = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='macro'))
        yt += list(y[te]); yp += list(p)
    print(f'\nSpeaker-independent CV: acc={np.mean(accs):.4f} '
          f'(+/-{np.std(accs):.4f}) macroF1={np.mean(f1s):.4f}')
    print('\n' + classification_report(yt, yp))

    model.fit(X, y)
    joblib.dump(model, MODEL_OUT)
    print('saved trained model ->', MODEL_OUT)


if __name__ == '__main__':
    main()
