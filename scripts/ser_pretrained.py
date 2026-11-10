# -*- coding: utf-8 -*-
"""Pretrained real-speech speech-emotion classifier (HuggingFace).

Wraps a HuggingFace audio-classification model (default
`superb/wav2vec2-base-superb-er`) to score a stretch of audio into the six
lowercase emotions the frontend expects. The model weights are downloaded on
first use (needs internet + the `transformers`/`torch` packages).

Used by `scripts/analyze_behavior.py` when `VOICE_MODEL` is set to
`wav2vec2`. Falls back to `None` if the model cannot be loaded, in which case
the caller keeps the classic CREMA-D RandomForest.
"""
import os


def model_id():
    return os.environ.get("SER_MODEL", "superb/wav2vec2-base-superb-er")


EMO_ALIAS = {
    "anger": "angry", "angry": "angry", "ang": "angry",
    "disgust": "disgust", "dis": "disgust",
    "fear": "fear", "fearful": "fear", "fea": "fear",
    "happy": "happy", "happiness": "happy", "hap": "happy",
    "neutral": "neutral", "neu": "neutral",
    "sad": "sad", "sadness": "sad",
}

ALL_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

_pipe = None


def _ensure_pipe():
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
        except Exception:  # transformers not installed
            return None
        _pipe = pipeline("audio-classification", model=model_id(), framework="pt")
    return _pipe


def classify_audio(audio_path, start=0.0, end=None, sr=16000):
    """Return a normalized {emotion: probability} dict or None on failure."""
    pipe = _ensure_pipe()
    if pipe is None:
        return None
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, offset=start,
                            duration=(end - start) if end is not None else None)
    except Exception:
        return None
    try:
        preds = pipe(y, top_k=6)
    except Exception:
        return None
    pm = {}
    for p in preds:
        key = str(p.get("label", "")).lower().strip()
        key = EMO_ALIAS.get(key, key)
        pm[key] = pm.get(key, 0.0) + float(p.get("score", 0.0))
    for e in ALL_EMOTIONS:
        pm.setdefault(e, 0.0)
    tot = sum(pm.values()) or 1.0
    return {k: v / tot for k, v in pm.items()}

import joblib
import numpy as np
import torch as _torch
from transformers import Wav2Vec2Model as _W2V, AutoFeatureExtractor as _FE
import librosa as _lb

_EMB_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "real_ser_embeds.joblib")
_EMB_MODEL_ID = "superb/wav2vec2-base-superb-er"
_emb_clf = None
_emb_fe = None
_emb_w2v = None


def _ensure_emb():
    global _emb_clf, _emb_fe, _emb_w2v
    if _emb_clf is None:
        _emb_clf = joblib.load(_EMB_MODEL_PATH)
        _emb_fe = _FE.from_pretrained(_EMB_MODEL_ID)
        _emb_w2v = _W2V.from_pretrained(_EMB_MODEL_ID)
        _emb_w2v.eval()
    return _emb_clf, _emb_fe, _emb_w2v


def classify_audio_emb(audio_path, start=0.0, end=None, sr=16000):
    """Probabilities from deep wav2vec2 embeddings + trained logistic
    regression (the improved model). Falls back to None on any failure."""
    try:
        clf, fe, w2v = _ensure_emb()
        y, _ = _lb.load(audio_path, sr=sr, offset=start,
                         duration=(end - start) if end is not None else None)
        if len(y) < 1600:
            y = np.pad(y, (0, 1600 - len(y)))
        inp = fe(y, sampling_rate=sr, return_tensors="pt")
        with _torch.no_grad():
            out = w2v(**inp)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        probs = clf.predict_proba(emb.reshape(1, -1))[0]
        pm = {}
        for c, p in zip(clf.classes_, probs):
            name = EMO_ALIAS.get(str(c).lower().strip(), str(c).lower().strip())
            pm[name] = pm.get(name, 0.0) + float(p)
        for e in ALL_EMOTIONS:
            pm.setdefault(e, 0.0)
        tot = sum(pm.values()) or 1.0
        return {k: v / tot for k, v in pm.items()}
    except Exception:
        return None
