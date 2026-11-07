"""
Full voice-prediction pipeline.

    audio -> prosody features + tone label
          + (optional) Gnani Prisma transcript
          -> unified "market mood" report

Works OFFLINE (no API key) using a synthetic demo audio file, and LIVE
when GNANI_API_KEY is set.

Usage:
    python scripts/predict.py --audio data/audio/clip.wav
    python scripts/predict.py --audio data/audio/clip.wav --model models/tone_model_v1.joblib
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

from features.extract import extract_features  # noqa: E402

def _default_model():
    v2 = os.path.join(base, 'models', 'tone_model_v2.joblib')
    return v2 if os.path.exists(v2) else os.path.join(base, 'models', 'tone_model_v1.joblib')


DEFAULT_MODEL = _default_model()
DEFAULT_AUDIO = os.path.join(base, 'data', 'audio', 'demo.wav')


def _english_label(label):
    return {
        'bullish': 'Bullish / risk-on tone',
        'bearish': 'Bearish / risk-off tone',
        'neutral': 'Neutral tone',
        'hesitant': 'Hesitant / uncertain tone',
        'confident': 'Confident tone',
    }.get(label, label)


def load_model(path):
    if not os.path.exists(path):
        sys.exit(f'Model not found: {path} (run train_tone.py first)')
    bundle = joblib.load(path)
    return bundle['model'], sorted(bundle['classes']), bundle['features']


def predict_tone(model, feature_names, features):
    vec = np.array([[features.get(k, 0.0) for k in feature_names]])
    probs = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    conf = float(max(probs))
    return pred, conf, {c: float(p) for c, p in zip(model.classes_, probs)}


def get_transcript(audio_path):
    """Return transcript via Prisma; if no key, return a synthetic demo transcript."""
    if not os.environ.get('GNANI_API_KEY'):
        return {
            'mode': 'offline',
            'transcript': (
                'We see strong demand across our core segments this quarter. '
                'Margins are stable and we are confident in the outlook ahead.'),
        }
    try:
        from gnani_client import GnaniClient, GnaniError
        client = GnaniClient()
        r = client.transcribe(audio_path)
        r['mode'] = 'live'
        return r
    except GnaniError as e:
        return {'mode': 'error', 'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', default=DEFAULT_AUDIO)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        # synthesize a demo clip so the whole thing runs offline
        try:
            import soundfile as sf
        except ImportError:
            sys.exit('pip install soundfile')
        os.makedirs(os.path.dirname(args.audio), exist_ok=True)
        sr = 16000
        t = np.arange(sr) / sr
        # a demo "speech-like" tone: rising pitch => confident/bullish
        y = 0.5 * np.sin(2 * np.pi * (200 + 40 * t) * t)
        sf.write(args.audio, y, sr)
        print(f'[offline] synthesized demo audio -> {args.audio}')

    model, classes, fnames = load_model(args.model)
    feats = extract_features(args.audio)
    tone, conf, probs = predict_tone(model, fnames, feats)
    trans = get_transcript(args.audio)

    report = {
        'audio': args.audio,
        'tone_label': tone,
        'tone_label_english': _english_label(tone),
        'tone_confidence': conf,
        'tone_probabilities': probs,
        'features_extracted': feats,
        'transcript': trans,
        'verdict': f"Overall market mood signal: {_english_label(tone)} "
                   f"(confidence {conf:.0%}).",
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print('=' * 60)
        print('VOICE PREDICTION REPORT')
        print('=' * 60)
        print(f'  Tone: {report["tone_label_english"]} '
              f'(confidence {conf:.0%})')
        print(f'  Transcript mode: {trans.get("mode")}')
        print(f'  Transcript: {trans.get("transcript", trans.get("error"))}')
        print(f'  Verdict: {report["verdict"]}')
        print('=' * 60)


if __name__ == '__main__':
    main()
