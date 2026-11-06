"""
Minimal HTTP server for the Render demo deployment.
Exposes:
  GET  /health   -> {"status":"ok"}
  POST /predict  -> runs the prediction pipeline on an uploaded audio file
Run:
  python scripts/demo_server.py
"""
import io
import json
import os
import sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

from flask import Flask, jsonify, request  # noqa: E402

from scripts.predict import load_model, predict_tone, DEFAULT_MODEL  # noqa: E402
from features.extract import extract_features  # noqa: E402

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

MODEL_PATH = os.environ.get('TONE_MODEL', DEFAULT_MODEL)
_model = None
_fnames = None


def _get_model():
    global _model, _fnames
    if _model is None:
        _model, _classes, _fnames = load_model(MODEL_PATH)
    return _model, _fnames


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'missing audio file'}), 400
    f = request.files['audio']
    tmp = os.path.join(base, 'data', 'audio', '_upload.wav')
    f.save(tmp)
    try:
        model, fnames = _get_model()
        feats = extract_features(tmp)
        tone, conf, probs = predict_tone(model, fnames, feats)
        return jsonify({
            'tone_label': tone,
            'tone_confidence': conf,
            'tone_probabilities': probs,
            'features_extracted': feats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
