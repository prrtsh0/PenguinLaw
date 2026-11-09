"""
PenguinLaw demo web app (NEW app; does NOT replace demo_server.py).

- Serves static/index.html at "/".
- GET /analyze/<clip_id> accepts ONLY clip1|clip2|clip3 (404 otherwise) and
  returns the cached result JSON.
- Optional ?live=1 re-runs the pipeline for that clip, rate-limited and with
  a hard cap on total live calls.

Run:
    gunicorn scripts.web_app:app
    # or locally: python scripts/web_app.py
"""
import os
import sys
import time

from flask import (Flask, jsonify, send_from_directory, abort, request,
                 send_file)
from werkzeug.exceptions import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_clip import analyze  # noqa: E402
from scripts.tone_setter import merge as apply_setter  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE, 'static')
RESULTS_DIR = os.path.join(BASE, 'data', 'results')

ALLOWED_CLIPS = {'clip1', 'clip2', 'clip3'}
LIVE_CAP = 50          # hard cap on total live pipeline calls (process lifetime)
LIVE_WINDOW_SEC = 60   # rate-limit window
LIVE_MAX_PER_WINDOW = 5

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

_live_call_count = {'n': 0}
_live_timestamps = []
_lock = __import__('threading').Lock()


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/demo_clips/<path:filename>')
def demo_clip(filename):
    # serve a demo audio clip (whitelisted basenames only)
    if filename not in ('clip1.wav', 'clip2.wav', 'clip3.wav'):
        abort(404)
    return send_from_directory(os.path.join(BASE, 'data', 'demo_clips'), filename)


@app.route('/analyze/<clip_id>')
def analyze_clip(clip_id):
    if clip_id not in ALLOWED_CLIPS:
        abort(404, description='Unknown clip')

    live = request.args.get('live') == '1'

    if live:
        if not _allow_live():
            return jsonify({'error': 'Rate limit exceeded. Try again shortly.'}), 429

    result_path = os.path.join(RESULTS_DIR, f'{clip_id}.json')
    try:
        if live:
            result = analyze(clip_id, force_live=True)
        else:
            if not os.path.exists(result_path):
                abort(404, description='No cached result for this clip yet')
            result = analyze(clip_id, force_live=False, use_cache=True)
        # ensure tone-setter verdict is present
        if 'tone_setter' not in result:
            result = apply_setter(clip_id)
    except HTTPException:
        raise
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(result)


def _allow_live():
    global _live_call_count, _live_timestamps
    with _lock:
        if _live_call_count['n'] >= LIVE_CAP:
            return False
        now = time.time()
        # drop old timestamps
        _live_timestamps = [t for t in _live_timestamps
                            if now - t < LIVE_WINDOW_SEC]
        if len(_live_timestamps) >= LIVE_MAX_PER_WINDOW:
            return False
        _live_timestamps.append(now)
        _live_call_count['n'] += 1
        return True


DOC_PATHS = {
    "README.md": "README.md",
    "JOURNEY.md": "JOURNEY.md",
    "PLAN.md": "PLAN.md",
    "benchmarks.md": "benchmarks.md",
    "ERRORS.md": os.path.join("docs", "ERRORS.md"),
}


@app.route("/docs/<name>")
def repo_doc(name):
    rel = DOC_PATHS.get(name)
    if not rel:
        abort(404, description="Unknown doc")
    doc_path = os.path.join(BASE, rel)
    if not os.path.exists(doc_path):
        abort(404, description="Doc not found")
    return send_file(doc_path, mimetype="text/markdown; charset=utf-8")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
