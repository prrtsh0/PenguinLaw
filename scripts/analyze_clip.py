"""
Analyze a single audio clip into timestamped segments, each with:
  - transcript text (via Gnani Prisma)
  - Evon JSON mood: {sentiment, certainty, pressure, one_line_reason}
  - prosody features (via features/extract.py, supporting signal only)

Writes data/results/<clip_id>.json.

Usage:
    python scripts/analyze_clip.py clip1
    python scripts/analyze_clip.py clip1 --live        # force a live re-run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract import extract_features  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = {
    'clip1': os.path.join(BASE, 'data', 'demo_clips', 'clip1.wav'),
    'clip2': os.path.join(BASE, 'data', 'demo_clips', 'clip2.wav'),
    'clip3': os.path.join(BASE, 'data', 'demo_clips', 'clip3.wav'),
}
RESULTS_DIR = os.path.join(BASE, 'data', 'results')
SEGMENT_SECONDS = 10.0

TONE_LABELS = ['bullish', 'bearish', 'neutral', 'hesitant', 'confident']


class EvonClient:
    """Minimal wrapper around the Gnani Evon chat endpoint (OpenAI-compatible).

    Uses the openai library (already a dependency) against api.gnani.ai.
    Falls back to a structured parse of the text response on JSON failure.
    """

    def __init__(self):
        self.api_key = os.environ.get('GNANI_API_KEY')
        if not self.api_key:
            raise RuntimeError('GNANI_API_KEY not set (env var)')
        self.base_url = os.environ.get('GNANI_BASE_URL', 'https://api.gnani.ai')
        self.model = os.environ.get('GNANI_EVON_MODEL', 'evon')
        self.timeout = int(os.environ.get('GNANI_TIMEOUT_SECONDS', '120'))
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f'Evon client init failed: {e}')

    def analyze_segment(self, text):
        prompt = (
            "You are a market-mood analyst. Given ONE speaker segment from a "
            "markets conversation, return STRICT JSON only (no prose) with "
            "exactly these keys:\n"
            "{\n"
            '  "sentiment": "<one of bullish|bearish|neutral|hesitant|confident>",\n'
            '  "certainty": <0.0-1.0>,\n'
            '  "pressure": <0.0-1.0>,   # how much this segment pushes the market mood\n'
            '  "one_line_reason": "<short plain-English reason>"\n'
            "}\n"
            f"SEGMENT TEXT: \"{text}\""
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            request_timeout=self.timeout,
        )
        content = resp.choices[0].message.content or ''
        return self._parse(content)

    @staticmethod
    def _parse(content):
        """Best-effort JSON extraction with a safe fallback."""
        parsed = None
        try:
            # find the first {...} JSON object in the response
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end > start:
                candidate = content[start:end + 1]
                parsed = json.loads(candidate)
        except Exception:
            parsed = None

        if not isinstance(parsed, dict):
            # fallback: derive sentiment from obvious words
            low = content.lower()
            default = {
                'sentiment': 'neutral',
                'certainty': 0.5,
                'pressure': 0.5,
                'one_line_reason': 'Evon returned non-JSON; unparsed.',
            }
            if any(w in low for w in ['bullish', 'optimistic', 'strong', 'up']):
                default['sentiment'] = 'bullish'
            elif any(w in low for w in ['bearish', 'weak', 'down', 'risk-off']):
                default['sentiment'] = 'bearish'
            elif any(w in low for w in ['hesitant', 'uncertain', 'unsure']):
                default['sentiment'] = 'hesitant'
            elif any(w in low for w in ['confident', 'firm', 'steady']):
                default['sentiment'] = 'confident'
            return default

        # prune keys
        out = {
            'sentiment': str(parsed.get('sentiment', 'neutral')),
            'certainty': float(parsed.get('certainty', 0.5)),
            'pressure': float(parsed.get('pressure', 0.5)),
            'one_line_reason': str(parsed.get('one_line_reason', '')),
        }
        if out['sentiment'] not in TONE_LABELS:
            out['sentiment'] = 'neutral'
        out['certainty'] = max(0.0, min(1.0, out['certainty']))
        out['pressure'] = max(0.0, min(1.0, out['pressure']))
        return out


def segment_transcript(segments, total_dur):
    """Split the full transcript into 10s windows with text per window."""
    # segments from Prisma typically carry start/end or timing; if none, window.
    if not segments:
        return [{'start': 0.0, 'end': total_dur, 'text': ''}]
    # If Prisma gave timestamped segments, aggregate them into windows.
    windows = []
    n = max(1, int(total_dur // SEGMENT_SECONDS) + 1)
    for i in range(n):
        start = i * SEGMENT_SECONDS
        end = min((i + 1) * SEGMENT_SECONDS, total_dur)
        text = ' '.join(
            s.get('text', '') for s in segments
            if _seg_overlaps(s, start, end)
        ).strip()
        windows.append({'start': round(start, 1), 'end': round(end, 1), 'text': text})
    return windows


def _seg_overlaps(seg, start, end):
    s = seg.get('start') or seg.get('begin') or seg.get('offset') or 0.0
    e = seg.get('end') or seg.get('finish') or seg.get('duration') or s
    if isinstance(e, (int, float)) and e == s:
        e = s + 1.0  # fallback for segment reports
    try:
        s = float(s)
        e = float(e) + (float(e) <= float(s) and 1.0 or 0.0)
    except (TypeError, ValueError):
        return False
    return not (e <= start or s >= end)


def analyze(clip_id, force_live=False, use_cache=True):
    if clip_id not in CLIPS:
        raise ValueError(f'Unknown clip: {clip_id}')
    audio = CLIPS[clip_id]
    if not os.path.exists(audio):
        raise FileNotFoundError(audio)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f'{clip_id}.json')

    # cached
    if not force_live and use_cache and os.path.exists(result_path):
        with open(result_path, encoding='utf-8') as fh:
            return json.load(fh)

    # Prisma transcription (existing client)
    from gnani_client import GnaniClient, GnaniError  # noqa: E402
    try:
        client = GnaniClient()
    except GnaniError as e:
        raise RuntimeError(f'Prisma setup failed: {e}')

    try:
        tr = client.transcribe(audio)
    except GnaniError as e:
        raise RuntimeError(f'Prisma transcription failed: {e}')

    transcript = tr.get('transcript', '')
    prisma_segments = tr.get('segments') or []
    total_dur = float(tr.get('duration_sec') or SEGMENT_SECONDS)

    windows = segment_transcript(prisma_segments, total_dur)

    # Evon per segment (skip empty text windows -> neutral fallback)
    evon = EvonClient()
    segments = []
    for w in windows:
        text = w['text']
        if not text.strip():
            segments.append({
                'start': w['start'],
                'end': w['end'],
                'text': '',
                'mood': {
                    'sentiment': 'neutral',
                    'certainty': 0.0,
                    'pressure': 0.0,
                    'one_line_reason': 'No speech detected in this window.',
                },
            })
            continue
        mood = evon.analyze_segment(text)
        segments.append({
            'start': w['start'],
            'end': w['end'],
            'text': text,
            'mood': mood,
        })

    # Prosody (supporting signal only)
    try:
        feats = extract_features(audio)
    except SystemExit:
        feats = {}

    result = {
        'clip_id': clip_id,
        'audio_path': os.path.relpath(audio, BASE),
        'duration_sec': round(total_dur, 1),
        'transcript': transcript,
        'segments': segments,
        'prosody': feats,
    }
    with open(result_path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip_id', choices=sorted(CLIPS))
    ap.add_argument('--live', action='store_true', help='force live re-run')
    args = ap.parse_args()

    try:
        result = analyze(args.clip_id, force_live=args.live)
    except Exception as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    print(f"\n=== {result['clip_id']} ===")
    print(f"transcript: {result['transcript'][:120]}...")
    for s in result['segments']:
        m = s['mood']
        print(f"  [{s['start']:.0f}-{s['end']:.0f}s] "
              f"{m['sentiment']:9s} c={m['certainty']:.2f} "
              f"p={m['pressure']:.2f} | {m['one_line_reason'][:50]}")
    print(f"wrote {os.path.join('data','results', args.clip_id + '.json')}")


if __name__ == '__main__':
    main()
