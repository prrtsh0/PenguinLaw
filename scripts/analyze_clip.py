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


class RuleBasedMood:
    """Deterministic fallback mood mapper (used when Evon is unavailable).

    This is a heuristic, not a trained model. It keys off simple sentiment
    word cues and is documented as such.
    """

    POSITIVE = ('bullish', 'optimistic', 'strong', 'up', 'growth', 'beat',
                'record', 'gain', 'positive', 'momentum', 'outperform')
    NEGATIVE = ('bearish', 'weak', 'down', 'decline', 'miss', 'cut',
                'loss', 'negative', 'risk', 'underperform', 'drop')
    HESITANT = ('hesitant', 'uncertain', 'unsure', 'maybe', 'might',
                'concern', 'challenge', 'unclear', 'worried')
    CONFIDENT = ('confident', 'firm', 'steady', 'certain', 'clearly',
                 'surely', 'committed', 'definitely')

    def analyze_segment(self, text):
        low = text.lower()
        pos = sum(1 for w in self.POSITIVE if w in low)
        neg = sum(1 for w in self.NEGATIVE if w in low)
        hes = sum(1 for w in self.HESITANT if w in low)
        conf = sum(1 for w in self.CONFIDENT if w in low)

        # decide sentiment
        if pos > neg + hes + conf:
            sentiment = 'bullish'
        elif neg > pos + hes + conf:
            sentiment = 'bearish'
        elif conf > hes and conf >= pos and conf >= neg:
            sentiment = 'confident'
        elif hes > pos + neg + conf:
            sentiment = 'hesitant'
        else:
            sentiment = 'neutral'

        total = pos + neg + hes + conf
        certainty = min(0.95, 0.4 + 0.1 * total) if total else 0.3
        pressure = min(0.9, 0.3 + 0.12 * (pos + neg + conf))
        reason = (f"Rule-based fallback: {pos} positive, {neg} negative, "
                  f"{hes} hesitant, {conf} confident cues.")
        return {
            'sentiment': sentiment,
            'certainty': round(certainty, 2),
            'pressure': round(pressure, 2),
            'one_line_reason': reason,
        }


class EvonClient:
    """Wrapper for a hosted chat/Evon endpoint (OpenAI-compatible).

    Reads GNANI_EVON_API_URL from env, optional; if unset or the call fails,
    we fall back to RuleBasedMood. Evon itself is open-weight on Hugging Face;
    if you have a hosted endpoint, set GNANI_EVON_API_URL and
    GNANI_EVON_MODEL.
    """

    def __init__(self):
        # Real hosted LLM reasoning. Provider auto-detected:
        #   GNANI_LLM_PROVIDER = openai | anthropic   (infer if unset)
        #   OpenAI  -> OPENAI_API_KEY, default model gpt-4o-mini
        #   Anthropic -> ANTHROPIC_API_KEY, default model claude-3-5-haiku
        self.provider = (os.environ.get('GNANI_LLM_PROVIDER') or '').lower()
        self.model = os.environ.get('GNANI_LLM_MODEL', '')
        self.timeout = int(os.environ.get('GNANI_TIMEOUT_SECONDS', '120'))
        self._oai_key = os.environ.get('OPENAI_API_KEY')
        self._llm_base_url = os.environ.get('GNANI_LLM_BASE_URL') or None
        self._anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        self._client = None
        self._fallback = RuleBasedMood()

        if not self.provider:
            if self._anthropic_key:
                self.provider = 'anthropic'
            elif self._oai_key:
                self.provider = 'openai'

    def analyze_segment(self, text):
        if self.provider == 'anthropic' and self._anthropic_key:
            try:
                return self._anthropic_analyze(text)
            except Exception as e:
                out = self._fallback.analyze_segment(text)
                out['one_line_reason'] = (
                    f"Anthropic unavailable ({type(e).__name__}); {out['one_line_reason']}"
                )
                return out
        if self.provider == 'openai' and self._oai_key:
            try:
                return self._openai_analyze(text)
            except Exception as e:
                out = self._fallback.analyze_segment(text)
                out['one_line_reason'] = (
                    f"OpenAI unavailable ({type(e).__name__}); {out['one_line_reason']}"
                )
                return out
        return self._fallback.analyze_segment(text)

    def _openai_analyze(self, text):
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(api_key=self._oai_key,
                                  base_url=self._llm_base_url)
        model = self.model or 'gpt-4o-mini'
        prompt = self._prompt()
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt.format(text=text)}],
            temperature=0.0,
            request_timeout=self.timeout,
        )
        return self._parse(resp.choices[0].message.content or '')

    def _anthropic_analyze(self, text):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._anthropic_key)
        model = self.model or 'claude-3-5-haiku-latest'
        msg = self._client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": self._prompt().format(text=text)}],
        )
        content = ''.join(
            b.text for b in msg.content if getattr(b, 'type', '') == 'text')
        return self._parse(content)

    @staticmethod
    def _prompt():
        return (
            "You are a market-mood analyst. Given ONE speaker segment from a "
            "markets conversation, return STRICT JSON only (no prose) with "
            "exactly these keys:\n"
            "{\n"
            '  "sentiment": "<one of bullish|bearish|neutral|hesitant|'
            'confident>",\n'
            '  "certainty": <0.0-1.0>,\n'
            '  "pressure": <0.0-1.0>,\n'
            '  "one_line_reason": "<short plain-English reason>"\n'
            "}\n"
             'SEGMENT TEXT: "{text}"'
        )

    @staticmethod
    def _parse(content):
        """Best-effort JSON extraction with a safe fallback."""
        parsed = None
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end > start:
                candidate = content[start:end + 1]
                parsed = json.loads(candidate)
        except Exception:
            parsed = None

        if not isinstance(parsed, dict):
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
    """Split the transcript into windows.

    If Prisma returned timestamped segments, aggregate them into ~10s windows.
    Otherwise (REST returns a single transcript with no timing), split the
    transcript into sentence-ish chunks and spread them evenly across the clip
    duration so the tone-setter still has multiple segments to work with.
    """
    if not segments:
        return [{'start': 0.0, 'end': total_dur, 'text': ''}]
    # If any segment carries a timestamp, use time-based windows.
    has_timing = any(
        s.get('start') is not None or s.get('begin') is not None
        or s.get('offset') is not None
        for s in segments
    )
    if not has_timing:
        return _sentence_windows([s.get('text', '') for s in segments],
                                 total_dur)

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


def _sentence_windows(texts, total_dur):
    """Spread joined sentence-like chunks across `total_dur` evenly."""
    joined = ' '.join(t for t in texts if t).strip()
    if not joined:
        return [{'start': 0.0, 'end': total_dur, 'text': ''}]
    # crude sentence split
    parts = []
    import re as _re
    for chunk in _re.split(r'(?<=[.!?])\s+', joined):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    if not parts:
        parts = [joined]
    n = len(parts)
    step = total_dur / n
    windows = []
    for i, p in enumerate(parts):
        windows.append({
            'start': round(i * step, 1),
            'end': round((i + 1) * step, 1),
            'text': p,
        })
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

    # Prisma transcription (new live client for api.vachana.ai)
    from scripts.gnani_live import GnaniLiveClient, GnaniLiveError  # noqa: E402
    try:
        client = GnaniLiveClient()
    except GnaniLiveError as e:
        raise RuntimeError(f'Prisma setup failed: {e}')

    try:
        tr = client.transcribe(audio, language_code='en-IN',
                               model='prisma-v3.0')
    except GnaniLiveError as e:
        raise RuntimeError(f'Prisma transcription failed: {e}')

    transcript = tr.get('transcript', '')
    prisma_segments = tr.get('segments') or []

    # Real duration from the audio file (REST returns no duration)
    try:
        import soundfile as sf
        info = sf.info(audio)
        total_dur = float(info.frames) / float(info.samplerate)
    except Exception:
        total_dur = float(tr.get('duration_sec') or SEGMENT_SECONDS)

    # If REST returned no timestamped segments, treat the full transcript as a
    # single pseudo-segment so sentence-windows can be built.
    if not prisma_segments and transcript.strip():
        prisma_segments = [{'text': transcript}]

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
