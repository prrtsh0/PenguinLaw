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

from features.extract import extract_features, extract_window_features  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREMA_SAMPLES = {
    # emotion -> (filename, friendly label)
    'angry':   ('1001_TAI_ANG_XX.wav', 'Angry (acted)'),
    'angry2':  ('1001_IEO_ANG_LO.wav', 'Angry low (acted)'),
    'disgust': ('1001_DFA_DIS_XX.wav', 'Disgust (acted)'),
    'disgust2':('1001_IEO_DIS_LO.wav', 'Disgust low (acted)'),
    'fear':    ('1001_TIE_FEA_XX.wav', 'Fear (acted)'),
    'fear2':   ('1001_IEO_FEA_MD.wav', 'Fear medium (acted)'),
    'happy':   ('1001_TIE_HAP_XX.wav', 'Happy (acted)'),
    'happy2':  ('1001_TAI_HAP_XX.wav', 'Happy (acted)'),
    'neutral': ('1001_TAI_NEU_XX.wav', 'Neutral (acted)'),
    'neutral2':('1001_IOM_NEU_XX.wav', 'Neutral (acted)'),
    'sad':     ('1001_IOM_SAD_XX.wav', 'Sad (acted)'),
    'sad2':    ('1001_MTI_SAD_XX.wav', 'Sad (acted)'),
}

CLIPS = {
    'clip1': os.path.join(BASE, 'data', 'demo_clips', 'clip1.wav'),
    'clip2': os.path.join(BASE, 'data', 'demo_clips', 'clip2.wav'),
    'clip3': os.path.join(BASE, 'data', 'demo_clips', 'clip3.wav'),
}
for _i, (_key, (_fn, _label)) in enumerate(CREMA_SAMPLES.items(), start=4):
    CLIPS[f'clip{_i}'] = os.path.join(BASE, 'data', 'demo_clips', f'clip{_i}.wav')

RESULTS_DIR = os.path.join(BASE, 'data', 'results')
SEGMENT_SECONDS = 10.0

TONE_LABELS = ['bullish', 'bearish', 'neutral', 'hesitant', 'confident']


class RuleBasedMood:
    """Deterministic fallback mood mapper (used when Evon is unavailable).

    This is a heuristic, not a trained model. It keys off simple sentiment
    word cues and is documented as such.
    """

    POSITIVE = {
        "bullish": 3, "optimistic": 2, "strong": 2, "up": 1.5,
        "growth": 2, "beat": 2.5, "beats": 2.5, "record": 2,
        "gain": 1.5, "gains": 1.5, "positive": 2, "momentum": 2,
        "outperform": 2.5, "rally": 2.5, "surge": 2, "soar": 2.5,
        "expand": 1.5, "recover": 1.5, "improve": 1.5, "exceed": 2,
        "buy": 1.5, "opportunity": 1.5, "profit": 2, "earnings": 1,
        "revenue": 1, "high": 1, "solid": 1.5, "robust": 2,
        "healthy": 1.5, "accelerate": 2, "resilient": 2, "jump": 2,
        "climb": 1.5, "rise": 1.5, "rising": 1.5, "higher": 1.5,
        "favorable": 1.5,
    }

    NEGATIVE = {
        "bearish": 3, "weak": 2, "down": 1.5, "decline": 2,
        "miss": 2.5, "cut": 2, "loss": 2, "negative": 2, "risk": 2,
        "underperform": 2.5, "drop": 2, "fall": 1.5, "fell": 2,
        "slump": 2.5, "weakness": 2, "volatile": 2, "uncertainty": 1.5,
        "lower": 1.5, "sell": 1.5, "poor": 1.5, "downgrade": 2.5,
        "warn": 2, "deteriorate": 2.5, "struggle": 2, "pressure": 1.5,
        "below": 1.5, "short": 1, "debt": 1, "layoff": 2,
        "restructuring": 2, "sluggish": 2, "disappoint": 2.5,
    }

    HESITANT = {
        "hesitant": 2.5, "uncertain": 2.5, "unsure": 2.5, "maybe": 1.5,
        "might": 1.5, "could": 1, "concern": 2, "challenge": 2,
        "unclear": 2.5, "worried": 2, "ambiguous": 2, "speculate": 2,
        "contemplate": 1.5, "questionable": 2, "tepid": 2, "mixed": 1.5,
    }

    CONFIDENT = {
        "confident": 2.5, "firm": 2, "steady": 1.5, "certain": 2.5,
        "clearly": 2, "surely": 2, "committed": 2, "definitely": 2.5,
        "absolutely": 2.5, "guarantee": 3, "certainly": 2.5,
        "undoubtedly": 2.5, "maintain": 2, "conviction": 3,
        "will": 1, "should": 1, "forecast": 1.5, "target": 1.5,
    }

    NEGATORS = ("not", "no", "never", "without", "lacks",
                "fail", "fails", "failed", "reject", "denies",
                "cannot", "hardly", "barely")


    def _score(self, low):
        tokens = low.split()
        pos = neg = hes = conf = 0.0
        n = len(tokens)
        i = 0
        while i < n:
            w = tokens[i]
            negated = False
            if i > 0 and tokens[i-1] in self.NEGATORS:
                negated = True
            cat = None
            weight = 0.0
            lw = w.lower()
            if lw in self.POSITIVE:
                cat, weight = "pos", self.POSITIVE[lw]
            elif lw in self.NEGATIVE:
                cat, weight = "neg", self.NEGATIVE[lw]
            elif lw in self.HESITANT:
                cat, weight = "hes", self.HESITANT[lw]
            elif lw in self.CONFIDENT:
                cat, weight = "conf", self.CONFIDENT[lw]
            if cat:
                if negated and cat in ("pos", "neg"):
                    if cat == "pos":
                        neg += weight
                    else:
                        pos += weight
                elif cat == "pos":
                    pos += weight
                elif cat == "neg":
                    neg += weight
                elif cat == "hes":
                    hes += weight
                elif cat == "conf":
                    conf += weight
            i += 1
        return pos, neg, hes, conf


    def analyze_segment(self, text, prosody=None):
        low = text.lower()
        pos, neg, hes, conf = self._score(low)

        if pos >= max(neg, hes, conf) and pos > 0:
            sentiment = "bullish"
        elif neg > pos and neg >= hes and neg >= conf:
            sentiment = "bearish"
        elif conf > hes and conf >= pos and conf >= neg:
            sentiment = "confident"
        elif hes >= max(pos, neg, conf) and hes > 0:
            sentiment = "hesitant"
        else:
            sentiment = "neutral"

        total = pos + neg + hes + conf
        certainty = min(0.97, 0.35 + 0.08 * total) if total else 0.3
        pressure = min(0.95, 0.3 + 0.12 * (pos + neg + conf))

        if prosody:
            energy = prosody.get("energy_mean", 0.0)
            flux = prosody.get("spectral_flux_mean", 0.0)
            rate = prosody.get("speech_rate_seg_per_sec", 0.0)
            pitch_std = prosody.get("pitch_std", 0.0)
            if energy > 0.15:
                pressure = min(0.97, pressure + 0.1)
            if flux > 0.05:
                pressure = min(0.97, pressure + 0.08)
            if 1.5 < rate < 12:
                certainty = min(0.97, certainty + 0.08)
            certainty = min(0.97, max(0.1, certainty - pitch_std / 2000.0))

        reason = (f"Lexicon-weighted {int(pos)}+/{int(neg)}-/{int(hes)}?"
                  f"/{int(conf)}! cues.")
        if prosody:
            reason += " [prosody used]"
        return {
            "sentiment": sentiment,
            "certainty": round(certainty, 2),
            "pressure": round(pressure, 2),
            "one_line_reason": reason,
        }

class GeminiClient:
    """Tone analysis via Google Gemini (AI Studio free tier).

    Reads LLM_API_KEY (API key) and LLM_MODEL (model id, default
    gemini-2.0-flash-lite) from the environment. Sends ONE request for all
    segments of a clip and requires a strict JSON array back — one object per
    segment: {market_related, stance, certainty, pressure, reason}. Validates
    the JSON and retries once; if it still fails it raises an error with the
    raw body. There is NO fallback to the lexicon.
    """

    DEFAULT_MODEL = 'gemini-2.0-flash-lite'

    def __init__(self):
        self.api_key = os.environ.get('LLM_API_KEY', '').strip()
        self.model = (os.environ.get('LLM_MODEL', '').strip()
                      or self.DEFAULT_MODEL)
        self.timeout = int(os.environ.get('GNANI_TIMEOUT_SECONDS', '120'))
        if not self.api_key:
            raise RuntimeError(
                'LLM_API_KEY is not set; cannot run the Gemini tone step.'
            )

    def _request(self, texts):
        import requests
        url = ('https://generativelanguage.googleapis.com/v1beta/models/'
               f'{self.model}:generateContent')
        headers = {'x-goog-api-key': self.api_key,
                   'Content-Type': 'application/json'}
        lines = '\n'.join(f'[{i}] "{t}"' for i, t in enumerate(texts))
        prompt = (
            'You are a market-mood analyst. Below is a series of timestamped '
            'speaker segments from a markets conversation. For EACH segment, '
            'output exactly one JSON object in the SAME order, forming a JSON '
            'array. Return STRICT JSON ONLY \u2014 no prose, no markdown fences.\n'
            'Each object must have exactly these keys:\n'
            '{\n'
            '  "market_related": <true|false>,\n'
            '  "stance": "<risk_on|risk_off|uncertain|none>",\n'
            '  "certainty": <0.0-1.0>,\n'
            '  "pressure": <0.0-1.0>,\n'
            '  "reason": "<short plain-English reason>"\n'
            '}\n'
            'Segments:\n' + lines
        )
        body = {
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.0,
                                 'responseMimeType': 'application/json'},
        }
        resp = requests.post(url, headers=headers, json=body,
                             timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f'Gemini HTTP {resp.status_code}: {resp.text[:300]}'
            )
        data = resp.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f'Gemini unexpected response envelope: {json.dumps(data)[:300]}'
            )

    def analyze_clip(self, segments):
        texts = [s['text'] for s in segments]
        last_raw = ''
        for attempt in range(2):      # validate + retry once
            last_raw = self._request(texts)
            moods = self._parse(last_raw, len(segments))
            if moods is not None:
                return moods
        raise RuntimeError(
            'Gemini returned invalid JSON after one retry. Raw body:\n'
            + last_raw[:500]
        )

    @staticmethod
    def _parse(content, expected):
        text = content.strip()
        if text.startswith('```'):
            text = text.strip('`').lstrip()
            if text.lower().startswith('json'):
                text = text[4:]
        try:
            arr = json.loads(text)
        except Exception:
            return None
        if not isinstance(arr, list) or len(arr) != expected:
            return None
        stance_map = {'risk_on': 'bullish', 'risk_off': 'bearish',
                      'uncertain': 'hesitant', 'none': 'neutral'}
        out = []
        for item in arr:
            if not isinstance(item, dict):
                return None
            stance = str(item.get('stance', 'none')).lower()
            sentiment = stance_map.get(stance, 'neutral')
            if item.get('market_related') is False:
                sentiment = 'neutral'
            try:
                certainty = float(item.get('certainty', 0.5))
                pressure = float(item.get('pressure', 0.5))
            except (TypeError, ValueError):
                return None
            out.append({
                'sentiment': sentiment,
                'certainty': max(0.0, min(1.0, certainty)),
                'pressure': max(0.0, min(1.0, pressure)),
                'one_line_reason': str(item.get('reason', '')),
            })
        return out


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

    def analyze_segment(self, text, prosody=None):
        if self.provider == 'anthropic' and self._anthropic_key:
            try:
                return self._anthropic_analyze(text)
            except Exception as e:
                out = self._fallback.analyze_segment(text, prosody=prosody)
                out['one_line_reason'] = (
                    f"Anthropic unavailable ({type(e).__name__}); {out['one_line_reason']}"
                )
                return out
        if self.provider == 'openai' and self._oai_key:
            try:
                return self._openai_analyze(text)
            except Exception as e:
                out = self._fallback.analyze_segment(text, prosody=prosody)
                out['one_line_reason'] = (
                    f"OpenAI unavailable ({type(e).__name__}); {out['one_line_reason']}"
                )
                return out
        return self._fallback.analyze_segment(text, prosody=prosody)

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
    # Record stt_engine ONLY if Prisma actually produced non-empty text (the
    # only STT path here is Gnani Prisma; this analyze() raises on Prisma
    # failure, so any saved transcript was produced by Prisma). Leave it unset
    # otherwise so the UI shows "not recorded".
    stt_engine = 'prisma-v3.0' if transcript.strip() else None

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

    # Batch tone analysis via Google Gemini (ONE request per clip). No lexicon
    # fallback: if Gemini returns bad JSON twice, analyze raises an error.
    gemini = GeminiClient()
    segments = []
    pending = []            # indices of windows with real speech (in order)
    for i, w in enumerate(windows):
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
        else:
            pending.append(i)
            segments.append({
                'start': w['start'],
                'end': w['end'],
                'text': text,
                'mood': None,
            })

    if pending:
        moods = gemini.analyze_clip([segments[i] for i in pending])
        for i, mood in zip(pending, moods):
            segments[i]['mood'] = mood

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
        'tone_engine': gemini.model,
    }
    if stt_engine:
        result['stt_engine'] = stt_engine
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
