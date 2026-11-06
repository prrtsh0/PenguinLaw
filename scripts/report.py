"""
Phase 3: Reasoning report via Gnani Evon (30B) — with OFFLINE fallback.

Usage:
    python scripts/report.py --tone bullish --confidence 0.83
"""
import argparse
import os
import sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

TONE_DESCRIPTIONS = {
    'bullish': 'bullish (risk-on) tone',
    'bearish': 'bearish (risk-off) tone',
    'neutral': 'neutral tone',
    'hesitant': 'hesitant / uncertain tone',
    'confident': 'confident tone',
}


def generate_report(tone, confidence, transcript=None):
    """Return a 2-3 sentence market-mood explanation."""
    tone_desc = TONE_DESCRIPTIONS.get(tone, tone)
    if os.environ.get('GNANI_API_KEY'):
        try:
            # Live path: use Evon (OpenAI-compatible chat endpoint) for reasoning.
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ['GNANI_API_KEY'],
                base_url=os.environ.get('GNANI_BASE_URL',
                                        'https://api.gnani.ai'),
            )
            prompt = (
                f'The audio tone of this speaker is {tone_desc} '
                f'(confidence {confidence:.0%}). '
                f'Transcript: {transcript or "(none)"} '
                'In 2-3 sentences, explain what this implies for market mood '
                'and what a trader/investor might watch for. Be concrete and '
                'avoid overclaiming prediction.'
            )
            resp = client.chat.completions.create(
                model=os.environ.get('GNANI_EVON_MODEL', 'evon'),
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=180,
            )
            return {
                'mode': 'live-evon',
                'explanation': resp.choices[0].message.content.strip(),
            }
        except Exception as e:  # fall through to offline if API fails
            fallback = f'[evon unavailable: {e}] '
    else:
        fallback = ''

    explanation = (
        f'{fallback}The speaker\'s {tone_desc} (confidence {confidence:.0%}) '
        f'suggests the market may interpret this as {tone if tone != "neutral" else "balanced"}. '
        'Watch for follow-through: sustained risk-on/off tone across multiple '
        'clips strengthens the signal; isolated clips should be treated as '
        'noise. This is a signal, not a prediction.'
    )
    return {'mode': 'offline', 'explanation': explanation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tone', required=True)
    ap.add_argument('--confidence', type=float, required=True)
    ap.add_argument('--transcript', default=None)
    args = ap.parse_args()

    report = generate_report(args.tone, args.confidence, args.transcript)
    print(f'mode: {report["mode"]}')
    print(report['explanation'])


if __name__ == '__main__':
    main()
