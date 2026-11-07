"""
Tone-setter heuristic.

Given an analyzed clip (with per-segment sentiment), find the FIRST segment
where sentiment shifts away from the opening baseline AND that shift persists
for the next 2+ segments. That segment is the "tone-setter"; later segments
moving the SAME direction are "followers".

Reads data/results/<clip_id>.json, merges the verdict, writes back.

Usage:
    python scripts/tone_setter.py clip1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, 'data', 'results')

# rough "direction" grouping: two tones are the "same direction" if they share
# a bullish-ish or bearish-ish pole.
BULLISH_SET = {'bullish', 'confident'}
BEARISH_SET = {'bearish', 'hesitant'}
NEUTRAL = 'neutral'


def _pole(sentiment):
    if sentiment in BULLISH_SET:
        return 'bullish'
    if sentiment in BEARISH_SET:
        return 'bearish'
    return 'neutral'


def _same_direction(a, b):
    return _pole(a) == _pole(b)


def find_setter(segments):
    """Return (setter_index, followers_indices, note)."""
    if not segments:
        return None, [], 'No segments to analyze.'
    moods = [s.get('mood', {}).get('sentiment', 'neutral') for s in segments]

    baseline = moods[0]
    baseline_pole = _pole(baseline)

    # scan for first shift away from baseline that persists for 2+ segments
    for i in range(1, len(moods)):
        pole = _pole(moods[i])
        if pole == baseline_pole:
            continue
        # shift found at i; check persistence over next 2 segments
        horizon = moods[i + 1:i + 3]
        if len(horizon) < 2:
            continue
        persisted = all(_same_direction(moods[i], h) for h in horizon)
        if not persisted:
            continue
        # setter = segment i
        followers = [j for j in range(i + 1, len(moods))
                     if _same_direction(moods[j], moods[i])]
        note = (f"Tone shifts from '{baseline}' at segment {i + 1} "
                f"('{moods[i]}') and holds for the next 2+ segments, "
                f"making it the tone-setter. "
                f"{len(followers)} later segment(s) follow the same direction.")
        return i, followers, note

    return None, [], (
        f"No persistent sentiment shift found; tone stays near '{baseline}' "
        f"throughout. No clear tone-setter."
    )


def merge(clip_id):
    path = os.path.join(RESULTS_DIR, f'{clip_id}.json')
    if not os.path.exists(path):
        sys.exit(f'No result for {clip_id} (run analyze_clip.py first)')
    with open(path, encoding='utf-8') as fh:
        result = json.load(fh)

    setter_index, followers, note = find_setter(result.get('segments', []))
    setter = None
    if setter_index is not None:
        setter = {
            'segment_index': setter_index,
            'segment': result['segments'][setter_index],
        }
    verdict = {
        'setter_segment': setter,
        'followers': [result['segments'][i] for i in followers],
        'confidence_note': note,
    }
    result['tone_setter'] = verdict

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip_id', choices=['clip1', 'clip2', 'clip3'])
    args = ap.parse_args()
    result = merge(args.clip_id)
    v = result['tone_setter']
    print(f"=== {args.clip_id} tone-setter ===")
    print(v['confidence_note'])
    if v['setter_segment']:
        s = v['setter_segment']['segment']
        print(f"setter @ {s['start']:.0f}s "
              f"-> {s['mood']['sentiment']}")
    print(f"followers: {len(v['followers'])}")
    print(f"wrote {os.path.join('data','results', args.clip_id + '.json')}")


if __name__ == '__main__':
    main()
