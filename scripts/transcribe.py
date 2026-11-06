"""
CLI: transcribe an audio file with Gnani Prisma.
Usage:
    python scripts/transcribe.py ../data/audio/clip.wav --language en
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnani_client import GnaniClient, GnaniError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', help='path to audio file')
    ap.add_argument('--language', default='en')
    ap.add_argument('--json', action='store_true', help='print JSON result')
    args = ap.parse_args()

    try:
        client = GnaniClient()
        result = client.transcribe(args.audio, language=args.language)
    except GnaniError as e:
        sys.exit(f'Error: {e}')

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print('TRANSCRIPT:')
        print(result['transcript'])
        if result['segments']:
            print(f"\nSEGMENTS ({len(result['segments'])}):")
            for s in result['segments'][:10]:
                print(' ', s)


if __name__ == '__main__':
    main()
