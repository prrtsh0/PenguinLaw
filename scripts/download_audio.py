"""
Download public audio clips for labeling/feature extraction.

Usage:
    python scripts/download_audio.py --out ../data/audio --limit 5
Sources (public, free):
  - podcast RSS feeds (mp3 enclosures) via feedparser
  - YouTube audio via yt-dlp (if installed)
"""
import argparse
import os
import re
import sys
from urllib.parse import urlparse

import requests


def _validate_link(link):
    """Only allow http/https URLs (reject shell metacharacters / other schemes)."""
    if not link:
        return False
    parsed = urlparse(link)
    if parsed.scheme not in ('http', 'https'):
        return False
    # ensure no disallowed characters survive URL parsing
    if any(ch in link for ch in ';"`$()|&<>\\'):
        return False
    return True


def download_podcast_rss(feed_url, out_dir, limit=5):
    """Download episode mp3s from an RSS feed (safe, no shell)."""
    try:
        import feedparser
    except ImportError:
        sys.exit('pip install feedparser')
    os.makedirs(out_dir, exist_ok=True)
    feed = feedparser.parse(feed_url)
    got = 0
    for entry in feed.entries:
        if got >= limit:
            break
        enc = getattr(entry, 'enclosures', None)
        if not enc:
            continue
        link = (enc[0].get('href') if enc else None)
        if not link or '.mp3' not in link or not _validate_link(link):
            continue
        title = re.sub(r'[^\w\- ]', '', entry.get('title', 'clip'))
        out_path = os.path.join(out_dir, f'{title[:40]}.mp3')
        try:
            with requests.get(link, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=65536):
                        fh.write(chunk)
        except requests.RequestException as e:
            print(f'  skip {link[:60]}... ({e})')
            continue
        if os.path.getsize(out_path) > 0:
            print(f'  saved {out_path}')
            got += 1
    print(f'Downloaded {got} clips to {out_dir}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rss', help='podcast RSS feed URL')
    ap.add_argument('--out', default='../data/audio')
    ap.add_argument('--limit', type=int, default=5)
    args = ap.parse_args()
    if args.rss:
        download_podcast_rss(args.rss, args.out, args.limit)
    else:
        print('Provide --rss <feed_url>. '
              'Example: a market/finance podcast RSS feed.')

if __name__ == '__main__':
    main()
