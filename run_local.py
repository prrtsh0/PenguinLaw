"""
Local launcher: loads `.env` then runs the Gnani pipeline/clips.

Lets you store `GNANI_API_KEY` in `.env` (gitignored) without modifying any
frozen project code. Everything here is a NEW helper file.

Usage:
    python run_local.py analyze clip1            # analyze one clip (cached)
    python run_local.py analyze clip1 --live     # force live re-run
    python run_local.py all                      # analyze clip1, clip2, clip3
    python run_local.py app                      # start web_app (dev server)
"""
import os
import sys

# load .env from project root
here = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(here, '.env'))
except ImportError:
    print('warning: python-dotenv not installed; rely on shell env vars')

sys.path.insert(0, here)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == 'analyze':
        from scripts.analyze_clip import analyze, CLIPS
        clip = args[1] if len(args) > 1 else 'clip1'
        live = '--live' in args
        if clip not in CLIPS:
            sys.exit(f'unknown clip {clip}')
        result = analyze(clip, force_live=live)
        print(f"analyzed {clip}; wrote data/results/{clip}.json")
    elif cmd == 'all':
        from scripts.analyze_clip import analyze, CLIPS
        from scripts.tone_setter import merge
        for clip in sorted(CLIPS):
            result = analyze(clip, force_live=True)
            merge(clip)
            print(f"analyzed {clip}")
    elif cmd == 'app':
        from scripts.web_app import app
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
