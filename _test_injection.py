"""Test that download_audio._validate_link rejects injection payloads."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.download_audio import _validate_link

cases_ok = [
    'https://cdn.example.com/ep1.mp3',
    'http://feed.example.org/a-mp3',
]
cases_bad = [
    'https://x/1.mp3"; <payload>; "',   # classic injection
    'https://x/1.mp3`id`',             # backticks
    'https://x/1.mp3$(whoami)',        # command substitution
    'javascript:alert(1)',             # non-http scheme
    'file:///etc/passwd',              # non-http scheme
    'https://x/1.mp3 & del C:\\*',     # shell &, backslash
]

for c in cases_ok:
    assert _validate_link(c), f'should accept: {c}'
for c in cases_bad:
    assert not _validate_link(c), f'should reject: {c}'

# confirm the old risky pattern no longer runs any shell code
assert 'os.system' not in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'scripts', 'download_audio.py'), encoding='utf-8').read()
print('INJECTION_GUARD_OK')
