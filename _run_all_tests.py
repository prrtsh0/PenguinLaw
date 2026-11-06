"""Master verification runner for the voice_prediction build."""
import os
import subprocess
import sys

base = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FILES = [
    'gnani_client.py',
    'features/extract.py',
    'scripts/download_audio.py',
    'scripts/transcribe.py',
    'scripts/make_synthetic_data.py',
    'scripts/train_tone.py',
    'scripts/predict.py',
    'scripts/report.py',
    'scripts/demo_server.py',
    '_selftest.py',
    '_test_injection.py',
]

failures = []

# 1) syntax compile all
print('== syntax compile ==')
for f in FILES:
    r = subprocess.run([PY, '-m', 'py_compile', os.path.join(base, f)],
                       capture_output=True)
    status = 'OK' if r.returncode == 0 else 'FAIL'
    print(f'  {status:4s} {f}')
    if r.returncode != 0:
        failures.append(f'syntax {f}: {r.stderr.decode()[:200]}')

# 2) feature self-test
print('== feature self-test ==')
r = subprocess.run([PY, os.path.join(base, '_selftest.py')], capture_output=True)
print(' ', r.stdout.decode().strip() or r.stderr.decode().strip())
if r.returncode != 0:
    failures.append('selftest failed')

# 3) injection guard
print('== injection guard ==')
r = subprocess.run([PY, os.path.join(base, '_test_injection.py')],
                   capture_output=True)
out = r.stdout.decode().strip()
print(' ', out)
if 'INJECTION_GUARD_OK' not in out:
    failures.append('injection guard failed')

# 4) report.py offline
print('== report.py offline ==')
r = subprocess.run([PY, os.path.join(base, 'scripts', 'report.py'),
                    '--tone', 'bullish', '--confidence', '0.83'],
                   capture_output=True)
print(' ', r.stdout.decode().strip().splitlines()[0])
if r.returncode != 0:
    failures.append('report.py failed')

# 5) demo_server import (no run)
print('== demo_server import ==')
r = subprocess.run([PY, '-c',
                    'import sys; sys.path.insert(0, r"%s"); '
                    'from scripts.demo_server import app; print("SERVER_IMPORT_OK")' % base],
                   cwd=base, capture_output=True)
out = (r.stdout + r.stderr).decode().strip()
print(' ', out.splitlines()[-1] if out else '(empty)')
if 'SERVER_IMPORT_OK' not in out:
    failures.append('demo_server import failed')

print('\n==== RESULT ====')
if failures:
    print('FAILURES:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
else:
    print('ALL CHECKS PASSED')
