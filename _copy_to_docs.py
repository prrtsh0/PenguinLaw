import os, shutil

src = os.getcwd()
docs = os.path.join(os.path.expanduser('~'), 'Documents')
dst = os.path.join(docs, 'PenguinLaw')

# copy tree, preserving .git, skipping __pycache__ for cleanliness
def ignore(d, names):
    return [n for n in names if n == '__pycache__']

shutil.copytree(src, dst, ignore=ignore)

# verify
def dir_size(p):
    total = 0
    for root, dirs, files in os.walk(p):
        if '.git' in root:
            continue
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total

print('copied to', dst)
print('dst size bytes ~', dir_size(dst))
print('README exists:', os.path.exists(os.path.join(dst, 'README.md')))
print('JOURNEY exists:', os.path.exists(os.path.join(dst, 'JOURNEY.md')))
print('.env exists:', os.path.exists(os.path.join(dst, '.env')))
print('clip1 exists:', os.path.exists(os.path.join(dst, 'data', 'demo_clips', 'clip1.wav')))
