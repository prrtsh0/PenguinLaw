import os, shutil

home = os.path.expanduser('~')
docs = os.path.join(home, 'Documents')
print('home =', home)
print('docs exists =', os.path.isdir(docs))

src = os.getcwd()
dst = os.path.join(docs, 'PenguinLaw')
print('src =', src)
print('dst =', dst)
print('dst exists =', os.path.isdir(dst))

# compute size of src (excluding .git to be quick? include all)
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

print('src size bytes ~', dir_size(src))
print('dst size bytes ~', dir_size(dst) if os.path.isdir(dst) else 0)
