import sys, os
base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
import numpy as np
from features.extract import extract_features, feature_names

sr = 16000
y = np.sin(2*np.pi*220*np.arange(sr//2)/sr) * 0.5
try:
    import soundfile as sf
except ImportError:
    print('SOUNDFILE_MISSING'); sys.exit(2)
probe = os.path.join(base, 'data', '_probe.wav')
os.makedirs(os.path.dirname(probe), exist_ok=True)
sf.write(probe, y, sr)
feats = extract_features(probe, sr=sr)
print('FEATURES:', len(feats))
print('KEYS:', list(feats.keys())[:5])
print('NAMES:', len(feature_names()))
os.remove(probe)
print('TEST_OK')
