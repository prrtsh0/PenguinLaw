"""
Extract prosody features from an audio file using librosa.
Returns a flat dict of ~20-40 floats (the feature vector).

Usage:
    python -c "from features.extract import extract_features; print(extract_features('clip.wav'))"
"""
import os

import numpy as np


def extract_features(audio_path, sr=16000):
    try:
        import librosa
    except ImportError:
        raise SystemExit('pip install librosa soundfile')

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        return {}

    import scipy.signal as sps
    f = {}
    dur = len(y) / sr

    # pitch (f0) using librosa pyin
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'), sr=sr)
        f0_voiced = f0[voiced_flag]
        if f0_voiced.size:
            f['pitch_mean'] = float(np.mean(f0_voiced))
            f['pitch_std'] = float(np.std(f0_voiced))
            f['pitch_range'] = float(np.percentile(f0_voiced, 95) -
                                     np.percentile(f0_voiced, 5))
        else:
            f['pitch_mean'] = f['pitch_std'] = f['pitch_range'] = 0.0
    except Exception:
        f['pitch_mean'] = f['pitch_std'] = f['pitch_range'] = 0.0

    # energy / RMS
    rms = librosa.feature.rms(y=y)[0]
    f['energy_mean'] = float(np.mean(rms))
    f['energy_std'] = float(np.std(rms))

    # speech rate proxy & pause ratio (via energy-thresholded activity)
    act = rms > (np.percentile(rms, 30) + 1e-9)
    f['activity_ratio'] = float(np.mean(act))
    # approxim. speech rate: voiced segments per second
    n_seg = max(1, len(np.where(np.diff(act.astype(int)) == 1)[0]))
    f['speech_rate_seg_per_sec'] = float(n_seg / dur)

    # MFCCs (timbre)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    for i, v in enumerate(mfcc_mean):
        f[f'mfcc_{i}'] = float(v)

    # spectral flux / articulation
    sf = librosa.onset.onset_strength(y=y, sr=sr)
    f['spectral_flux_mean'] = float(np.mean(sf))

    f['duration_sec'] = float(dur)
    return f


def extract_window_features(audio_path, start_sec, end_sec, sr=16000):
    # extract prosody features for a time window [start_sec, end_sec]
    try:
        import librosa
    except ImportError:
        raise SystemExit("pip install librosa soundfile")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    s = max(0, int(start_sec * sr))
    e = min(len(y), int(end_sec * sr))
    seg = y[s:e]
    if seg.size == 0:
        return {}
    f = {}
    dur = len(seg) / sr
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            seg, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"), sr=sr)
        f0_voiced = f0[voiced_flag]
        if f0_voiced.size:
            f["pitch_mean"] = float(np.mean(f0_voiced))
            f["pitch_std"] = float(np.std(f0_voiced))
            f["pitch_range"] = float(np.percentile(f0_voiced, 95) -
                                     np.percentile(f0_voiced, 5))
        else:
            f["pitch_mean"] = f["pitch_std"] = f["pitch_range"] = 0.0
    except Exception:
        f["pitch_mean"] = f["pitch_std"] = f["pitch_range"] = 0.0
    rms = librosa.feature.rms(y=seg)[0]
    f["energy_mean"] = float(np.mean(rms))
    f["energy_std"] = float(np.std(rms))
    act = rms > (np.percentile(rms, 30) + 1e-9)
    f["activity_ratio"] = float(np.mean(act))
    n_seg = max(1, len(np.where(np.diff(act.astype(int)) == 1)[0]))
    f["speech_rate_seg_per_sec"] = float(n_seg / dur) if dur else 0.0
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    for i, v in enumerate(mfcc_mean):
        f["mfcc_" + str(i)] = float(v)
    sf = librosa.onset.onset_strength(y=seg, sr=sr)
    f["spectral_flux_mean"] = float(np.mean(sf))
    f["duration_sec"] = float(dur)
    return f

def feature_names():
    """Return ordered feature keys for building a training matrix."""
    # probe with a tiny synthetic signal
    import numpy as np
    sr = 16000
    y = np.random.randn(sr // 4)
    import soundfile as sf
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        feats = extract_features(tmp.name, sr=sr)
    os.unlink(tmp.name)
    return sorted(feats.keys())
