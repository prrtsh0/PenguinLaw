# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = r'C:\Users\Pritesh\Documents\PenguinLaw\data\dl\MELD.Features.Models\features'

# MELD 7 emotions -> our 6-class schema
MAP = {
    'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
    'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': None,
}
SPLIT_NAMES = ['train', 'dev', 'test']


def load_embeddings():
    with open(os.path.join(BASE, 'audio_embeddings_feature_selection_emotion.pkl'), 'rb') as f:
        splits = pickle.load(f, encoding='latin1')  # list of 3 dicts
    return splits


def load_meta():
    with open(os.path.join(BASE, 'data_emotion.p'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # list of 6
    return data


def assemble():
    embs = load_embeddings()
    data = load_meta()
    # full metadata list (dialog, utterance, y)
    meta = data[0]
    lab = {f"{m['dialog']}_{m['utterance']}": m['y'] for m in meta}
    X, y, groups = [], [], []
    for si, emb_dict in enumerate(embs):
        for key, vec in emb_dict.items():
            label = MAP.get(lab.get(key, ''))
            if label is None:
                continue
            X.append(vec)
            y.append(label)
            # group by (split, dialogue) to avoid cross-split dialog-key collisions
            groups.append(f"{si}:{key.split('_')[0]}")
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    return X, y, groups


def main():
    X, y, groups = assemble()
    print('X', X.shape)
    print('classes', sorted(set(y)))
    print('dialogues', len(set(groups)))
    from collections import Counter
    print('label counts', dict(Counter(y)))

    # unseen-dialogue CV
    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='macro'))
    print(f'\nMELD (real dialogue) unseen-dialogue CV: acc={np.mean(accs):.4f} '
          f'macroF1={np.mean(f1s):.4f}')

    # full fit + save
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m.fit(X, y)
    out = r'C:\Users\Pritesh\Documents\PenguinLaw\models\meld_ser_embeds.joblib'
    import joblib
    joblib.dump(m, out)
    print('saved ->', out)


if __name__ == '__main__':
    main()
