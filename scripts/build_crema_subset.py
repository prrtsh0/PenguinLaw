# -*- coding: utf-8 -*-
# Build a class-balanced CREMA-D subset and extract the 22 prosody features.
# Speaker (actor) id is recorded for speaker-independent splitting.
import os, sys, csv, io, json, time, random, urllib.request, urllib.parse, argparse
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import extract_features, feature_names

CREMA_TREE = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
CREMA_BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/"
EMOTIONS = ["ANG","DIS","FEA","HAP","NEU","SAD"]

def fetch_tree():
    req = urllib.request.Request(CREMA_TREE, headers={"User-Agent":"node","Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read().decode("utf-8"))
    wavs = [t["path"] for t in j["tree"] if t["path"].endswith(".wav")]
    return wavs

def parse_crema(path):
    base = os.path.basename(path).replace(".wav","")
    parts = base.split("_")
    if len(parts) < 3:
        return None
    actor = parts[0]
    emo = parts[2].upper()
    return actor, emo

def download_once(path, dest, retries=2):
    url = CREMA_BASE + urllib.parse.quote(path)
    for _ in range(retries+1):
        try:
            with requests.get(url, timeout=90) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
            return True
        except Exception:
            time.sleep(1)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_emotion", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/crema_subset.csv")
    ap.add_argument("--audio_dir", default="data/crema_audio")
    ap.add_argument("--cache_tree", default="data/crema_tree.json")
    args = ap.parse_args()
    random.seed(args.seed)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(base, args.audio_dir)
    os.makedirs(audio_dir, exist_ok=True)
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # fetch/load tree
    cache = os.path.join(base, args.cache_tree)
    if os.path.exists(cache):
        with open(cache) as f:
            wavs = json.load(f)
    else:
        wavs = fetch_tree()
        with open(cache, "w") as f:
            json.dump(wavs, f)
    print("total crema wavs", len(wavs))
    # group by emotion
    by_emo = {e: [] for e in EMOTIONS}
    for p in wavs:
        parsed = parse_crema(p)
        if parsed and parsed[1] in by_emo:
            by_emo[parsed[1]].append((p, parsed[0]))
    counts = {e: len(v) for e, v in by_emo.items()}
    print("available per emotion:", counts)
    # balanced selection
    selected = []
    for e in EMOTIONS:
        pool = by_emo[e]
        random.shuffle(pool)
        k = min(args.n_per_emotion, len(pool))
        selected.extend(pool[:k])
    print("selected clips", len(selected))
    # feature extraction (sequential, robust)
    keys = feature_names()
    rows = []
    failed = 0
    for i, (path, actor) in enumerate(selected):
        if i % 200 == 0:
            print("processed", i, "/", len(selected))
        local = os.path.join(audio_dir, os.path.basename(path))
        if not os.path.exists(local):
            if not download_once(path, local):
                failed += 1
                continue
        try:
            feats = extract_features(local)
        except Exception as e:
            failed += 1
            continue
        row = {"id": os.path.basename(path), "actor": actor, "label": parse_crema(path)[1]}
        for k in keys:
            row[k] = feats.get(k, 0.0)
        rows.append(row)
    print("failed", failed, "rows", len(rows))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","actor","label"]+keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
