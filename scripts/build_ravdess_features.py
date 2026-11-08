# -*- coding: utf-8 -*-
# Read RAVDESS from HuggingFace parquet (official files), decode WAV bytes,
# extract the 22 prosody features, and map labels to the 6 CREMA emotions.
import os, sys, csv, io, tempfile, argparse
import pyarrow.parquet as pq
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract import extract_features, feature_names

# label_text -> CREMA emotion (only the 6 shared classes)
RAV_TO_CREMA = {
    "angry": "ANG",
    "disgust": "DIS",
    "fearful": "FEA",
    "happy": "HAP",
    "neutral": "NEU",
    "sad": "SAD",
}

PARQUET_URL = "https://huggingface.co/datasets/quinnlue/ravdess_emotional_speech_audio/resolve/main/data/train-00000-of-00002.parquet"

def download_parquet(dest):
    import requests
    r = requests.get(PARQUET_URL, timeout=300, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/ravdess.parquet")
    ap.add_argument("--out", default="data/ravdess_features.csv")
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pq_path = os.path.join(base, args.parquet)
    if not os.path.exists(pq_path):
        print("downloading RAVDESS parquet...")
        download_parquet(pq_path)
    tbl = pq.read_table(pq_path, use_threads=False)
    d = tbl.to_pydict()
    wf = d["wf"]
    label_text = d["label_text"]
    paths = [w.get("path", "") for w in wf]
    print("total rows", len(wf))
    keys = feature_names()
    rows = []
    skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(len(wf)):
            emo = RAV_TO_CREMA.get(str(label_text[i]).lower())
            if emo is None:
                skipped += 1
                continue
            data = wf[i]["bytes"]
            wav = os.path.join(tmp, f"{i}.wav")
            with open(wav, "wb") as f:
                f.write(data)
            try:
                feats = extract_features(wav)
            except Exception as e:
                skipped += 1
                continue
            actor = ""
            # RAVDESS filename: 03-01-XX-XX-XX-XX-XX.wav ; actor is field 7
            fn = os.path.basename(str(paths[i]))
            parts = fn.replace(".wav","").split("-")
            if len(parts) >= 7:
                actor = parts[6]
            row = {"id": fn, "actor": actor, "label": emo}
            for k in keys:
                row[k] = feats.get(k, 0.0)
            rows.append(row)
            if (i+1) % 300 == 0:
                print("processed", i+1, "/", len(wf))
    print("rows mapped", len(rows), "skipped", skipped)
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","actor","label"]+keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
