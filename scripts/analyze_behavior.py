# -*- coding: utf-8 -*-
"""
Behavioural analysis of conversation clips via Gemini.

For each clip in data/results/clip*.json it adds an "analysis" object:
  - trains (or re-trains) the CREMA-D RandomForest on the 22 prosody features
  - computes per-3-second-window emotion probabilities for the clip audio
  - sends ONE Gemini request per clip with a fixed SYSTEM PROMPT and the
    transcript (with segment times), window voice probabilities, and the
    clip-level prosody summary
  - requires strict JSON back, validates evidence strings against the
    transcript, drops observations whose quote does not appear verbatim
  - records the exact model name in "engine"
  - no silent fallback: if JSON fails twice it raises with the raw body
"""
import os
import sys
import csv
import json
import argparse
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract import feature_names, extract_window_features  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
WINDOW_SEC = 3.0
SYSTEM_PROMPT = (
    "You are a senior behavioural analyst reviewing a recorded conversation. "
    "You never decide whether anyone is lying, guilty or truthful, and you never "
    "imply it; no method can do that reliably from a recording. You identify "
    "observable indicators a skilled interviewer would probe: contradictions, "
    "evasions, unanswered questions, answers to unasked questions, minimising or "
    "hedging language, sudden topic shifts, over-rehearsed or over-controlled "
    "phrasing, mismatch between words and voice, and details that are missing "
    "where detail would be expected. Every observation must quote exact words "
    "from the transcript (max 15 words), give the time in seconds if known, "
    "state the voice cue at that moment if one exists, and say in one sentence "
    "why it matters. For the main observations, also give the most plausible "
    "innocent explanation. Voice scores are hints only and never evidence by "
    "themselves. If the clip cannot support an observation, return an empty "
    "observations list and say why in limits. Return ONLY JSON: "
    '{"engine":"<model name>","situation":"","baseline":"","observations":['
    '{"type":"contradiction|evasion|hedging|deflection|minimising|'
    'unasked_answer|unanswered_question|topic_shift|over_control|'
    'voice_mismatch|missing_detail|other","evidence":"","time":null,"voice":null,'
    '"why_it_matters":"","strength":"weak|moderate|strong"}],'
    '"innocent_explanations":[],"what_to_ask_next":[],"impact":"","limits":""}'
)

VALID_TYPES = {
    "contradiction", "evasion", "hedging", "deflection", "minimising",
    "unasked_answer", "unanswered_question", "topic_shift", "over_control",
    "voice_mismatch", "missing_detail", "other",
}


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def train_model(crema_csv, seed=42, trees=300):
    keys = feature_names()
    data = load_csv(crema_csv)
    X = np.array([[float(r[k]) for k in keys] for r in data])
    y = np.array([r["label"] for r in data])
    clf = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    return clf, keys


def window_probabilities(audio_path, duration_sec, clf, keys):
    """Return a list of {start,end,probabilities} per 3-second window."""
    out = []
    t = 0.0
    while t < duration_sec - 0.01:
        end = min(t + WINDOW_SEC, duration_sec)
        feats = extract_window_features(audio_path, t, end)
        if feats:
            row = np.array([[float(feats[k]) for k in keys]])
            probs = clf.predict_proba(row)[0]
            prob_map = {c: round(float(p), 4) for c, p in zip(clf.classes_, probs)}
            out.append({"start": round(t, 2), "end": round(end, 2),
                        "probabilities": prob_map})
        t = end
    return out


def prosody_summary(prosody):
    keep = ["pitch_mean", "pitch_std", "pitch_range", "energy_mean",
            "energy_std", "activity_ratio", "speech_rate_seg_per_sec",
            "duration_sec"]
    lines = []
    for k in keep:
        if k in prosody:
            v = prosody[k]
            try:
                v = round(float(v), 3)
            except (TypeError, ValueError):
                pass
            lines.append(f"{k}={v}")
    return "; ".join(lines)


def build_transcript(transcript, segments):
    s = transcript.strip()
    if segments:
        s += "\n\nSEGMENT TIMES:"
        for seg in segments:
            s += f"\n  [{seg.get('start')}-{seg.get('end')}] {seg.get('text','')}"
    return s


def call_gemini(api_key, model, system_prompt, user_text, timeout=120):
    import requests
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Gemini unexpected response envelope: {json.dumps(data)[:300]}")


def extract_json(text):
    """Robustly pull a JSON object out of a model reply."""
    t = text.strip()
    # if it is already (or wrapped) valid json, try directly
    try:
        return json.loads(t)
    except Exception:
        pass
    # fall back to first { ... last }
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(t[start:end + 1])
        return obj
    raise ValueError("no JSON object found")


def parse_json_strict(raw, model):
    """Validate JSON; returns dict. Always includes engine."""
    try:
        obj = extract_json(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON from model: {e}\nRAW BODY:\n{raw}")
    if not isinstance(obj, dict):
        raise ValueError(f"JSON is not an object.\nRAW BODY:\n{raw}")
    obj["engine"] = model
    return obj


def filter_evidence(obj, transcript):
    """Drop observations whose evidence quote is not in the transcript."""
    if not isinstance(obj.get("observations"), list):
        return obj
    t = transcript.lower()
    kept = []
    for obs in obj["observations"]:
        if not isinstance(obs, dict):
            continue
        ev = str(obs.get("evidence") or "").strip()
        if not ev:
            continue
        # force type into allowed set
        typ = str(obs.get("type") or "other").lower()
        if typ not in VALID_TYPES:
            typ = "other"
            obs["type"] = typ
        if ev.lower() in t:
            obs["type"] = typ
            kept.append(obs)
    obj["observations"] = kept
    return obj


def load_dotenv(base):
    """Minimal .env loader (no dependency). Populates os.environ."""
    path = os.path.join(base, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if k and k not in os.environ:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip ids, e.g. clip1 clip2 clip3 (default: all)")
    ap.add_argument("--crema", default="data/crema_subset.csv")
    ap.add_argument("--results_dir", default="data/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--no-retrain", action="store_true",
                    help="skip model training if model already cached")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(base)
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip() or "gemini-2.0-flash-lite"
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set; cannot run Gemini analysis.")

    # train CREMA-D RF
    print("training CREMA-D RandomForest ...", flush=True)
    clf, keys = train_model(os.path.join(base, args.crema),
                             seed=args.seed, trees=args.trees)

    results_dir = os.path.join(base, args.results_dir)
    if args.clips:
        clip_files = [os.path.join(results_dir, c + ".json") for c in args.clips]
    else:
        clip_files = sorted(
            os.path.join(results_dir, f) for f in os.listdir(results_dir)
            if f.startswith("clip") and f.endswith(".json"))

    for clip_path in clip_files:
        if not os.path.exists(clip_path):
            print(f"!! missing clip file: {clip_path}")
            continue
        with open(clip_path, "r", encoding="utf-8") as fh:
            clip = json.load(fh)

        clip_id = clip.get("clip_id", os.path.basename(clip_path))
        print(f"\n=== analysing {clip_id} ===", flush=True)

        audio_rel = clip.get("audio_path")
        audio_path = os.path.join(base, audio_rel) if not os.path.isabs(audio_rel) else audio_rel
        if not os.path.exists(audio_path):
            raise RuntimeError(f"audio not found: {audio_path}")

        duration = float(clip.get("duration_sec", 0))
        wins = window_probabilities(audio_path, duration, clf, keys)
        print(f"  windows: {len(wins)}", flush=True)

        # clip-level prosody summary
        prosody = clip.get("prosody") or {}
        prosody_txt = prosody_summary(prosody)
        # audio-level prosody from the clip's prosody dict (may have all 22)
        prosody_all = prosody

        # build input
        transcript = clip.get("transcript", "")
        segments = clip.get("segments") or []
        transcript_txt = build_transcript(transcript, segments)

        win_lines = []
        for w in wins:
            pb = " ".join(f"{k}={w['probabilities'][k]}" for k in sorted(w["probabilities"]))
            win_lines.append(f"[{w['start']}-{w['end']}] {pb}")
        windows_txt = "\n".join(win_lines)
        if not windows_txt:
            windows_txt = "(no voiced windows)"

        user_text = (
            "CLIP: " + clip_id + "\n\n"
            "TRANSCRIPT WITH SEGMENT TIMES:\n" + transcript_txt + "\n\n"
            "PER-3-SECOND-WINDOW VOICE EMOTION PROBABILITIES "
            "(from our trained CREMA-D model; weak signal, about 50% accuracy "
            "on unseen actors):\n" + windows_txt + "\n\n"
            "CLIP-LEVEL PROSODY SUMMARY:\n" + prosody_txt + "\n"
        )

        raw = ""
        parsed = None
        for attempt in range(2):  # validate + retry once
            raw = call_gemini(api_key, model, SYSTEM_PROMPT, user_text)
            try:
                parsed = parse_json_strict(raw, model)
                break
            except ValueError as e:
                if attempt == 0:
                    print(f"  retry once (invalid JSON): {str(e)[:120]}", flush=True)
                    continue
                raise RuntimeError(str(e)) from e

        parsed = filter_evidence(parsed, transcript)

        # final structural cleanup
        parsed.setdefault("situation", "")
        parsed.setdefault("baseline", "")
        parsed.setdefault("innocent_explanations", [])
        parsed.setdefault("what_to_ask_next", [])
        parsed.setdefault("impact", "")
        parsed.setdefault("limits", "")
        if not isinstance(parsed.get("observations"), list):
            parsed["observations"] = []

        analysis = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "engine": model,
            "clip_id": clip_id,
            **parsed,
        }
        clip["analysis"] = analysis

        with open(clip_path, "w", encoding="utf-8") as fh:
            json.dump(clip, fh, ensure_ascii=False, indent=2)

        print(f"  wrote analysis -> {clip_path}", flush=True)

    print("\ndone.")


if __name__ == "__main__":
    main()
