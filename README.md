# PenguinLaw — Who Gives the Order? Tone-Sensitive Market Mood

**Built for the Gnani Great Indian AI Internship Challenge (submission: Nov 10, 2026).**

Penguins follow each other — but *who gives the order?* PenguinLaw tackles
that question for financial commentary. Instead of just tracking the herd
(followers), our pipeline detects **who/what originates a market-mood wave**
and how it propagates through public speech.

A pipeline that takes a spoken audio clip (earnings call / market podcast),
transcribes it with **Gnani Prisma**, analyzes **prosody** (how it's said),
and outputs a **market-mood signal** with a plain-English explanation.

> **Key idea:** the *way* someone speaks carries signal beyond the words.
> Our small, self-trained classifier learns tone (bullish/bearish/neutral/
> hesitant/confident) from audio features that text-only systems never see.

## The PenguinLaw concept — "who gives the order?"

Penguins follow one another (a swarm/herd), yet the market moves when someone
**takes the lead**. PenguinLaw flips the usual crowd-analysis question:

- **Surface:** "What is the mood of the crowd?" (sentiment of all speakers)
- **PenguinLaw:** "Who gave the order, and does the crowd follow?" —
  identify the *origin* of an opinion wave and measure whether tone
  propagates (leader → followers) or diverges.

In Phase 1, we score a single clip's tone (bullish/bearish/neutral/hesitant/
confident). The footnote extension is a **leader/follower detector** — track
which speaker's tone *precedes* others and correlates with the herd — turning
a sentiment score into an "opinion origin" signal. That is the spirit of the
Internship award categories (real-world impact, creativity, crowd favourite).
> This is a **signal**, not a prediction oracle — an honest edge.

---

## 1. Quick start (works fully OFFLINE — no API key needed)

```bash
cd voice_prediction
pip install -r requirements.txt

# 1) Generate synthetic labeled data (demonstrates the whole path)
python scripts/make_synthetic_data.py --out data/features_synth.csv --n 300

# 2) Train the tone classifier
python scripts/train_tone.py --data data/features_synth.csv --out models/tone_model_v1.joblib

# 3) Run the full prediction pipeline (auto-synthesizes a demo clip)
python scripts/predict.py

# 4) Generate the reasoning report (offline fallback)
python scripts/report.py --tone bullish --confidence 0.83
```

## 2. Live mode (requires your Gnani API key)

```bash
# set your key as an environment variable (Windows / PowerShell):
#   $env:GNANI_API_KEY="your-key-here"
# or, permanently, in your shell profile / terminal session.
python scripts/transcribe.py data/audio/clip.wav --language en
# pipeline will now use Prisma for transcription; report.py will use Evon
```

> The scripts read keys from environment variables directly (they do **not**
> load a `.env` file), so just export `GNANI_API_KEY` before running. Other
> optional vars: `GNANI_BASE_URL`, `GNANI_PRISMA_MODEL`, `GNANI_EVON_MODEL`.

## 2b. Train on REAL pre-labeled speech (no manual labeling)

Use a public emotion-speech dataset (CREMA-D, 7,442 pre-labeled clips) and a
documented emotion→market-tone map. One command downloads a balanced subset,
extracts the 22 features, and trains a model:

```bash
python scripts/build_from_emotion_dataset.py --source crema --n 150 \
    --train --out data/features_real.csv --model-out models/tone_model_v2.joblib
```

- `--n` = clips to use (150 is a good starter; 400+ for a stronger model)
- It auto-balances across the 5 tone classes and skips failed downloads.
- `--source local --local_dir <folder>` also works if you already downloaded the
  dataset (e.g. unzipped RAVDESS or CREMA-D audio).
- After training, `predict.py` auto-prefers `models/tone_model_v2.joblib`.

Caveat: these are acted/studio clips. Use real earnings-call audio
(`data/labels_template.csv` + `build_features_csv.py`) for production
fine-tuning. The emotion-trained model is an excellent, honest baseline.

**Coming back to chat when you have the key:** you'll just export
`GNANI_API_KEY` and re-run `predict.py` + `report.py` — the code is already
wired for live mode.

---

## 3. Project structure

```
voice_prediction/
├── gnani_client.py          # Gnani Prisma (STT) + Timbre (TTS) API client
├── features/extract.py      # librosa prosody feature extractor (22 features)
├── scripts/
│   ├── download_audio.py    # fetch public podcast-RSS mp3s (safe, no shell)
│   ├── transcribe.py        # CLI: audio -> transcript (Prisma)
│   ├── make_synthetic_data.py  # offline training data generator
│   ├── train_tone.py        # train + version tone classifier
│   ├── predict.py           # full pipeline (audio -> tone -> report)
│   └── report.py            # Evon reasoning (offline fallback)
├── data/                    # audio, transcripts, labeled data (gitignored)
├── models/                  # versioned models + metrics CSV
├── PLAN.md                  # detailed design & drift strategy
└── (keys via environment variables — no .env file)
```

---

## 4. Data plan

- **Your only trained model:** the small tone classifier.
- **Training data:** hand-label ~200–400 clips
  (`data/labels_template.csv`) as `bullish/bearish/neutral/hesitant/confident`.
- **Sources:** public earnings-call audio + market-podcast RSS feeds
  (`scripts/download_audio.py`).
- **Gnani's models (Prisma/Evon)** are pre-trained — you call them, not train them.

## 5. Drift management (post-deploy)

| Mechanism | Where |
|---|---|
| Model versioning (date + data hash) | `train_tone.py` saves `tone_model_vN.joblib` + `_metrics.csv` |
| Runtime feature logging | `predict.py` outputs full feature vector per call |
| Shadow retrain | train a new version, compare `test_accuracy`/F1, ship if better |
| Honest framing | tone + a *signal*, never a guaranteed prediction |

## 6. Deployment / demo

- **Local demo (for the video):** run the scripts above in a terminal or
  `gold`.ipynb`; record a 60-second screen video. Post with `#GnaniAI` and
  `#GreatIndianAIInternshipChallenge`.
- **Render (free web tier):** see `render.yaml`. Note: free tier sleeps after
  ~15 min idle → reload before recording.

---

## 7. Contest notes

- Public demo required on LinkedIn/X with `#GnaniAI` + `#GreatIndianAIInternshipChallenge`.
- Categories this fits: best solo/duo project, real-world impact, creativity,
  Hinglish/code-mixed voice (Prisma supports multiple languages), noisy audio,
  Tier-2/3 towns, best 60s demo, crowd favourite.
- Selection = verified engagement on your nominated posts + shortlisting +
  individual interviews + written offer.
