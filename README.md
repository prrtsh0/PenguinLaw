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

## 2c. Voice emotion model & honesty

Each clip gets per-3-second-window emotion probabilities from a CREMA-D
RandomForest (`scripts/analyze_behavior.py`; features in `features/extract.py`,
model in `models/`). Three honest facts we want on the record:

1. **It is ~48% accurate on unseen actors.** Speaker-independent cross-validation
   (held-out speakers, 6 classes, chance = 0.167) gives ~0.48:
   `python eval/baseline_cv.py`.
2. **Raw argmax labels are noisy.** On real conversational audio the model
   labeled nearly every window "angry" (domain shift — acted studio clips vs.
   real speech). So we added a **confidence / domain guard**: a window whose top
   non-neutral emotion scores below `CONFIDENCE_THRESHOLD` (default **0.6**) is
   reported as neutral instead of asserting an emotion we cannot support.
3. **The threshold is a coverage/reliability trade-off.** Sweep it with
   `python eval/threshold_sweep.py` (on the 3 demo clips, 30 windows):

   | threshold | neutral | angry |
   |---|---:|---:|
   | 0.50 | 2 | 28 |
   | 0.60 | 20 | 10 |
   | 0.70 | 30 | 0 |

   So 0.6 retains some emotion blocks (20 neutral + 10 angry); 0.7 is stricter
   and reports all 30 windows neutral. Tune per deployment with the
   `--voice-conf-threshold` CLI flag (default 0.6).

Utilities: `eval/compare_windows.py` (old argmax vs. guarded labels),
`eval/apply_guard_to_results.py` (re-guard cached result JSONs; backs up
originals as `_orig_clip*.json`).

## 3. Project structure

```
PenguinLaw/
├── features/extract.py           # librosa prosody features (22 cols)
├── scripts/
│   ├── analyze_behavior.py       # CREMA-D RF + per-window probs + guard + Gemini
│   ├── web_app.py                # Flask demo app (serves static/ + cached results)
│   └── (transcribe/report/train_tone/predict/...)
├── eval/
│   ├── baseline_cv.py            # speaker-independent CV (~0.48)
│   ├── threshold_sweep.py        # guard-threshold trade-off
│   ├── compare_windows.py        # old vs guarded labels
│   └── apply_guard_to_results.py # re-guard result JSONs
├── data/                         # crema_subset.csv, demo_clips, results/*.json
├── models/                       # versioned tone models + metrics CSV
├── static/                       # index.html, docs.html, benchmarks.html
├── PLAN.md · JOURNEY.md · README.md
└── keys via environment variables (no .env file)
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
