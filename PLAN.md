# PenguinLaw — Detailed Plan
## "Who Gives the Order? Tone-Sensitive Market Mood from Spoken Audio"

**Owner:** <project owner>
**Repo:** https://github.com/prrtsh0/PenguinLaw
**Status:** COMPLETE for offline build (Phases 1–3 scaffolding + training + pipeline verified).
Live-mode only waits on your Gnani API key.

---

## 0. Context & scope decision

- MiroFish (github.com/666ghj/MiroFish) is NOT our codebase. It is a **reference**
  for how a swarm-simulation prediction engine works. We borrow its *idea* (seed
  info -> simulate -> predict), not its code.
- We build a **standalone** project: `voice_prediction/` in this workspace.
- Goal for the Gnani Great Indian AI Internship Challenge:
  A pipeline that takes a **spoken audio clip** (earnings call / market podcast),
  transcribes it with **Gnani Prisma**, analyzes the **prosody (how it's said)**,
  and outputs a **market-mood signal** with a plain-English explanation.
  Submission deadline: Nov 10, 2026. Demo video required (public, #GnaniAI).

---

## 1. Architecture (3 phases)

```
audio clip (.wav/.mp3)
   │
   ├─▶ Phase 1: Gnani Prisma API  ──▶ transcript (text) + timestamps
   │
   ├─▶ Phase 2: our prosody feature extractor (librosa) ──▶ feature vector
   │                    │
   │                    ▼
   │              tone classifier (trained on our labeled clips)
   │                    │
   │                    ▼
   │          tone label + confidence (bullish/bearish/neutral/hesitant/confident)
   │
   └─▶ Phase 3: combine transcript + tone -> one "market mood" report
                (uses Gnani Evon 30B for reasoning/summary)
```

---

## 2. Phase 1 — Voice ingestion (DB: build now)

| File | Purpose |
|---|---|
| `gnani_client.py` | Prisma STT + Timbre TTS API wrapper (already drafted) |
| `scripts/download_audio.py` | Pull public audio (earnings calls, podcast RSS) |
| `scripts/transcribe.py` | CLI: audio file -> transcript via Prisma |
| `data/` | Raw audio + transcripts (gitignored) |

**Deliverable:** `python scripts/transcribe.py clip.wav` prints a transcript + segments.

---

## 3. Phase 2 — Tone model (the only thing we TRAIN)

### 3.1 Labels (the dataset)
- Classes: `bullish, bearish, neutral, hesitant, confident`.
- Source: ~200–400 short clips hand-labeled by us (ground truth).
- Split: 70% train / 20% val / 10% test.

### 3.2 Features (extracted per clip, ~20–40 floats)
From `librosa`:
| Feature | What it measures |
|---|---|
| Pitch variance / range | emotional arousal, excitement |
| Speech rate (syllables/sec) | urgency / confidence |
| Pause ratio & pause duration | hesitation / emphasis |
| Energy (RMS) mean & variance | loudness / confidence |
| MFCCs (12–13) | timbre / vocal quality |
| Spectral flux | articulation / motion |

### 3.3 Model
- Small scikit-learn classifier (Random Forest) or tiny PyTorch MLP.
- Trains in minutes on CPU (or free Google Colab GPU).
- Save as `models/tone_model_v1.onnx` (portable).

### 3.4 Drift management (post-deploy)
| Mechanism | Action |
|---|---|
| Versioning | every model saved with date + data hash |
| Runtime feature logging | every prediction appends feature vector to a CSV |
| Distribution watch | compare new vectors to training distribution; warn if shifted |
| Shadow retrain | retrain on new labeled clips; ship only if F1 improves |
| Honest framing | report tone as a *signal*, not an oracle |

---

## 4. Phase 3 — Report & demo

- `scripts/predict.py` = full pipeline: audio -> transcript + tone -> mood report.
- Define `scripts/report.py` = combine transcript + tone via Evon into 2–3 sentences.
- One Jupyter notebook `demo.ipynb` runs one real clip end-to-end and prints JSON.
- That notebook + a 60-second screen recording = the contest video.

---

## 5. Data sources (PUBLIC, free, legal)

### 5.1 Earnings-call audio
| Source | How |
|---|---|
| Company investor-relations pages | many host MP3/WAV replay links |
| Yahoo Finance / market-data sites | earnings-call audio (when public) |
| Podcasts that republish calls | RSS via `feedparser` |

### 5.2 Market-podcast RSS feeds (most reliable, downloadable)
- Use `feedparser` to read an RSS feed, then download the episode MP3.
- Example categories: "Markets", "Earnings", "Finance" podcasts.
- Choose feeds with a **direct mp3 enclosure** in the RSS item.

### 5.3 YouTube audio (if needed)
- `yt-dlp` → extract audio to `.wav`.

### 5.4 Who does what
- **I (assistant):** write all downloader/trainer/predictor scripts.
- **You (user):** run `scripts/download_audio.py` (needs internet, feeds list),
  provide your Gnani API key, and hand-label the ~200–400 clips (or we can
  start with a smaller ~50-clip proof).

---

## 6. Tooling / platform summary

| Concern | Choice |
|---|---|
| Python | 3.11–3.13 (already installed: 3.13.3) |
| Audio I/O | `librosa`, `soundfile`, `ffmpeg` |
| Gnani | Prisma API (STT), Timbre (TTS optional), Evon (reasoning, HF) |
| Training | local CPU / Google Colab |
| Storage | this workspace `data/`; later Google Drive |
| Version control | GitHub (private repo) — later |
| Demo | Render free tier (or local Jupyter for the video) |

---

## 7. Immediate TODO (next actions)
- [x] Phase 1 scripts (transcribe.py, download_audio.py) — built & syntax-verified
- [x] features/extract.py — librosa feature extraction (22 features, tested)
- [x] scripts/train_tone.py — train & save model + metrics CSV (tested on synthetic data)
- [x] scripts/predict.py — end-to-end pipeline (verified offline)
- [x] scripts/report.py — Evon reasoning with offline fallback (verified)
- [x] scripts/demo_server.py + render.yaml — deployment blueprint
- [x] README.md + docs/ERRORS.md — full documentation & error log
- [ ] LIVE MODE: set GNANI_API_KEY env var, re-run transcribe/predict/report
- [ ] Hand-label real clips (data/labels_template.csv) for production training
- [ ] Record 60-second demo video and post with #GnaniAI #GreatIndianAIInternshipChallenge
