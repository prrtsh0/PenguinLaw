# PenguinLaw — Journey Log

This file is the story of the build: where we started, what we decided, what we
discovered, and where we are now. It is written for the judges and for anyone
trying to understand *why* the project looks the way it does. The technical
how-to lives in `README.md`; this is the *why*.

---

## 1. How we started

The idea came from a simple observation: on earnings calls and market podcasts,
**the way a speaker talks** (pitch, pace, confidence, hesitation, emphasis)
carries signal beyond the literal words. A CEO's shaky delivery on a "great"
quarter is often more informative than the prepared remarks.

We originally called this a "podcaster tone analyzer" and considered whether
tone could *predict* market moves. We quickly scoped that down to something
honest and provable: **detect the tone-setter** — the segment where the mood of
a conversation decisively shifts, and the followers that move with it. We are
explicitly *not* claiming prediction; tone is a signal, not a prediction.

---

## 2. Key decisions (and why)

| Decision | What we chose | Why |
|---|---|---|
| Build scope | A small, reliable 3-clip demo (not a "full app") | Deadline (Nov 10) + focus on correctness over breadth |
| Tech stack | Python + Flask + existing pipeline | Reuse the existing repo; deploy on Render free tier |
| Gnani models | Prisma (STT) for the transcript; Evon (reasoning) for per-segment mood | Prisma gives real timestamped speech; Evon gives the market-mood read |
| Tone analysis | Heuristic tone-setter (sentiment shift that persists 2+ segments) | Interpretable, no fabricated accuracy, easy to explain |
| Data | Real public audio (Planet Money + a second market source) | Genuine speech, not synthetic |
| Honesty | Never fabricate numbers; the UI shows only real pipeline output | Contest rules + trust |

---

## 3. What we discovered (the important part)

### 3.x The voice emotion model is ~48% accurate, so we added a confidence guard

The per-window emotion labels come from a CREMA-D RandomForest. Speaker-independent
cross-validation (held-out actors, `eval/baseline_cv.py`) gives only ~0.48
accuracy, and on real conversational audio the raw argmax model labeled nearly
every window "angry" — a classic domain shift from acted studio clips to real
speech. Instead of pretending the model is reliable, we added a **confidence /
domain guard**: a window whose top non-neutral emotion is below a threshold is
reported as neutral. Default is 0.7 (strict: the demo clips read all-neutral);
0.6 is the middle ground (20 neutral + 10 angry on the demo). The trade-off is
explorable via `eval/threshold_sweep.py`.

### 3.1 The frozen Gnani client pointed at a dead endpoint

The original `gnani_client.py` used:
- Host: `https://api.gnani.ai`
- Auth: `Authorization: Bearer <key>`
- STT: `POST /v1/audio/transcriptions`
- Model name: `prisma`

We verified against the official docs (`docs.gnani.ai`) that the **real** Gnani
Speech API is different:

| Real API | Value |
|---|---|
| Host | `https://api.vachana.ai` |
| Auth header | `X-API-Key-ID: <key>` |
| STT endpoint | `POST /stt/v3` (multipart form) |
| Model names | `prisma-v2.5`, `prisma-v3.0` |
| TTS endpoint | `POST /api/v1/tts/inference` |

`api.gnani.ai` does not even resolve via DNS; `api.vachana.ai` does. Because the
existing `gnani_client.py` is frozen (we agreed not to edit it), we added a thin
new live client (`scripts/gnani_live.py`) that speaks the real protocol — leaving
the frozen file untouched.

### 3.2 Audio length limit

The real REST STT transcribes clips up to **30 seconds**. Our first demo clips
were 45s. We recut them to **30s** to stay within the synchronous REST limit
(we chose REST over the async Batch API to keep the demo simple and fast).

### 3.3 Evon is open-weight, not a hosted REST API

The docs list Evon **v3.3** as an open-weight model on **Hugging Face** (Apache
2.0). There is no hosted "Evon" inference endpoint in the Speech API reference.
We therefore:
- Use **Prisma** for transcription (hosted, real).
- Use the **Evon** reasoning *approach* but call it through a hosted chat
  endpoint if available, otherwise fall back to a documented rule-based mood
  mapper. The tone-setter itself is a transparent heuristic, not a black box.

---

## 4. Where we are now

- ✅ Baseline test suite passes (`_run_all_tests.py` → ALL CHECKS PASSED).
- ✅ 3 real 30s demo clips (`data/demo_clips/clip{1,2,3}.wav`).
- ✅ `scripts/analyze_clip.py` — Prisma → timestamped transcript, split into
  segments, per-segment mood (via Evon/fallback), prosody features, writes
  `data/results/<clip_id>.json`.
- ✅ `scripts/tone_setter.py` — heuristic setter/follower detection.
- ✅ `scripts/web_app.py` — Flask app (`/analyze/<clip_id>` for clip1/2/3,
  `?live=1` rate-limited); serves `static/index.html`.
- ✅ `static/index.html` — mobile-first single page: clip buttons, audio player,
  transcript, per-segment mood, tone-setter verdict card.
- ✅ `.env` created (gitignored) + `run_local.py` launcher; `.env.example`
  committed as a template.
- ✅ Deploy blueprint: `render_web.yaml` (`gunicorn scripts.web_app:app`,
  Python 3.12.7 pinned).
- 🔜 **Next:** run live Prisma + Evon on the 3 clips, commit cached JSON,
  verify end-to-end in a browser, deploy to Render, then rename → "Tonesetter".

---

## 5. What is real vs. what is heuristic

**Real (from live pipeline):**
- Transcript text (Prisma).
- Per-segment mood labels if Evon/chat endpoint responds.
- Prosody features extracted from the audio.
- The setter/follower *segments* and *timing*.

**Heuristic / explainable:**
- The tone-setter rule (first persisting sentiment shift) is our own logic,
  deliberately simple and legible.
- The mood→tone mapping if we fall back to rules.

**Known limitations:**
- A 30s clip is a snapshot, not a full market signal.
- Tone is subjective; Evon's labels are best-effort and not a verified model.
- We do not claim predictive accuracy and never show synthetic-data accuracy.

---

## 6. Why this is credible

- We show the *real* pipeline output — nothing hardcoded or fabricated.
- We document the exact API we discovered, including the broken original host.
- We draw an explicit line between measured signal and heuristic interpretation.
- The UI footer states: *"Results from a real pipeline run on this clip. Tone is
  a signal, not a prediction."*

---

*This log is maintained as the build progresses. Last updated: build session
that added the web app, tone-setter, .env launcher, and the API-discovery notes.*
