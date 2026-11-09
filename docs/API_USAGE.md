# API usage — Gnani AI & Google Gemini

This project calls two external APIs: **Gnani AI** (speech) and **Google Gemini**
(language / tone). Nothing runs without keys — both are read from environment
variables (see `.env.example`).

## 1. Gnani AI — speech services

Base URL `https://api.gnani.ai`, authorized with a Bearer token from
`GNANI_API_KEY`. Implemented in `gnani_client.py`.

| Capability | Endpoint | Model | Code |
|---|---|---|---|
| Speech-to-text (STT) | `POST /v1/audio/transcriptions` | `prisma` (`GNANI_PRISMA_MODEL`) | `GnaniClient.transcribe(audio_path, language)` |
| Text-to-speech (TTS) | `POST /v1/audio/speech` | `timbre` | `GnaniClient.synthesize(text, voice)` |
| Reasoning (30B) | hosted endpoint | `evon` (`GNANI_EVON_MODEL`) | `scripts/report.py` |

Where it is used:

- `scripts/transcribe.py` — CLI wrapper around `GnaniClient.transcribe`.
- `scripts/analyze_clip.py` — builds the transcript used for the tone step.
- `scripts/predict.py` — end-to-end predict using Gnani STT when
  `GNANI_API_KEY` is set.
- `scripts/report.py` — Phase-3 reasoning report via Gnani Evon (with offline
  fallback).

Env vars: `GNANI_API_KEY` (required), `GNANI_BASE_URL`,
`GNANI_PRISMA_MODEL`, `GNANI_TIMEOUT_SECONDS`, `GNANI_EVON_MODEL`.

## 2. Google Gemini — language / tone analysis

REST call to
`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`,
key from `LLM_API_KEY` (AI Studio free tier). Default model:
`gemini-2.0-flash-lite`.

| Where | What it does | Code |
|---|---|---|
| `scripts/analyze_clip.py` | Tone / verdict analysis of a clip (one request per clip) | `GeminiClient` |
| `scripts/analyze_behavior.py` | Behavioural observations from transcript + per-window voice probabilities | `call_gemini(...)` |

Env vars: `LLM_API_KEY` (required), `LLM_MODEL` (optional override).

## Notes on privacy & cost

- Both APIs send audio/text off-device.
- Both are rate-limited and capped in `web_app.py` (`LIVE_CAP`,
  `LIVE_WINDOW_SEC`, `LIVE_MAX_PER_WINDOW`).
- Keys are gitignored via `.env`.
