# /brag — PenguinLaw (Rendered locally, brag-style)

A 20s, 1280x720, 24fps launch video for PenguinLaw, produced without the
Hyperframes/HeyGen pipeline (no API credentials / no system ffmpeg on this
machine) by compositing real product data with PIL and encoding with OpenCV.

## What the video shows

| Time | Scene | What appears |
|------|-------|--------------|
| 0:00–0:03 | Title | PenguinLaw — Voice & Bias Analysis Console |
| 0:03–0:08 | Library | Five real clips (Powell, Obama, 3× Planet Money) with fused emotion dots |
| 0:08–0:13 | Timeline | Powell's per-3s emotion timeline sweeping in (acoustic + lexical fusion) |
| 0:13–0:17 | Analyst read | A traceable observation: type / when / evidence / voice probs |
| 0:17–0:20 | Outro | "You built it. Now brag." |

## Everything is driven by real repo data
- Clip names/topics: `scripts/web_app.py` SPECIAL_CLIP_NAME
- Emotion window labels: `data/results/clip4.json` voice_windows (fused flags)
- Analyst observation: real `"hedging"` evidence row from the clip's analysis

## Deviations from stock `/brag`
- **No Hyperframes / HeyGen render** — that step needs `npx hyperframes`
  + an API key (absent) and ffmpeg on PATH (absent).
- **Music/SFX absent** — `/brag` layers a CC-BY track + Kenney SFX via
  Hyperframes. This render is silent; a music bed can be added later.
- **Voiceover off** (matches `/brag` default).
