"""
Gnani AI API client — Prisma (STT) + Timbre (TTS).
Self-contained; no dependency on MiroFish.
"""
import os
from urllib.parse import urljoin

import requests


def _load_dotenv():
    """Populate os.environ from a local .env (never overrides set vars)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.dirname(here), here):
        dotenv = os.path.join(base, '.env')
        if os.path.exists(dotenv):
            with open(dotenv, encoding='utf-8') as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            return


class GnaniError(RuntimeError):
    pass


class GnaniClient:
    def __init__(self, api_key=None, base_url=None, timeout=120):
        _load_dotenv()
        self.api_key = api_key or os.environ.get('GNANI_API_KEY')
        self.base_url = (base_url or os.environ.get(
            'GNANI_BASE_URL', 'https://api.gnani.ai')).rstrip('/')
        self.timeout = timeout
        if not self.api_key:
            raise GnaniError('GNANI_API_KEY not set (env var)')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
        }

    def transcribe(self, audio_path, language='en'):
        if not os.path.exists(audio_path):
            raise GnaniError(f'audio file not found: {audio_path}')
        url = urljoin(self.base_url + '/', 'v1/audio/transcriptions')
        with open(audio_path, 'rb') as fh:
            files = {'file': (os.path.basename(audio_path), fh)}
            data = {'model': os.environ.get('GNANI_PRISMA_MODEL', 'prisma'),
                    'language': language}
            resp = requests.post(url, headers=self.headers,
                                 files=files, data=data, timeout=self.timeout)
        if resp.status_code >= 400:
            raise GnaniError(f'Prisma {resp.status_code}: {resp.text[:200]}')
        payload = resp.json()
        transcript = payload.get('text') or payload.get('transcript') or ''
        if not transcript.strip():
            raise GnaniError('Prisma returned empty transcript')
        return {
            'transcript': transcript.strip(),
            'language': payload.get('language', language),
            'duration_sec': payload.get('duration_sec'),
            'segments': payload.get('segments') or payload.get('utterances') or [],
        }

    def synthesize(self, text, voice='default', out_path=None):
        url = urljoin(self.base_url + '/', 'v1/audio/speech')
        body = {'model': 'timbre', 'input': text, 'voice': voice}
        resp = requests.post(url, headers=self.headers, json=body,
                             timeout=self.timeout)
        if resp.status_code >= 400:
            raise GnaniError(f'Timbre {resp.status_code}: {resp.text[:200]}')
        if out_path:
            with open(out_path, 'wb') as fh:
                fh.write(resp.content)
        return resp.content
