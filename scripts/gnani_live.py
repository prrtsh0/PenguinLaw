"""
Gnani live speech client — talks to the REAL production API.

Verified against the official docs (docs.gnani.ai):
  - Host:     https://api.vachana.ai   (api.gnani.ai does NOT resolve)
  - Auth:     X-API-Key-ID header      (not `Authorization: Bearer`)
  - STT:      POST /stt/v3  (multipart form; audio_file/model/language_code)
  - Model:    prisma-v2.5 (default) or prisma-v3.0

This is a NEW file; the frozen gnani_client.py is intentionally untouched.
"""
import os

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


class GnaniLiveError(RuntimeError):
    pass


class GnaniLiveClient:
    def __init__(self, api_key=None, base_url=None, timeout=120):
        _load_dotenv()
        self.api_key = api_key or os.environ.get('GNANI_API_KEY')
        self.base_url = (base_url or os.environ.get(
            'GNANI_BASE_URL', 'https://api.vachana.ai')).rstrip('/')
        self.timeout = timeout
        if not self.api_key:
            raise GnaniLiveError('GNANI_API_KEY not set (env var)')
        self.headers = {'X-API-Key-ID': self.api_key}

    def transcribe(self, audio_path, language_code='en-IN', model='prisma-v3.0'):
        """Transcribe a short clip (<= 30s) over REST."""
        if not os.path.exists(audio_path):
            raise GnaniLiveError(f'audio file not found: {audio_path}')
        url = f'{self.base_url}/stt/v3'
        with open(audio_path, 'rb') as fh:
            files = {'audio_file': (os.path.basename(audio_path), fh,
                                    'audio/wav')}
            data = {'language_code': language_code}
            if model:
                data['model'] = model
            resp = requests.post(url, headers=self.headers, files=files,
                                 data=data, timeout=self.timeout)
        if resp.status_code >= 400:
            raise GnaniLiveError(
                f'Prisma {resp.status_code}: {resp.text[:240]}')
        payload = resp.json()
        if not payload.get('success'):
            err = payload.get('error') or {}
            raise GnaniLiveError(
                f"Prisma error {err.get('type')}: {err.get('message')}")
        transcript = payload.get('transcript', '')
        return {
            'transcript': transcript,
            'language': payload.get('detected_language_code', language_code),
            'duration_sec': None,
            'segments': [],
            'raw': {k: payload.get(k) for k in (
                'model', 'detected_language_code', 'processing_time',
                'end_to_end_latency') if k in payload},
        }
