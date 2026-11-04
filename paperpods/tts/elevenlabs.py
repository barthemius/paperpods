from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request


class ElevenLabsTTS:
    def __init__(self, *, api_key: str, voice_id: str, model_id: str) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id

    def synthesize(self, *, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
        body = json.dumps(
            {
                "text": text,
                "model_id": self._model_id,
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
            }
        ).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={
                "xi-api-key": self._api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req) as response:
                audio_bytes = response.read()
        except error.HTTPError as exc:
            details = _read_http_error_body(exc)
            if exc.code == 402:
                raise RuntimeError(
                    "ElevenLabs returned HTTP 402 (Payment Required). "
                    "Your account likely has insufficient credits or billing is not active."
                    + (f" Details: {details}" if details else "")
                ) from exc
            raise RuntimeError(
                f"ElevenLabs API request failed with HTTP {exc.code}: {exc.reason}."
                + (f" Details: {details}" if details else "")
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"ElevenLabs API request failed due to a network error: {exc.reason}") from exc
        output_path.write_bytes(audio_bytes)


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
        return body[:500]
    except (OSError, UnicodeDecodeError, ValueError):
        return ""
