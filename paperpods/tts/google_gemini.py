from __future__ import annotations

import base64
import mimetypes
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


class GoogleGeminiTTS:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        voice_name: str,
        style: str | None = None,
        output_format: str = "mp3",
    ) -> None:
        normalized_output_format = output_format.strip().lower()
        if normalized_output_format not in {"mp3", "wav"}:
            raise ValueError("Unsupported Gemini TTS output format. Use 'mp3' or 'wav'.")
        self._api_key = api_key
        self._model_id = model_id
        self._voice_name = voice_name
        self._style = style
        self.audio_extension = normalized_output_format

    def synthesize(self, *, text: str, output_path: Path) -> None:
        genai, _types = _import_genai()
        # TTS interactions live on v1beta per current Gemini speech docs.
        client = genai.Client(api_key=self._api_key, http_options={"api_version": "v1beta"})

        text_content: dict[str, object] = {"type": "text", "text": text}
        if self._style:
            text_content["annotations"] = [{"type": "speech_metadata", "style": self._style}]

        request_input = [{"type": "user_input", "content": [text_content]}]
        generation_config = {"speech_config": [{"voice": self._voice_name}]}

        try:
            interaction = client.interactions.create(
                model=self._model_id,
                input=request_input,
                response_format={"type": "audio"},
                generation_config=generation_config,
            )
        except Exception as exc:  # pragma: no cover - network/provider failure path
            raise RuntimeError(f"Google Gemini TTS request failed: {exc}") from exc

        audio_b64 = _extract_audio_data(interaction)
        if not audio_b64:
            raise RuntimeError("Google Gemini TTS returned no audio data.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_bytes = base64.b64decode(audio_b64)
        mime_type = _extract_audio_mime_type(interaction)
        wav_bytes = output_bytes
        if mime_type and not _is_wav_mime(mime_type):
            wav_bytes = convert_to_wav(output_bytes, mime_type)

        if self.audio_extension == "wav":
            output_path.write_bytes(wav_bytes)
            return
        _transcode_wav_to_mp3(wav_bytes=wav_bytes, output_path=output_path)


def _import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dependency missing path
        raise RuntimeError(
            "Google Gemini TTS requires the `google-genai` package. "
            "Install dependencies with `uv sync` or `pip install google-genai`."
        ) from exc
    return genai, types


def _is_wav_mime(mime_type: str) -> bool:
    if not mime_type:
        return False
    return "wav" in mime_type.lower()


def _which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _transcode_wav_to_mp3(*, wav_bytes: bytes, output_path: Path) -> None:
    ffmpeg_path = _which_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "Gemini MP3 output requires ffmpeg, but it was not found in PATH. "
            "Install ffmpeg or set GEMINI_TTS_OUTPUT_FORMAT=wav."
        )

    with tempfile.TemporaryDirectory(prefix="paperpods-gemini-tts-") as temp_dir:
        input_wav = Path(temp_dir) / "input.wav"
        input_wav.write_bytes(wav_bytes)
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            details = f" Details: {stderr[:500]}" if stderr else ""
            raise RuntimeError(f"ffmpeg failed to convert Gemini WAV output to MP3.{details}") from exc


def _extract_audio_data(interaction: object) -> str | None:
    output_audio = getattr(interaction, "output_audio", None)
    data = getattr(output_audio, "data", None)
    if data:
        return data

    steps = getattr(interaction, "steps", None) or []
    for step in reversed(list(steps)):
        content_items = getattr(step, "content", None) or []
        for item in reversed(list(content_items)):
            if getattr(item, "type", None) == "audio" and getattr(item, "data", None):
                return item.data
    return None


def _extract_audio_mime_type(interaction: object) -> str:
    output_audio = getattr(interaction, "output_audio", None)
    mime_type = getattr(output_audio, "mime_type", None)
    if mime_type:
        return mime_type

    steps = getattr(interaction, "steps", None) or []
    for step in reversed(list(steps)):
        content_items = getattr(step, "content", None) or []
        for item in reversed(list(content_items)):
            if getattr(item, "type", None) == "audio":
                value = getattr(item, "mime_type", None)
                if value:
                    return value
    return ""


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    sample_rate = params["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    if not mime_type:
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    guessed_ext = mimetypes.guess_extension(mime_type)
    if guessed_ext == ".wav":
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    parts = mime_type.split(";")
    for param in parts:
        item = param.strip()
        if item.lower().startswith("rate="):
            try:
                rate = int(item.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif "audio/l" in item.lower():
            try:
                marker = item.lower().split("audio/l", 1)[1]
                bits_per_sample = int(marker)
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}
