from io import BytesIO
from pathlib import Path
import base64
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from paperpods.tts import elevenlabs as elevenlabs_module
from paperpods.tts.elevenlabs import ElevenLabsTTS
from paperpods.tts import google_gemini as google_gemini_module
from paperpods.tts.google_gemini import GoogleGeminiTTS
from paperpods.tts.google_gemini import convert_to_wav, parse_audio_mime_type


def test_elevenlabs_402_is_reported_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_402(_req):
        raise HTTPError(
            url="https://api.elevenlabs.io/v1/text-to-speech/test",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=BytesIO(b'{"detail":"insufficient credits"}'),
        )

    monkeypatch.setattr(elevenlabs_module.request, "urlopen", _raise_402)
    tts = ElevenLabsTTS(api_key="k", voice_id="v", model_id="m")

    with pytest.raises(RuntimeError, match="HTTP 402"):
        tts.synthesize(text="hello", output_path=tmp_path / "out.mp3")


def test_parse_audio_mime_type_reads_pcm_rate() -> None:
    parsed = parse_audio_mime_type("audio/L16;rate=22050")
    assert parsed["bits_per_sample"] == 16
    assert parsed["rate"] == 22050


def test_convert_to_wav_writes_riff_header() -> None:
    pcm_data = b"\x00\x00\x01\x01"
    wav_data = convert_to_wav(pcm_data, "audio/L16;rate=24000")
    assert wav_data.startswith(b"RIFF")
    assert b"WAVE" in wav_data[:16]
    assert wav_data.endswith(pcm_data)


def test_gemini_tts_uses_interactions_payload_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected_audio = b"RIFFTEST"

    class FakeInteractions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_audio=SimpleNamespace(
                    data=base64.b64encode(expected_audio).decode("ascii"),
                    mime_type="audio/wav",
                )
            )

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.interactions = FakeInteractions()

    class FakeGenAI:
        Client = FakeClient

    monkeypatch.setattr(google_gemini_module, "_import_genai", lambda: (FakeGenAI, object()))
    tts = GoogleGeminiTTS(
        api_key="k",
        model_id="gemini-3.8-flash-tts",
        voice_name="Bodi",
        style="warm and precise",
        output_format="wav",
    )
    tts.synthesize(text="hello", output_path=tmp_path / "out.wav")

    assert captured["http_options"] == {"api_version": "v1beta"}
    assert captured["response_format"] == {"type": "audio"}
    assert captured["generation_config"] == {"speech_config": [{"voice": "Bodi"}]}
    assert captured["input"][0]["content"][0]["annotations"][0]["type"] == "speech_metadata"
    assert captured["input"][0]["content"][0]["annotations"][0]["style"] == "warm and precise"
    assert (tmp_path / "out.wav").read_bytes() == expected_audio


def test_gemini_tts_mp3_uses_ffmpeg_transcoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    expected_audio = b"RIFFTEST"

    class FakeInteractions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_audio=SimpleNamespace(
                    data=base64.b64encode(expected_audio).decode("ascii"),
                    mime_type="audio/wav",
                )
            )

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.interactions = FakeInteractions()

    class FakeGenAI:
        Client = FakeClient

    def fake_run(command, check, capture_output, text):  # type: ignore[no-untyped-def]
        _ = (check, capture_output, text)
        output_path = Path(command[-1])
        output_path.write_bytes(b"MP3DATA")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(google_gemini_module, "_import_genai", lambda: (FakeGenAI, object()))
    monkeypatch.setattr(google_gemini_module, "_which_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(google_gemini_module.subprocess, "run", fake_run)

    tts = GoogleGeminiTTS(
        api_key="k",
        model_id="gemini-3.8-flash-tts",
        voice_name="Bodi",
        output_format="mp3",
    )
    out_path = tmp_path / "out.mp3"
    tts.synthesize(text="hello", output_path=out_path)

    assert out_path.read_bytes() == b"MP3DATA"


def test_gemini_tts_mp3_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeInteractions:
        @staticmethod
        def create(**kwargs):
            _ = kwargs
            return SimpleNamespace(
                output_audio=SimpleNamespace(
                    data=base64.b64encode(b"RIFFTEST").decode("ascii"),
                    mime_type="audio/wav",
                )
            )

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.interactions = FakeInteractions()

    class FakeGenAI:
        Client = FakeClient

    monkeypatch.setattr(google_gemini_module, "_import_genai", lambda: (FakeGenAI, object()))
    monkeypatch.setattr(google_gemini_module, "_which_ffmpeg", lambda: None)

    tts = GoogleGeminiTTS(
        api_key="k",
        model_id="gemini-3.8-flash-tts",
        voice_name="Bodi",
        output_format="mp3",
    )

    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        tts.synthesize(text="hello", output_path=tmp_path / "out.mp3")
