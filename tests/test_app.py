from pathlib import Path

import pytest

from paperpods.app import build_pipeline
from paperpods.config import Settings


def _base_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        llm_provider="openai",
        openai_api_key="openai-key",
        openai_model="gpt-4.1-mini",
        gemini_api_key=None,
        gemini_model="gemini-2.5-flash",
        tts_provider="elevenlabs",
        elevenlabs_api_key="eleven-key",
        elevenlabs_voice_id="voice",
        elevenlabs_model_id="model",
        gemini_tts_model="gemini-3.8-flash-tts",
        gemini_tts_voice_name="Bodi",
        gemini_tts_style=None,
        gemini_tts_output_format="mp3",
    )


def test_build_pipeline_uses_elevenlabs_provider(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    pipeline = build_pipeline(settings)
    assert pipeline._llm.provider_name == "openai"
    assert pipeline._tts is not None
    assert pipeline._tts.provider_name == "elevenlabs"


def test_build_pipeline_uses_gemini_llm_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGeminiProvider:
        provider_name = "gemini"

        def __init__(self, *, api_key: str, model_name: str) -> None:
            self.model_name = model_name

        def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
            return "ok"

    monkeypatch.setattr("paperpods.app.GoogleGeminiProvider", FakeGeminiProvider)
    settings = _base_settings(tmp_path)
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "gemini-key"
    settings.openai_api_key = None
    pipeline = build_pipeline(settings)
    assert pipeline._llm.provider_name == "gemini"


def test_build_pipeline_uses_gemini_provider(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    settings.tts_provider = "gemini"
    settings.gemini_api_key = "gemini-key"
    settings.elevenlabs_api_key = None
    pipeline = build_pipeline(settings)
    assert pipeline._tts is not None
    assert pipeline._tts.provider_name == "gemini"
    assert pipeline._tts.audio_extension == "mp3"


def test_build_pipeline_rejects_unknown_llm_provider(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    settings.llm_provider = "unknown-provider"
    with pytest.raises(RuntimeError, match="Unsupported PAPERPODS_LLM_PROVIDER"):
        build_pipeline(settings)


def test_build_pipeline_rejects_unknown_tts_provider(tmp_path: Path) -> None:
    settings = _base_settings(tmp_path)
    settings.tts_provider = "unknown-provider"
    with pytest.raises(RuntimeError, match="Unsupported PAPERPODS_TTS_PROVIDER"):
        build_pipeline(settings)
