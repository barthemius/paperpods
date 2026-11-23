from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    data_dir: Path
    llm_provider: str
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str
    script_length_mode: str
    tts_provider: str
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str
    elevenlabs_model_id: str
    gemini_tts_model: str
    gemini_tts_voice_name: str
    gemini_tts_style: str | None
    gemini_tts_output_format: str


DEFAULT_GEMINI_TTS_STYLE = "natural, clear Received Pronunciation British English"


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("PAPERPODS_DATA_DIR", "~/.paperpods")).expanduser()
    return Settings(
        data_dir=data_dir,
        llm_provider=os.getenv("PAPERPODS_LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        script_length_mode=os.getenv("PAPERPODS_SCRIPT_LENGTH_MODE", "capped").strip().lower(),
        tts_provider=os.getenv("PAPERPODS_TTS_PROVIDER", "elevenlabs").strip().lower(),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
        gemini_tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
        gemini_tts_voice_name=os.getenv("GEMINI_TTS_VOICE_NAME", "Bodi"),
        gemini_tts_style=(
            os.getenv("GEMINI_TTS_STYLE", DEFAULT_GEMINI_TTS_STYLE).strip()
            or DEFAULT_GEMINI_TTS_STYLE
        ),
        gemini_tts_output_format=os.getenv("GEMINI_TTS_OUTPUT_FORMAT", "mp3").strip().lower(),
    )
