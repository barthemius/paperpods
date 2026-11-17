from __future__ import annotations

from .config import Settings
from .llm.google_gemini_provider import GoogleGeminiProvider
from .llm.openai_provider import OpenAIProvider
from .pipeline import PaperPodsPipeline
from .storage import JobStore
from .tts.elevenlabs import ElevenLabsTTS
from .tts.google_gemini import GoogleGeminiTTS


def build_pipeline(settings: Settings) -> PaperPodsPipeline:
    store = JobStore(settings.data_dir / "paperpods.db")
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when PAPERPODS_LLM_PROVIDER=openai.")
        llm = OpenAIProvider(api_key=settings.openai_api_key, model_name=settings.openai_model)
    elif settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when PAPERPODS_LLM_PROVIDER=gemini.")
        llm = GoogleGeminiProvider(api_key=settings.gemini_api_key, model_name=settings.gemini_model)
    else:
        raise RuntimeError(
            f"Unsupported PAPERPODS_LLM_PROVIDER '{settings.llm_provider}'. "
            "Use 'openai' or 'gemini'."
        )

    if settings.tts_provider == "elevenlabs":
        tts = (
            ElevenLabsTTS(
                api_key=settings.elevenlabs_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id=settings.elevenlabs_model_id,
            )
            if settings.elevenlabs_api_key
            else None
        )
    elif settings.tts_provider == "gemini":
        tts = (
            GoogleGeminiTTS(
                api_key=settings.gemini_api_key,
                model_id=settings.gemini_tts_model,
                voice_name=settings.gemini_tts_voice_name,
                style=settings.gemini_tts_style,
                output_format=settings.gemini_tts_output_format,
            )
            if settings.gemini_api_key
            else None
        )
    else:
        raise RuntimeError(
            f"Unsupported PAPERPODS_TTS_PROVIDER '{settings.tts_provider}'. "
            "Use 'elevenlabs' or 'gemini'."
        )

    return PaperPodsPipeline(
        data_dir=settings.data_dir,
        store=store,
        llm=llm,
        tts=tts,
        tts_provider_name=settings.tts_provider,
    )
