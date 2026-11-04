from __future__ import annotations

from .config import Settings
from .llm.openai_provider import OpenAIProvider
from .pipeline import PaperPodsPipeline
from .storage import JobStore
from .tts.elevenlabs import ElevenLabsTTS


def build_pipeline(settings: Settings) -> PaperPodsPipeline:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    store = JobStore(settings.data_dir / "paperpods.db")
    llm = OpenAIProvider(api_key=settings.openai_api_key, model_name=settings.openai_model)
    tts = (
        ElevenLabsTTS(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id,
        )
        if settings.elevenlabs_api_key
        else None
    )
    return PaperPodsPipeline(data_dir=settings.data_dir, store=store, llm=llm, tts=tts)

