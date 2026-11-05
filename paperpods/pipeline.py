from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .llm.base import LLMProvider
from .pdf_extract import extract_pdf_text
from .prompts import build_script_messages
from .storage import JobStore
from .tts.elevenlabs import ElevenLabsTTS


@dataclass(slots=True)
class ProcessResult:
    job_id: str
    script_path: Path
    audio_path: Path | None
    cached: bool


MAX_SCRIPT_CHARS = 10_000
MAX_SHORTENING_ATTEMPTS = 2
logger = logging.getLogger("paperpods")


class PaperPodsPipeline:
    def __init__(self, *, data_dir: Path, store: JobStore, llm: LLMProvider, tts: ElevenLabsTTS | None) -> None:
        self._data_dir = data_dir
        self._store = store
        self._llm = llm
        self._tts = tts
        self._script_dir = self._data_dir / "scripts"
        self._audio_dir = self._data_dir / "audio"
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def _shorten_script(self, *, original_prompt: str, script: str, attempt: int) -> str:
        if len(script) <= MAX_SCRIPT_CHARS:
            return script
        if attempt >= MAX_SHORTENING_ATTEMPTS:
            return self._normalize_script_length(script)

        shorten_prompt = (
            "You are editing a spoken scientific podcast script for a text-to-speech engine. "
            "Return a shorter version that stays under 10,000 characters and keeps the most important "
            "research narrative intact. Preserve technical meaning while trimming repetition and low-value detail. "
            "Do not cut in the middle of a sentence. Make it naturally readable aloud.\n\n"
            f"Original script:\n{script}"
        )
        return self._llm.generate_script(system_prompt="You are a scientific script editor.", user_prompt=shorten_prompt)

    def _normalize_script_length(self, script: str) -> str:
        if len(script) <= MAX_SCRIPT_CHARS:
            return script

        trimmed = script[:MAX_SCRIPT_CHARS].rstrip()
        for separator in (". ", "\n\n", "\n", " "):
            cut_index = trimmed.rfind(separator)
            if cut_index > MAX_SCRIPT_CHARS * 0.7:
                return trimmed[: cut_index + len(separator)].rstrip()
        return trimmed

    def process(
        self,
        *,
        pdf_path: Path,
        title: str | None = None,
        style: str = "single-narrator-15-20min",
        formula_policy: str = "conceptual-plus-key-variables",
        generate_audio: bool = False,
    ) -> ProcessResult:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Starting processing for %s", pdf_path)
        pdf_hash = _hash_file(pdf_path)
        cached_job = self._store.find_cached_job(
            pdf_sha256=pdf_hash,
            require_audio=generate_audio,
        )
        if cached_job:
            if not cached_job.script_path:
                raise RuntimeError("Cache metadata is invalid: missing script_path.")
            logger.info("Using cached script for this paper (job %s)", cached_job.id)
            return ProcessResult(
                job_id=cached_job.id,
                script_path=Path(cached_job.script_path),
                audio_path=Path(cached_job.audio_path) if cached_job.audio_path else None,
                cached=True,
            )

        cached_script_job = self._store.find_cached_job(
            pdf_sha256=pdf_hash,
            require_audio=False,
        )
        if generate_audio and cached_script_job and cached_script_job.script_path:
            if self._tts is None:
                raise RuntimeError("ELEVENLABS_API_KEY is required for audio generation.")
            logger.info("Generating missing audio from cached script for job %s", cached_script_job.id)
            audio_path = self._audio_dir / f"{cached_script_job.id}.mp3"
            script = Path(cached_script_job.script_path).read_text(encoding="utf-8")
            self._tts.synthesize(text=script, output_path=audio_path)
            self._store.update_job(cached_script_job.id, status="completed", audio_path=audio_path, error=None)
            return ProcessResult(
                job_id=cached_script_job.id,
                script_path=Path(cached_script_job.script_path),
                audio_path=audio_path,
                cached=True,
            )

        logger.info("No cached result found. Extracting and generating a new script.")
        job_id = self._store.create_job(
            pdf_path=pdf_path,
            pdf_sha256=pdf_hash,
            title=title,
            provider=self._llm.provider_name,
            model=self._llm.model_name,
            style=style,
            formula_policy=formula_policy,
        )
        self._store.update_job(job_id, status="running")

        try:
            logger.info("Extracting PDF text for %s", pdf_path)
            paper = extract_pdf_text(pdf_path)
            system_prompt, user_prompt = build_script_messages(
                paper,
                title_override=title,
                style=style,
                formula_policy=formula_policy,
            )
            logger.info("Sending script generation request to %s/%s", self._llm.provider_name, self._llm.model_name)
            script = self._llm.generate_script(system_prompt=system_prompt, user_prompt=user_prompt)
            for attempt in range(MAX_SHORTENING_ATTEMPTS + 1):
                if len(script) <= MAX_SCRIPT_CHARS:
                    break
                logger.info("Script exceeded %s chars; attempting to shorten it (attempt %s/%s)", MAX_SCRIPT_CHARS, attempt + 1, MAX_SHORTENING_ATTEMPTS + 1)
                script = self._shorten_script(original_prompt=user_prompt, script=script, attempt=attempt)
            script = self._normalize_script_length(script)
            logger.info("Generated script length: %s chars", len(script))

            script_path = self._script_dir / f"{job_id}.txt"
            script_path.write_text(script, encoding="utf-8")

            audio_path: Path | None = None
            if generate_audio:
                if self._tts is None:
                    raise RuntimeError("ELEVENLABS_API_KEY is required for audio generation.")
                logger.info("Waiting for ElevenLabs audio generation for job %s", job_id)
                audio_path = self._audio_dir / f"{job_id}.mp3"
                self._tts.synthesize(text=script, output_path=audio_path)

            self._store.update_job(
                job_id,
                status="completed",
                script_path=script_path,
                audio_path=audio_path,
                error=None,
            )
            return ProcessResult(job_id=job_id, script_path=script_path, audio_path=audio_path, cached=False)
        except Exception as exc:
            self._store.update_job(job_id, status="failed", error=str(exc))
            raise


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
