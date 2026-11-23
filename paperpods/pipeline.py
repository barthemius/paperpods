from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .llm.base import LLMProvider
from .pdf_extract import extract_pdf_text
from .prompts import build_script_messages
from .storage import JobStore
from .tts.base import TTSProvider


@dataclass(slots=True)
class ProcessResult:
    job_id: str
    script_path: Path
    audio_path: Path | None
    cached: bool


MAX_SCRIPT_CHARS = 10_000
MAX_SHORTENING_ATTEMPTS = 2
DEFAULT_TTS_CHUNK_CHARS = 9_000
ELEVENLABS_TTS_CHUNK_CHARS = 9_000
GEMINI_TTS_CHUNK_CHARS = 20_000
logger = logging.getLogger("paperpods")


class PaperPodsPipeline:
    def __init__(
        self,
        *,
        data_dir: Path,
        store: JobStore,
        llm: LLMProvider,
        tts: TTSProvider | None,
        tts_provider_name: str = "elevenlabs",
    ) -> None:
        self._data_dir = data_dir
        self._store = store
        self._llm = llm
        self._tts = tts
        self._tts_provider_name = tts_provider_name
        self._script_dir = self._data_dir / "scripts"
        self._audio_dir = self._data_dir / "audio"
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, *, directory: Path, title: str | None, pdf_path: Path, extension: str) -> Path:
        stem = _slugify_title(title or pdf_path.stem)
        candidate = directory / f"{stem}.{extension}"
        counter = 2
        while candidate.exists():
            candidate = directory / f"{stem}_{counter}.{extension}"
            counter += 1
        return candidate

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

    def _normalize_script_with_policy(self, *, script: str, original_prompt: str, script_length_mode: str) -> str:
        normalized_mode = script_length_mode.strip().lower()
        if normalized_mode == "unlimited":
            return script.strip()
        if normalized_mode != "capped":
            raise ValueError("script_length_mode must be 'capped' or 'unlimited'.")

        for attempt in range(MAX_SHORTENING_ATTEMPTS + 1):
            if len(script) <= MAX_SCRIPT_CHARS:
                break
            logger.info(
                "Script exceeded %s chars; attempting to shorten it (attempt %s/%s)",
                MAX_SCRIPT_CHARS,
                attempt + 1,
                MAX_SHORTENING_ATTEMPTS + 1,
            )
            script = self._shorten_script(original_prompt=original_prompt, script=script, attempt=attempt)
        return self._normalize_script_length(script)

    def _tts_chunk_chars(self) -> int:
        provider = (self._tts_provider_name or "").strip().lower()
        if provider == "elevenlabs":
            return ELEVENLABS_TTS_CHUNK_CHARS
        if provider == "gemini":
            return GEMINI_TTS_CHUNK_CHARS
        return DEFAULT_TTS_CHUNK_CHARS

    def _split_script_for_tts(self, *, script: str, max_chars: int) -> list[str]:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        text = script.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
            if len(paragraph) <= max_chars:
                chunks = _append_piece(chunks, paragraph, max_chars)
                continue
            sentence_parts = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in [s.strip() for s in sentence_parts if s.strip()]:
                if len(sentence) <= max_chars:
                    chunks = _append_piece(chunks, sentence, max_chars)
                    continue
                for hard_piece in _hard_split(sentence, max_chars):
                    chunks = _append_piece(chunks, hard_piece, max_chars)
        return chunks

    def _synthesize_script_audio(self, *, script: str, output_path: Path) -> None:
        if self._tts is None:
            api_key_var = (
                "ELEVENLABS_API_KEY"
                if self._tts_provider_name == "elevenlabs"
                else "GEMINI_API_KEY"
                if self._tts_provider_name == "gemini"
                else "the provider API key"
            )
            raise RuntimeError(f"{self._tts_provider_name} TTS is not configured. Set {api_key_var} in your environment.")

        max_chars = self._tts_chunk_chars()
        chunks = self._split_script_for_tts(script=script, max_chars=max_chars)
        if not chunks:
            raise RuntimeError("Cannot synthesize audio from an empty script.")
        logger.info(
            "Synthesizing %s chunk(s) with %s provider (max %s chars/chunk).",
            len(chunks),
            self._tts.provider_name,
            max_chars,
        )
        if len(chunks) == 1:
            self._tts.synthesize(text=chunks[0], output_path=output_path)
            return

        with tempfile.TemporaryDirectory(prefix="paperpods-tts-chunks-") as temp_dir:
            temp_paths: list[Path] = []
            for idx, chunk_text in enumerate(chunks, start=1):
                chunk_path = Path(temp_dir) / f"chunk_{idx:04d}.{self._tts.audio_extension}"
                logger.info("Synthesizing chunk %s/%s", idx, len(chunks))
                self._tts.synthesize(text=chunk_text, output_path=chunk_path)
                temp_paths.append(chunk_path)
            _merge_audio_chunks(paths=temp_paths, output_path=output_path, extension=self._tts.audio_extension)

    def process(
        self,
        *,
        pdf_path: Path,
        title: str | None = None,
        style: str = "single-narrator-15-20min",
        formula_policy: str = "conceptual-plus-key-variables",
        generate_audio: bool = False,
        script_length_mode: str = "capped",
        force_rerun: bool = False,
    ) -> ProcessResult:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Starting processing for %s", pdf_path)
        pdf_hash = _hash_file(pdf_path)
        cached_job = None
        if not force_rerun:
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

        cached_script_job = None
        if not force_rerun:
            cached_script_job = self._store.find_cached_job(
                pdf_sha256=pdf_hash,
                require_audio=False,
            )
        if generate_audio and cached_script_job and cached_script_job.script_path:
            logger.info("Generating missing audio from cached script for job %s", cached_script_job.id)
            audio_path = self._artifact_path(
                directory=self._audio_dir,
                title=cached_script_job.title,
                pdf_path=Path(cached_script_job.pdf_path),
                extension=self._tts.audio_extension,
            )
            script = Path(cached_script_job.script_path).read_text(encoding="utf-8")
            self._synthesize_script_audio(script=script, output_path=audio_path)
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
            script_length_mode=script_length_mode,
            tts_chunk_chars=self._tts_chunk_chars() if generate_audio else None,
        )
        try:
            self._store.update_job(job_id, status="running", title=title)
            logger.info("Extracting PDF text for %s", pdf_path)
            paper = extract_pdf_text(pdf_path)
            paper_title = title or paper.title
            self._store.update_job(job_id, status="running", title=paper_title)
            system_prompt, user_prompt = build_script_messages(
                paper,
                title_override=title,
                style=style,
                formula_policy=formula_policy,
                script_length_mode=script_length_mode,
            )
            logger.info("Sending script generation request to %s/%s", self._llm.provider_name, self._llm.model_name)
            script = self._llm.generate_script(system_prompt=system_prompt, user_prompt=user_prompt)
            script = self._normalize_script_with_policy(
                script=script,
                original_prompt=user_prompt,
                script_length_mode=script_length_mode,
            )
            logger.info("Generated script length: %s chars", len(script))

            title_for_artifacts = title or paper.title
            script_path = self._artifact_path(
                directory=self._script_dir,
                title=title_for_artifacts,
                pdf_path=pdf_path,
                extension="txt",
            )
            script_path.write_text(script, encoding="utf-8")

            audio_path: Path | None = None
            if generate_audio:
                logger.info("Waiting for %s audio generation for job %s", self._tts.provider_name, job_id)
                audio_path = self._artifact_path(
                    directory=self._audio_dir,
                    title=title_for_artifacts,
                    pdf_path=pdf_path,
                    extension=self._tts.audio_extension,
                )
                self._synthesize_script_audio(script=script, output_path=audio_path)

            self._store.update_job(
                job_id,
                status="completed",
                title=title_for_artifacts,
                script_path=script_path,
                audio_path=audio_path,
                error=None,
            )
            return ProcessResult(job_id=job_id, script_path=script_path, audio_path=audio_path, cached=False)
        except Exception as exc:
            self._store.update_job(job_id, status="failed", error=str(exc))
            raise


def _slugify_title(value: str | None) -> str:
    cleaned = (value or "paper").strip()
    if not cleaned:
        cleaned = "paper"
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    slug = re.sub(r"_+", "_", normalized).strip("_").lower()
    return slug[:80] or "paper"


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _append_piece(chunks: list[str], piece: str, max_chars: int) -> list[str]:
    if not piece:
        return chunks
    if not chunks:
        return [piece]
    combined = f"{chunks[-1]}\n\n{piece}" if "\n" in piece else f"{chunks[-1]} {piece}"
    if len(combined) <= max_chars:
        chunks[-1] = combined
    else:
        chunks.append(piece)
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text[:max_chars]]
    parts: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        parts.append(current)
        if len(word) <= max_chars:
            current = word
            continue
        for i in range(0, len(word), max_chars):
            chunk = word[i : i + max_chars]
            if i + max_chars >= len(word):
                current = chunk
            else:
                parts.append(chunk)
    if current:
        parts.append(current)
    return parts


def _merge_audio_chunks(*, paths: list[Path], output_path: Path, extension: str) -> None:
    if not paths:
        raise RuntimeError("No audio chunks were produced.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_ext = extension.strip(".").lower()
    if normalized_ext == "wav":
        _merge_wav_chunks(paths=paths, output_path=output_path)
        return
    if normalized_ext == "mp3":
        _merge_mp3_chunks_with_ffmpeg(paths=paths, output_path=output_path)
        return
    raise RuntimeError(f"Unsupported audio extension for merge: {extension}")


def _merge_wav_chunks(*, paths: list[Path], output_path: Path) -> None:
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for path in paths[1:]:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getparams()[:4] != params[:4]:
                raise RuntimeError("WAV chunks use incompatible audio parameters; cannot merge safely.")
            frames.append(wav_file.readframes(wav_file.getnframes()))

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setparams(params)
        for frame_data in frames:
            output_wav.writeframes(frame_data)


def _merge_mp3_chunks_with_ffmpeg(*, paths: list[Path], output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "Merging MP3 chunks requires ffmpeg, but it was not found in PATH."
        )

    with tempfile.TemporaryDirectory(prefix="paperpods-merge-mp3-") as temp_dir:
        concat_file = Path(temp_dir) / "concat.txt"
        concat_lines = []
        for path in paths:
            escaped = str(path).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            details = f" Details: {stderr[:500]}" if stderr else ""
            raise RuntimeError(f"ffmpeg failed to merge MP3 chunks.{details}") from exc
