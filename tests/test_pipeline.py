from datetime import UTC, datetime, timedelta
from pathlib import Path
import wave

import pytest

from paperpods.pdf_extract import ExtractedPaper
from paperpods.pipeline import PaperPodsPipeline
from paperpods.storage import JobStore


class FakeLLM:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls = 0

    def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self._fail:
            raise RuntimeError("llm failed")
        return "Generated expert script."


class FakeLongLLM(FakeLLM):
    def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        _ = (system_prompt, user_prompt)
        return "Long sentence. " * 2_000


class FakeTTS:
    provider_name = "fake"
    audio_extension = "mp3"

    def synthesize(self, *, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")


def test_pipeline_process_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=["x = y + z"],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    store = JobStore(tmp_path / "paperpods.db")
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=FakeLLM(), tts=FakeTTS())

    result = pipeline.process(pdf_path=pdf_path, generate_audio=True)
    job = store.get_job(result.job_id)

    assert job is not None
    assert job.status == "completed"
    assert result.script_path.exists()
    assert len(result.script_path.read_text(encoding="utf-8")) <= 10_000
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.cached is False
    assert job.script_length_mode == "capped"


def test_pipeline_process_failure_updates_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=[],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    store = JobStore(tmp_path / "paperpods.db")
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=FakeLLM(fail=True), tts=None)

    with pytest.raises(RuntimeError, match="llm failed"):
        pipeline.process(pdf_path=pdf_path, generate_audio=False)

    jobs = store.list_jobs(limit=1)
    assert jobs[0].status == "failed"
    assert "llm failed" in (jobs[0].error or "")


def test_pipeline_marks_job_running_before_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paperpods import pipeline as pipeline_module

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    store = JobStore(tmp_path / "paperpods.db")
    observed_status: dict[str, str | None] = {"status": None}

    def fake_extract(_path: Path) -> ExtractedPaper:
        jobs = store.list_jobs(limit=1)
        observed_status["status"] = jobs[0].status if jobs else None
        return ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=[],
        )

    monkeypatch.setattr(pipeline_module, "extract_pdf_text", fake_extract)
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=FakeLLM(), tts=None)

    pipeline.process(pdf_path=pdf_path, generate_audio=False)

    assert observed_status["status"] == "running"


def test_pipeline_trims_script_to_max_characters() -> None:
    pipeline = PaperPodsPipeline(data_dir=Path("/tmp/paperpods-test"), store=JobStore(Path("/tmp/paperpods-test.db")), llm=FakeLLM(), tts=None)
    long_script = ("A " * 20_000) + "B"

    trimmed = pipeline._normalize_script_length(long_script)

    assert len(trimmed) <= 10_000
    assert trimmed == trimmed.rstrip()


def test_delete_hanging_jobs_removes_incomplete_records(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "paperpods.db")

    good_id = store.create_job(
        pdf_path=tmp_path / "good.pdf",
        pdf_sha256="goodhash",
        title="Good",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )
    store.update_job(good_id, status="completed", script_path=tmp_path / "good.txt", audio_path=tmp_path / "good.mp3")

    dangling_id_1 = store.create_job(
        pdf_path=tmp_path / "dangling-1.pdf",
        pdf_sha256="danglinghash1",
        title="Dangling 1",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )
    store.update_job(dangling_id_1, status="failed")

    stale_queued_id = store.create_job(
        pdf_path=tmp_path / "queued-stale.pdf",
        pdf_sha256="queuedstalehash",
        title="Queued stale",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(minutes=30)).isoformat(), stale_queued_id),
        )

    dangling_id_2 = store.create_job(
        pdf_path=tmp_path / "dangling-2.pdf",
        pdf_sha256="danglinghash2",
        title="Dangling 2",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )
    store.update_job(dangling_id_2, status="running")

    fresh_queued_id = store.create_job(
        pdf_path=tmp_path / "queued-fresh.pdf",
        pdf_sha256="queuedfreshhash",
        title="Queued fresh",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )

    remaining = store.list_jobs(limit=50)
    assert len(remaining) == 5

    removed = store.delete_hanging_jobs()

    assert removed == 1
    assert store.get_job(good_id) is not None
    assert store.get_job(dangling_id_1) is not None
    assert store.get_job(dangling_id_2) is not None
    assert store.get_job(stale_queued_id) is None
    assert store.get_job(fresh_queued_id) is not None


def test_clear_cache_for_pdf_removes_jobs_and_artifacts(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "paperpods.db")
    script_path = tmp_path / "paper.txt"
    audio_path = tmp_path / "paper.mp3"
    script_path.write_text("script", encoding="utf-8")
    audio_path.write_bytes(b"audio")

    job_id = store.create_job(
        pdf_path=tmp_path / "paper.pdf",
        pdf_sha256="paperhash",
        title="Paper",
        provider="fake",
        model="fake",
        style="single",
        formula_policy="conceptual",
    )
    store.update_job(
        job_id,
        status="completed",
        script_path=script_path,
        audio_path=audio_path,
    )

    removed = store.clear_cache_for_pdf(pdf_sha256="paperhash")

    assert removed == 1
    assert store.get_job(job_id) is None
    assert not script_path.exists()
    assert not audio_path.exists()


def test_pipeline_reuses_cached_script_without_regenerating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=[],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 paper")
    store = JobStore(tmp_path / "paperpods.db")
    llm = FakeLLM()
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=llm, tts=FakeTTS())

    first = pipeline.process(pdf_path=pdf_path, generate_audio=False)
    second = pipeline.process(pdf_path=pdf_path, generate_audio=False)

    assert llm.calls == 1
    assert second.cached is True
    assert second.job_id == first.job_id
    assert second.script_path == first.script_path


def test_pipeline_force_rerun_bypasses_cached_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=[],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 paper")
    store = JobStore(tmp_path / "paperpods.db")
    llm = FakeLLM()
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=llm, tts=None)

    first = pipeline.process(pdf_path=pdf_path, generate_audio=False)
    second = pipeline.process(pdf_path=pdf_path, generate_audio=False, force_rerun=True)

    assert llm.calls == 2
    assert first.cached is False
    assert second.cached is False
    assert second.job_id != first.job_id


def test_pipeline_generates_missing_audio_from_cached_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Paper",
            text="Paper content",
            formula_snippets=[],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 paper")
    store = JobStore(tmp_path / "paperpods.db")
    llm = FakeLLM()
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=llm, tts=FakeTTS())

    script_only = pipeline.process(pdf_path=pdf_path, generate_audio=False)
    with_audio = pipeline.process(pdf_path=pdf_path, generate_audio=True)

    assert llm.calls == 1
    assert with_audio.cached is True
    assert with_audio.job_id == script_only.job_id
    assert with_audio.audio_path is not None
    assert with_audio.audio_path.exists()


def test_pipeline_uses_human_readable_artifact_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="A Deep Study of Neural Dynamics",
            text="Paper content",
            formula_snippets=[],
        ),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    store = JobStore(tmp_path / "paperpods.db")
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=FakeLLM(), tts=FakeTTS())

    result = pipeline.process(pdf_path=pdf_path, generate_audio=True)

    assert result.script_path.name == "a_deep_study_of_neural_dynamics.txt"
    assert result.audio_path is not None
    assert result.audio_path.name == "a_deep_study_of_neural_dynamics.mp3"


def test_pipeline_unlimited_mode_keeps_long_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paperpods import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "extract_pdf_text",
        lambda _path: ExtractedPaper(
            title="Very Long Paper",
            text="Paper content",
            formula_snippets=[],
        ),
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    store = JobStore(tmp_path / "paperpods.db")
    pipeline = PaperPodsPipeline(data_dir=tmp_path, store=store, llm=FakeLongLLM(), tts=None)

    result = pipeline.process(pdf_path=pdf_path, generate_audio=False, script_length_mode="unlimited")

    script_content = result.script_path.read_text(encoding="utf-8")
    assert len(script_content) > 10_000
    job = store.get_job(result.job_id)
    assert job is not None
    assert job.script_length_mode == "unlimited"


def test_pipeline_synthesizes_multi_chunk_wav_audio(tmp_path: Path) -> None:
    class FakeWavTTS:
        provider_name = "fake"
        audio_extension = "wav"

        def synthesize(self, *, text: str, output_path: Path) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes((b"\x00\x00") * max(1, len(text)))

    pipeline = PaperPodsPipeline(
        data_dir=tmp_path,
        store=JobStore(tmp_path / "paperpods.db"),
        llm=FakeLLM(),
        tts=FakeWavTTS(),
        tts_provider_name="fake",
    )
    pipeline._tts_chunk_chars = lambda: 80  # force chunking for test
    text = "Chunk me. " * 200
    output_path = tmp_path / "merged.wav"
    pipeline._synthesize_script_audio(script=text, output_path=output_path)

    assert output_path.exists()
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnframes() > 0
