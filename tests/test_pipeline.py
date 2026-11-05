from pathlib import Path

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


class FakeTTS:
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


def test_pipeline_trims_script_to_max_characters() -> None:
    pipeline = PaperPodsPipeline(data_dir=Path("/tmp/paperpods-test"), store=JobStore(Path("/tmp/paperpods-test.db")), llm=FakeLLM(), tts=None)
    long_script = ("A " * 20_000) + "B"

    trimmed = pipeline._normalize_script_length(long_script)

    assert len(trimmed) <= 10_000
    assert trimmed.endswith(".") or trimmed.endswith("B") or trimmed.endswith(" ") is False


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
