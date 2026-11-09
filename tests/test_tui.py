from pathlib import Path

from paperpods.tui import PaperPodsBrowserApp


def test_clean_hanging_records_is_blocked_while_processing(tmp_path: Path, monkeypatch) -> None:
    app = PaperPodsBrowserApp(data_dir=tmp_path)
    app.is_processing = True
    notifications: list[tuple[str, str | None]] = []

    def fake_notify(message: str, *, severity: str | None = None) -> None:
        notifications.append((message, severity))

    monkeypatch.setattr(app, "notify", fake_notify)
    app.clean_hanging_records()

    assert notifications == [("Cleanup is disabled while processing is active.", "warning")]


def test_process_selected_pdf_launches_worker_with_callable(tmp_path: Path, monkeypatch) -> None:
    app = PaperPodsBrowserApp(data_dir=tmp_path)
    app.selected_pdf_path = Path("/tmp/sample.pdf")
    calls: dict[str, object] = {}

    monkeypatch.setattr(app, "_set_processing_state", lambda _flag: None)
    monkeypatch.setattr(app, "_set_progress", lambda _value, *, message: None)

    def fake_process_worker(*, pdf_path: Path, generate_audio: bool) -> None:
        calls["worker"] = (pdf_path, generate_audio)

    def fake_run_worker(work, **kwargs) -> None:  # type: ignore[no-untyped-def]
        calls["run_kwargs"] = kwargs
        work()

    monkeypatch.setattr(app, "_process_pdf_worker", fake_process_worker)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)

    app.process_selected_pdf(generate_audio=True)

    assert calls["worker"] == (Path("/tmp/sample.pdf"), True)
    assert calls["run_kwargs"] == {"thread": True}
