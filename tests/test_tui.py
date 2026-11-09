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
