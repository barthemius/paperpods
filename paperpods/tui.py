from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Footer, Header, ListItem, ListView, ProgressBar, Static

from .app import build_pipeline
from .config import load_settings
from .storage import JobStore


@dataclass(slots=True)
class LibraryEntry:
    title: str
    status: str
    pdf_path: str
    script_path: str | None = None
    audio_path: str | None = None


class PaperPodsBrowserApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #layout {
        height: 1fr;
    }

    #sidebar {
        width: 42;
        min-width: 30;
        border: solid #4a4a4a;
        padding: 1 1;
    }

    #browser {
        height: 1fr;
        min-height: 10;
        margin-top: 1;
        border: solid #4a4a4a;
    }

    #detail-pane {
        width: 1fr;
        border: solid #4a4a4a;
        padding: 1 1;
    }

    #progress-panel {
        height: auto;
        margin-top: 1;
        padding: 1 1;
        border: solid #4a4a4a;
    }

    #progress {
        width: 1fr;
        margin-top: 1;
    }

    .section-title {
        color: #c8d4ff;
        text-style: bold;
        margin: 0 0 1 0;
    }

    #details {
        width: 1fr;
        height: 1fr;
        padding: 1 1;
        background: #1d1d1d;
    }

    #actions {
        height: auto;
        margin-top: 1;
        layout: horizontal;
        width: 1fr;
        padding: 0 0;
    }

    #actions > Button {
        margin-right: 1;
    }

    #actions > Button:last-child {
        margin-right: 0;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("o", "open_selected", "Open selection"),
        ("c", "clean_hanging", "Clean hangs"),
    ]

    def __init__(self, *, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.entries: list[LibraryEntry] = []
        self.selected_entry: LibraryEntry | None = None
        self.selected_pdf_path: Path | None = None
        self.is_processing = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="layout"):
            with Vertical(id="sidebar"):
                yield Static("Processed papers", classes="section-title")
                yield ListView(id="jobs")
                yield Static("Browse PDFs", classes="section-title")
                yield DirectoryTree(str(Path.home()), id="browser")
            with Vertical(id="detail-pane"):
                yield Static("Details", classes="section-title")
                yield Static("Select a paper to inspect its files.", id="details")
                with Vertical(id="progress-panel"):
                    yield Static("Progress", classes="section-title")
                    yield ProgressBar(total=100, id="progress")
                    yield Static("Idle", id="progress-status")
                with Horizontal(id="actions"):
                    yield Button("Open script", id="open_script", variant="primary")
                    yield Button("Open audio", id="open_audio")
                    yield Button("Process PDF", id="process_pdf", variant="success")
                    yield Button("Process + audio", id="process_audio", variant="success")
                    yield Button("Clean hangs", id="clean_hanging", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        store = JobStore(self.data_dir / "paperpods.db")
        jobs = store.list_jobs(limit=100)
        self.entries = []
        for job in jobs:
            title = job.title or Path(job.pdf_path).stem.replace("_", " ").replace("-", " ")
            title = " ".join(title.split())
            self.entries.append(
                LibraryEntry(
                    title=title,
                    status=job.status,
                    pdf_path=job.pdf_path,
                    script_path=job.script_path,
                    audio_path=job.audio_path,
                )
            )

        list_view = self.query_one(ListView)
        list_view.clear()
        if not self.entries:
            list_view.append(ListItem(Static("No processed papers yet.")))
            self.selected_entry = None
            self.query_one("#details", Static).update("No papers found. Run `paperpods process` to generate one.")
            return

        for entry in self.entries:
            summary = f"{entry.title} [{entry.status}]"
            list_view.append(ListItem(Static(summary)))

        self.selected_entry = self.entries[0]
        self.update_details(self.selected_entry)

    def update_details(self, entry: LibraryEntry | None) -> None:
        if entry is None:
            selected_pdf = self.selected_pdf_path or "—"
            self.query_one("#details", Static).update(f"Selected PDF: {selected_pdf}\nNo processed paper selected.")
            return

        script_text = entry.script_path or "—"
        audio_text = entry.audio_path or "—"
        selected_pdf = self.selected_pdf_path or entry.pdf_path
        self.query_one("#details", Static).update(
            "\n".join(
                [
                    f"Title: {entry.title}",
                    f"Status: {entry.status}",
                    f"PDF: {entry.pdf_path}",
                    f"Selected PDF: {selected_pdf}",
                    f"Script: {script_text}",
                    f"Audio: {audio_text}",
                ]
            )
        )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if not self.entries:
            return
        list_view = event.list_view
        index = list_view.index
        if index is None:
            return
        if 0 <= index < len(self.entries):
            self.selected_entry = self.entries[index]
            self.update_details(self.selected_entry)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self.entries:
            return
        index = event.index
        if 0 <= index < len(self.entries):
            self.selected_entry = self.entries[index]
            self.update_details(self.selected_entry)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "clean_hanging":
            self.clean_hanging_records()
            return
        if button_id == "process_pdf":
            self.process_selected_pdf(generate_audio=False)
            return
        if button_id == "process_audio":
            self.process_selected_pdf(generate_audio=True)
            return
        if self.selected_entry is None:
            return
        if button_id == "open_script" and self.selected_entry.script_path:
            self.open_file(Path(self.selected_entry.script_path))
        elif button_id == "open_audio" and self.selected_entry.audio_path:
            self.open_file(Path(self.selected_entry.audio_path))

    def _set_processing_state(self, is_processing: bool) -> None:
        self.is_processing = is_processing
        for button_id in ("process_pdf", "process_audio", "open_script", "open_audio", "clean_hanging"):
            button = self.query_one(f"#{button_id}", Button)
            button.disabled = is_processing

    def _set_progress(self, value: float, *, message: str) -> None:
        progress_widget = self.query_one("#progress", ProgressBar)
        progress_widget.update(progress=value)
        self.query_one("#progress-status", Static).update(message)

    def action_refresh(self) -> None:
        self.refresh_jobs()

    def action_clean_hanging(self) -> None:
        self.clean_hanging_records()

    def action_open_selected(self) -> None:
        if self.selected_entry is None:
            return
        if self.selected_entry.audio_path:
            self.open_file(Path(self.selected_entry.audio_path))
        elif self.selected_entry.script_path:
            self.open_file(Path(self.selected_entry.script_path))

    def clean_hanging_records(self) -> None:
        store = JobStore(self.data_dir / "paperpods.db")
        removed = store.delete_hanging_jobs()
        self.notify(f"Removed {removed} hanging records.")
        self.refresh_jobs()

    def process_selected_pdf(self, *, generate_audio: bool) -> None:
        if self.selected_pdf_path is None:
            self.notify("Select a PDF in the file browser first.", severity="error")
            return
        if self.selected_pdf_path.suffix.lower() != ".pdf":
            self.notify("The selected file is not a PDF.", severity="error")
            return
        pdf_path = self.selected_pdf_path

        self._set_processing_state(True)
        self._set_progress(5, message="Preparing to process…")
        self.run_worker(self._process_pdf_worker, pdf_path=pdf_path, generate_audio=generate_audio)

    def _process_pdf_worker(self, *, pdf_path: Path, generate_audio: bool) -> None:
        try:
            self.call_from_thread(self._set_progress, 15, message="Loading settings…")
            settings = load_settings()
            self.call_from_thread(self._set_progress, 35, message="Building pipeline…")
            pipeline = build_pipeline(settings)
            self.call_from_thread(self._set_progress, 55, message="Processing document…")
            result = pipeline.process(pdf_path=pdf_path, generate_audio=generate_audio)
            self.call_from_thread(self._set_progress, 100, message="Finalizing output…")
            self.call_from_thread(self.notify, f"Processing {pdf_path.name}; job={result.job_id}")
            self.call_from_thread(self.refresh_jobs)
        except Exception as exc:  # pragma: no cover - UI feedback path
            self.call_from_thread(self.notify, f"Could not process PDF: {exc}", severity="error")
        finally:
            self.call_from_thread(self._set_processing_state, False)
            self.call_from_thread(self._set_progress, 0, message="Idle")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        pdf_path = Path(event.path)
        if pdf_path.suffix.lower() != ".pdf":
            self.selected_pdf_path = None
            self.notify("Only PDF files can be processed.", severity="warning")
            self.update_details(self.selected_entry)
            return
        self.selected_pdf_path = pdf_path
        self.notify(f"Selected PDF: {pdf_path}")
        self.update_details(self.selected_entry)

    def open_file(self, path: Path) -> None:
        if not path.exists():
            self.notify(f"File not found: {path}", severity="error")
            return

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            self.notify(f"Could not open {path}: {exc}", severity="error")
