from __future__ import annotations

import logging
from pathlib import Path

import typer

from .app import build_pipeline
from .config import load_settings
from .storage import JobStore

logger = logging.getLogger("paperpods")
app = typer.Typer(help="PaperPods: scientific-paper to podcast-script/audio CLI.")


@app.command()
def process(
    pdf_path: Path = typer.Argument(..., help="Path to a scientific PDF."),
    title: str | None = typer.Option(None, help="Optional custom paper title."),
    audio: bool = typer.Option(False, "--audio", help="Generate ElevenLabs MP3 output."),
    style: str = typer.Option(
        "single-narrator-15-20min",
        help="Script style preset (single-narrator-15-20min, single-narrator-8-12min, two-host-dialogue-15-20min).",
    ),
    formula_policy: str = typer.Option(
        "conceptual-plus-key-variables",
        help="Formula policy (conceptual-plus-key-variables, high-level-only, detailed-derivation-summary).",
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress high-level progress logs."),
) -> None:
    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")
    logger.setLevel(log_level)

    settings = load_settings()
    logger.info("Loading PaperPods settings from environment.")
    pipeline = build_pipeline(settings)
    try:
        result = pipeline.process(
            pdf_path=pdf_path,
            title=title,
            style=style,
            formula_policy=formula_policy,
            generate_audio=audio,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.secho(f"Error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if result.cached:
        typer.echo("Cache hit: reusing existing artifacts.")
    typer.echo(f"Job: {result.job_id}")
    typer.echo(f"Script: {result.script_path}")
    if result.audio_path:
        typer.echo(f"Audio: {result.audio_path}")


@app.command("status")
def status_command(job_id: str = typer.Argument(..., help="Job ID.")) -> None:
    settings = load_settings()
    store = JobStore(settings.data_dir / "paperpods.db")
    job = store.get_job(job_id)
    if job is None:
        raise typer.BadParameter(f"No job found with id={job_id}")

    typer.echo(f"id={job.id}")
    typer.echo(f"status={job.status}")
    typer.echo(f"title={job.title or '-'}")
    typer.echo(f"pdf={job.pdf_path}")
    typer.echo(f"script={job.script_path or '-'}")
    typer.echo(f"audio={job.audio_path or '-'}")
    typer.echo(f"provider={job.provider}")
    typer.echo(f"model={job.model}")
    typer.echo(f"updated_at={job.updated_at.isoformat()}")
    if job.error:
        typer.echo(f"error={job.error}")


@app.command("list-jobs")
def list_jobs(limit: int = typer.Option(20, min=1, max=200, help="Maximum jobs to show.")) -> None:
    settings = load_settings()
    store = JobStore(settings.data_dir / "paperpods.db")
    jobs = store.list_jobs(limit=limit)
    if not jobs:
        typer.echo("No jobs yet.")
        return

    for job in jobs:
        typer.echo(
            f"{job.id} | {job.status} | {job.title or Path(job.pdf_path).name} | "
            f"script={job.script_path or '-'} | audio={job.audio_path or '-'}"
        )


@app.command("show-script")
def show_script(
    job_id: str = typer.Argument(..., help="Job ID."),
    print_content: bool = typer.Option(False, "--print-content", help="Print the script text to stdout."),
) -> None:
    settings = load_settings()
    store = JobStore(settings.data_dir / "paperpods.db")
    job = store.get_job(job_id)
    if job is None:
        raise typer.BadParameter(f"No job found with id={job_id}")
    if not job.script_path:
        raise typer.BadParameter(f"Job {job_id} has no generated script yet.")

    script_path = Path(job.script_path)
    typer.echo(str(script_path))
    if print_content:
        typer.echo("")
        typer.echo(script_path.read_text(encoding="utf-8"))


@app.command("tui")
def tui() -> None:
    """Open a lightweight terminal UI to browse generated scripts and audio."""
    from .tui import PaperPodsBrowserApp

    settings = load_settings()
    app_instance = PaperPodsBrowserApp(data_dir=settings.data_dir)
    app_instance.run()


if __name__ == "__main__":
    app()
