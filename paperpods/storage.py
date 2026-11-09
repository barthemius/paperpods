from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import JobRecord


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    pdf_sha256 TEXT,
                    title TEXT,
                    script_path TEXT,
                    audio_path TEXT,
                    error TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    style TEXT,
                    formula_policy TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "pdf_sha256" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN pdf_sha256 TEXT")
            if "style" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN style TEXT")
            if "formula_policy" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN formula_policy TEXT")

    def create_job(
        self,
        *,
        pdf_path: Path,
        pdf_sha256: str,
        title: str | None,
        provider: str,
        model: str,
        style: str,
        formula_policy: str,
    ) -> str:
        now = _now_iso()
        job_id = uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, pdf_path, pdf_sha256, title, script_path, audio_path, error, provider, model, style, formula_policy, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, str(pdf_path), pdf_sha256, title, provider, model, style, formula_policy, now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        title: str | None = None,
        script_path: Path | None = None,
        audio_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    title = COALESCE(?, title),
                    script_path = COALESCE(?, script_path),
                    audio_path = COALESCE(?, audio_path),
                    error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    title,
                    str(script_path) if script_path else None,
                    str(audio_path) if audio_path else None,
                    error,
                    _now_iso(),
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            return _row_to_job(row)

    def list_jobs(self, *, limit: int = 20) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [_row_to_job(row) for row in rows]

    def find_cached_job(
        self,
        *,
        pdf_sha256: str,
        require_audio: bool,
    ) -> JobRecord | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'completed'
                  AND pdf_sha256 = ?
                  AND script_path IS NOT NULL
                ORDER BY created_at DESC
                """,
                (pdf_sha256,),
            ).fetchall()

        for row in rows:
            job = _row_to_job(row)
            if not job.script_path or not Path(job.script_path).exists():
                continue
            if require_audio and (not job.audio_path or not Path(job.audio_path).exists()):
                continue
            return job
        return None

    def delete_hanging_jobs(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM jobs
                WHERE script_path IS NULL
                  AND audio_path IS NULL
                  AND status IN ('queued', 'running')
                """
            )
            return cursor.rowcount


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        status=row["status"],
        pdf_path=row["pdf_path"],
        pdf_sha256=row["pdf_sha256"] if "pdf_sha256" in row.keys() else None,
        title=row["title"],
        script_path=row["script_path"],
        audio_path=row["audio_path"],
        error=row["error"],
        provider=row["provider"],
        model=row["model"],
        style=row["style"] if "style" in row.keys() else None,
        formula_policy=row["formula_policy"] if "formula_policy" in row.keys() else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
