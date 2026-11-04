from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    pdf_path: str
    pdf_sha256: str | None
    title: str | None
    script_path: str | None
    audio_path: str | None
    error: str | None
    provider: str
    model: str
    style: str | None
    formula_policy: str | None
    created_at: datetime
    updated_at: datetime
