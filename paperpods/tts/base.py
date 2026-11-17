from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TTSProvider(Protocol):
    provider_name: str
    audio_extension: str

    def synthesize(self, *, text: str, output_path: Path) -> None:
        ...
