from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
        ...

