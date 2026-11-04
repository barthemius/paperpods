from __future__ import annotations

from openai import OpenAI


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)

    def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not response.choices:
            raise RuntimeError("OpenAI returned no completion choices.")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned empty completion content.")
        return content.strip()

