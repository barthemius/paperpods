from __future__ import annotations

RP_BRITISH_ENGLISH_TAG = "<Speak in a natural, clear Received Pronunciation British English>"


class GoogleGeminiProvider:
    provider_name = "gemini"

    def __init__(self, *, api_key: str, model_name: str) -> None:
        genai, types = _import_genai()
        self.model_name = model_name
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def generate_script(self, *, system_prompt: str, user_prompt: str) -> str:
        prompt_with_tag = _prepend_rp_tag(user_prompt)
        config = self._types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system_prompt,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt_with_tag,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - provider/network path
            raise RuntimeError(f"Google Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Google Gemini returned empty completion content.")
        return text


def _prepend_rp_tag(prompt: str) -> str:
    if prompt.lstrip().startswith(RP_BRITISH_ENGLISH_TAG):
        return prompt
    return f"{RP_BRITISH_ENGLISH_TAG}\n\n{prompt}"


def _import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dependency missing path
        raise RuntimeError(
            "Google Gemini provider requires the `google-genai` package. "
            "Install dependencies with `uv sync` or `pip install google-genai`."
        ) from exc
    return genai, types
