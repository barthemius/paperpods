from types import SimpleNamespace

import pytest

from paperpods.llm import google_gemini_provider as gemini_module
from paperpods.llm.google_gemini_provider import GoogleGeminiProvider, RP_BRITISH_ENGLISH_TAG


def test_google_gemini_provider_prepends_rp_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeGenerateConfig:
        def __init__(self, **kwargs) -> None:
            captured["config"] = kwargs

    class FakeTypes:
        GenerateContentConfig = FakeGenerateConfig

    class FakeModels:
        @staticmethod
        def generate_content(*, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config_obj"] = config
            return SimpleNamespace(text="ok")

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.models = FakeModels()

    class FakeGenAI:
        Client = FakeClient

    monkeypatch.setattr(gemini_module, "_import_genai", lambda: (FakeGenAI, FakeTypes))
    provider = GoogleGeminiProvider(api_key="k", model_name="gemini-2.5-flash")
    provider.generate_script(system_prompt="system", user_prompt="User prompt body.")

    assert captured["model"] == "gemini-2.5-flash"
    assert str(captured["contents"]).startswith(RP_BRITISH_ENGLISH_TAG)
    assert str(captured["contents"]).endswith("User prompt body.")


def test_google_gemini_provider_does_not_duplicate_rp_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeGenerateConfig:
        def __init__(self, **_kwargs) -> None:
            pass

    class FakeTypes:
        GenerateContentConfig = FakeGenerateConfig

    class FakeModels:
        @staticmethod
        def generate_content(*, model, contents, config):
            _ = (model, config)
            captured["contents"] = contents
            return SimpleNamespace(text="ok")

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.models = FakeModels()

    class FakeGenAI:
        Client = FakeClient

    monkeypatch.setattr(gemini_module, "_import_genai", lambda: (FakeGenAI, FakeTypes))
    provider = GoogleGeminiProvider(api_key="k", model_name="gemini-2.5-flash")
    provider.generate_script(
        system_prompt="system",
        user_prompt=f"{RP_BRITISH_ENGLISH_TAG}\n\nAlready tagged prompt.",
    )

    assert str(captured["contents"]).count(RP_BRITISH_ENGLISH_TAG) == 1
