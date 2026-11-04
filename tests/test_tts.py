from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from paperpods.tts import elevenlabs as elevenlabs_module
from paperpods.tts.elevenlabs import ElevenLabsTTS


def test_elevenlabs_402_is_reported_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise_402(_req):
        raise HTTPError(
            url="https://api.elevenlabs.io/v1/text-to-speech/test",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=BytesIO(b'{"detail":"insufficient credits"}'),
        )

    monkeypatch.setattr(elevenlabs_module.request, "urlopen", _raise_402)
    tts = ElevenLabsTTS(api_key="k", voice_id="v", model_id="m")

    with pytest.raises(RuntimeError, match="HTTP 402"):
        tts.synthesize(text="hello", output_path=tmp_path / "out.mp3")

