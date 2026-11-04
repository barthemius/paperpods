# PaperPods

PaperPods is an open-source, **CLI-first** tool that turns scientific PDFs into expert-level podcast scripts and optional ElevenLabs audio.

## What it does

- Extracts paper text from PDF files
- Uses an LLM to produce a 15–20 minute expert narration script
- Explains equation intent conceptually and defines key variables
- Optionally renders MP3 audio via ElevenLabs
- Stores local job history and artifacts for reproducibility
- Reuses cached results for identical paper + generation settings

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e ".[dev]"
```

3. Create local env file and fill credentials:

```bash
cp .env.example .env
# then edit .env with your keys
```

4. Run:

```bash
paperpods process ./my-paper.pdf --audio
```

## CLI commands

- `paperpods process <pdf_path> [--title ...] [--audio]`
- `paperpods status <job_id>`
- `paperpods list-jobs`
- `paperpods show-script <job_id> [--print-content]`

## Environment variables

- `PAPERPODS_DATA_DIR` (default: `~/.paperpods`)
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `ELEVENLABS_API_KEY` (required for audio generation)
- `ELEVENLABS_VOICE_ID` (default: `EXAVITQu4vr4xnSDxMaL`)
- `ELEVENLABS_MODEL_ID` (default: `eleven_multilingual_v2`)

PaperPods automatically loads these from `.env` in the project directory.

## Caching behavior

PaperPods fingerprints each PDF and reuses existing script/audio artifacts from `~/.paperpods` when the same paper is requested again with the same model/style/formula settings.

## Notes

- Current implementation is CLI-first and structured for future TUI extension.
- Provider abstraction is in place; OpenAI is the first concrete LLM backend.
