# whisp

[Русский](README.md) · **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)](#requirements)

One command turns a meeting recording into a speaker-labelled text transcript.
Transcription runs entirely on your Mac — no audio or video is uploaded anywhere.

```bash
whisp "Team call.m4a"
```

## Why

Call recordings pile up, and nobody ever listens to them again. Hosted transcription
services want the whole file uploaded, which is not always acceptable for client calls
and internal discussions.

`whisp` is a thin wrapper around [WhisperX](https://github.com/m-bain/whisperX) that
covers this locally:

- **Transcription** of audio and video (`.mp3`, `.m4a`, `.wav`, `.mp4`, `.webm`, `.mov`, and more).
- **Diarization** — lines are attributed to speakers (`SPEAKER_00`, `SPEAKER_01`, …).
- **Works from anywhere** — `whisp` is available as a normal shell command.
- **Optional summary** (`-sum`) — topics, decisions and action items.
- **Full macOS automation** — drop a file into a folder, pick up the transcript.

The output is a `<recording-name>.txt` file next to the source:

```
[SPEAKER_00]: Bring every workflow together in one place.
[SPEAKER_00]: From basic scenarios to company-wide management.
[SPEAKER_01]: Connect strategic goals to the team's daily work.
```

## Requirements

| Component | Purpose | Required |
|---|---|---|
| macOS | The project targets macOS (Apple Silicon) | yes |
| [Homebrew](https://brew.sh) | Installing `ffmpeg` | yes |
| `ffmpeg` | Audio and video decoding | yes |
| Python 3.11+ | Virtual environment with WhisperX | yes |
| A [Hugging Face](https://huggingface.co) account | Token for the diarization model | yes |
| [Claude CLI](https://claude.com/claude-code) | Only for the `-sum` flag | no |

Budget roughly **5 GB** of disk space: PyTorch accounts for most of it, and the Whisper
and diarization models are downloaded on first run.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/saktush/whisp.git ~/whisp
cd ~/whisp
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

### 3. Create the environment and install dependencies

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

The first install pulls in PyTorch and takes a few minutes.

### 4. Get a Hugging Face token

1. Create a read token: https://huggingface.co/settings/tokens
2. While signed in with the same account, accept the terms of the gated diarization
   model — **diarization will not work without this step**:
   https://huggingface.co/pyannote/speaker-diarization-community-1

### 5. Store the token in `.env`

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

`.env` is listed in `.gitignore` and will not be committed.

### 6. Make `whisp` a global command

```bash
echo "alias whisp='$HOME/whisp/whisp.sh'" >> ~/.zshrc
source ~/.zshrc
```

`whisp` now works from any directory.

## Usage

### Manually

```bash
whisp <file> [-sum] [--no-diarize]
```

- The transcript is written to `<file>.txt` **next to the source file**, not into the
  project folder.
- With `-sum`, an additional `<file>-summary.txt` is produced with a structured summary
  in Russian: topics, decisions, action items.
- A system sound plays when the run finishes.

The summary is deliberately a best-effort step: if it fails or exceeds the
`WHISP_SUMMARY_TIMEOUT` timeout you get a warning, but **the transcript itself is never
affected**.

Examples:

```bash
whisp ~/Downloads/"Team call.m4a"          # transcript only
whisp ~/Downloads/"Team call.m4a" -sum     # transcript + summary
```

- `--no-diarize` turns off speaker labelling. It only makes sense for
  recordings you know have a single voice — lectures, promo clips: it saves
  around 12% of the runtime, but strips the transcript of its only source of
  who-said-what.

### Automatically: transcribe on file drop

A macOS Folder Action watches the project folder and transcribes (and summarizes) any
audio or video file that lands in it.

Install:

```bash
./automation/install-folder-action.sh
```

macOS will ask once for permission to control System Events — grant it.

After that, just drop a file into `~/whisp`:

- unrelated file types are ignored;
- the script waits until the file has finished being written, so a partially copied
  file is never picked up;
- **at most one** transcription runs at a time, even if several files arrive at once —
  the rest queue up;
- a system notification appears when the run finishes;
- logs go to `automation/whisp-automation.log`.

Remove:

```bash
./automation/uninstall-folder-action.sh
```

If you edit `automation/whisp-folder-action.applescript`, re-run the install script to
recompile and reinstall the Folder Action.

## Configuration

Environment variables (they can also go straight into `.env`):

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token. Required |
| `WHISP_LANG` | `ru` | Language code of the recording (`en`, `de`, …) |
| `WHISP_MODEL` | `turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`) |
| `WHISP_COMPUTE_TYPE` | `int8` | CTranslate2 compute type (`int8`, `float32`) |
| `WHISP_DEVICE` | `auto` | Device for diarization and alignment: `auto`, `mps`, `cpu` |
| `WHISP_PARALLEL` | `1` | `0` runs diarization serially instead of alongside transcription |
| `WHISP_SUMMARY_TIMEOUT` | `900` | Summary generation timeout, seconds |
| `WHISP_BATCH_SIZE` | `8` | Transcription batch size |
| `WHISP_DIARIZE_BATCH_SIZE` | `64` | Diarization batch size |
| `WHISP_ASR_THREADS` | `0` | CTranslate2 threads; `0` keeps the whisperx default |

```bash
WHISP_LANG=en whisp interview.mp3
```

## What's in the repository

| File | Purpose |
|---|---|
| `whisp.sh` | Parses arguments, delegates to the Python driver, handles the summary and completion sound |
| `whisp-lib.sh` | Shared bash helpers for `whisp.sh`: the summary generation timeout |
| `whisp/` | Python pipeline driver: transcription and diarization run in parallel, device selection, timing |
| `pyproject.toml` | Environment dependencies (`pip install -e .`) |
| `automation/install-folder-action.sh` | Compiles and attaches the Folder Action |
| `automation/uninstall-folder-action.sh` | Detaches the Folder Action |
| `automation/whisp-folder-action.applescript` | The "items added to folder" handler |
| `automation/process-new-file.sh` | Type filter, write-completion wait, locking, runs `whisp.sh` |
| `tests/` | Tests: pytest suite for the `whisp/` package, bash test for `whisp-lib.sh` |
| `.github/workflows/release.yml` | Publishes a release on a `v*` tag |

## Privacy

- **Transcription and diarization run locally.** Your audio and video never leave the machine.
- **The `-sum` flag is the exception.** It sends the finished transcript text to the
  Anthropic API via the `claude` CLI. If the meeting content must not leave your machine,
  don't use `-sum` — transcription is unaffected either way.
- Network access is needed on first run to download models from Hugging Face; after that
  transcription works offline.
- `.gitignore` deliberately excludes **all** media files and `*.txt`, so recordings and
  transcripts can't be committed by accident.

## Limitations

- **Transcription runs on the CPU.** CTranslate2 has no Metal backend.
  Diarization and alignment do run on the GPU through MPS, and they run
  alongside transcription, so on Apple Silicon transcription is the
  bottleneck.
- **The summary is always in Russian** — the prompt is fixed in `whisp.sh`; changing
  `WHISP_LANG` affects the transcript, not the summary language.
- **Diarization requires accepting the terms** of the gated pyannote model (install
  step 4). Without it WhisperX fails with a model access error.
- The project is macOS-bound: Folder Actions, `osascript` notifications, `afplay` and `shlock`.

## License

[MIT](LICENSE).

This project uses, but does not bundle: [WhisperX](https://github.com/m-bain/whisperX)
(BSD-2-Clause), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT),
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT),
[PyTorch](https://pytorch.org) (BSD-3-Clause). Model terms are on their respective
Hugging Face pages.
