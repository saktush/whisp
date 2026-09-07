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
| [Claude CLI](https://claude.com/claude-code) | Only for `WHISP_SUMMARY_BACKEND=claude` | no |

Budget roughly **7 GB** of disk space: PyTorch accounts for most of it, and the Whisper
and diarization models are downloaded on first run. A further ~2.1 GB goes to the
language model used for summaries, downloaded the first time you pass `-sum`.

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
- With `-sum`, an additional `<file>-summary.txt` is produced with a structured summary:
  participants, topics, decisions, action items. The summary is produced by a **local**
  model through MLX, in the language given by `WHISP_LANG`. The participant roster is
  reconstructed from the transcript itself -- from whoever introduced themselves or was
  addressed by name.
- A system sound plays when the run finishes.

The summary is deliberately a best-effort step: if it fails or exceeds
`WHISP_SUMMARY_TIMEOUT`, you get a warning, but **the transcript itself is never
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

A macOS Folder Action watches a folder you name and transcribes (and summarizes) any
audio or video file that lands in it.

Install -- the folder is an argument:

```bash
./automation/install-folder-action.sh ~/Downloads/inbox
```

With no argument it watches the project folder. A dedicated inbox is usually better:
recordings and transcripts then stay out of the project's own files.

You can attach several folders. They share one compiled script, because the handler
takes the folder from the event itself.

macOS will ask once for permission to control System Events — grant it.

After that, just drop a file into the watched folder:

- unrelated file types are ignored;
- the script waits until the file has finished being written, so a partially copied
  file is never picked up;
- **at most one** transcription runs at a time, even if several files arrive at once —
  the rest queue up;
- a system notification appears when the run finishes;
- logs go to `automation/whisp-automation.log`.

Remove -- the same folder as an argument:

```bash
./automation/uninstall-folder-action.sh ~/Downloads/inbox
```

Only the whisp script is detached. If other Folder Action scripts are attached to that
folder, the action itself is kept; the shared compiled script is deleted only once the
last watched folder stops using it. Folder Actions are never disabled globally.

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
| `WHISP_SUMMARY_BACKEND` | `mlx` | Summary engine: `mlx` (local) or `claude` |
| `WHISP_SUMMARY_MODEL` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | Summary model; `sonnet` for the `claude` backend |
| `WHISP_SUMMARY_LANG` | value of `WHISP_LANG` | Summary language: `ru`, or English for anything else |
| `WHISP_SUMMARY_CHUNK_TOKENS` | `6000` | Fragment size; longer transcripts go through map-reduce |
| `WHISP_SUMMARY_MAX_TOKENS` | `3072` | Upper bound on summary length, in tokens |
| `WHISP_SUMMARY_PROMPT` | — | Directory with your own `single.txt`, `map.txt`, `reduce.txt` |
| `WHISP_SUMMARY_TIMEOUT` | `1800` | Summary generation timeout, seconds (model loading excluded) |
| `WHISP_BATCH_SIZE` | `8` | Transcription batch size |
| `WHISP_DIARIZE_BATCH_SIZE` | `64` | Diarization batch size |
| `WHISP_ASR_THREADS` | `0` | CTranslate2 threads; `0` keeps the whisperx default |

```bash
WHISP_LANG=en whisp interview.mp3
```

Custom prompts are supplied as a directory. Every file is optional: whatever you
leave out keeps the built-in prompt, so you can replace a single phase.

```bash
mkdir -p ~/whisp-prompts
# start from the built-in prompt and edit it
./.venv/bin/python -m whisp.summarize --print-prompt single > ~/whisp-prompts/single.txt
WHISP_SUMMARY_PROMPT=~/whisp-prompts whisp meeting.m4a -sum
```

| File | Used when |
|---|---|
| `single.txt` | The whole transcript fits one fragment |
| `map.txt` | Notes on one fragment of a long transcript |
| `reduce.txt` | Merging the notes into the final summary |

## What's in the repository

| File | Purpose |
|---|---|
| `whisp.sh` | Parses arguments, delegates to the Python driver, handles the summary and completion sound |
| `whisp-lib.sh` | Shared bash helpers for `whisp.sh`: the summary generation timeout |
| `whisp/` | Python pipeline driver: transcription and diarization run in parallel, device selection, timing |
| `whisp/summarize.py` | Summarization: chunking, map-reduce, CLI (`python -m whisp.summarize`) |
| `whisp/summary_backends.py` | Summary engines: local MLX and `claude` |
| `whisp/prompts.py` | Summary prompts for Russian and English, plus custom-prompt loading |
| `pyproject.toml` | Environment dependencies (`pip install -e .`) |
| `automation/install-folder-action.sh` | Compiles and attaches the Folder Action to a folder (argument) |
| `automation/uninstall-folder-action.sh` | Detaches the Folder Action from a folder (argument) |
| `automation/whisp-folder-action.applescript` | The "items added to folder" handler |
| `automation/process-new-file.sh` | Type filter, write-completion wait, locking, runs `whisp.sh` |
| `tests/` | Tests: pytest suite for the `whisp/` package, bash test for `whisp-lib.sh` |
| `.github/workflows/release.yml` | Publishes a release on a `v*` tag |

## Privacy

- **The whole pipeline runs locally**, summaries included: by default they are produced
  by a local model through MLX. Your audio, video and transcript never leave the machine.
- **The one exception is opting in with `WHISP_SUMMARY_BACKEND=claude`.** Only then is the
  transcript text sent to the Anthropic API via the `claude` CLI. It does not happen by default.
- Network access is needed on first run to download models from Hugging Face; after that
  everything, summaries included, works offline.
- `.gitignore` deliberately excludes **all** media files and `*.txt`, so recordings and
  transcripts can't be committed by accident.

## Limitations

- **Transcription runs on the CPU.** CTranslate2 has no Metal backend.
  Diarization runs on the GPU through MPS alongside transcription; alignment
  also runs on the GPU, but sequentially after transcription finishes, so on
  Apple Silicon transcription is the bottleneck.
- **Summaries come in Russian and English.** The language follows `WHISP_LANG` and can be
  overridden with `WHISP_SUMMARY_LANG`; anything other than Russian uses the English template.
- **The local model is weaker than the hosted one.** Summaries are produced by a compact 4B
  model. For maximum quality use `WHISP_SUMMARY_BACKEND=claude` — at the cost of sending the
  transcript to the Anthropic API.
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
