#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=whisp-lib.sh
source "$SCRIPT_DIR/whisp-lib.sh"

# Ensure ffmpeg (Homebrew) and claude (~/.local/bin) are on PATH regardless of
# caller's environment (e.g. the Folder Actions dispatcher runs with a minimal PATH).
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

# --- Argument parsing ---
FILE=""
SUMMARIZE=0
DIARIZE=1

for arg in "$@"; do
    case "$arg" in
        -sum)
            SUMMARIZE=1
            ;;
        --no-diarize)
            DIARIZE=0
            ;;
        -*)
            echo "Error: Unknown option '$arg'" >&2
            echo "Usage: whisp <filename> [-sum] [--no-diarize]" >&2
            exit 1
            ;;
        *)
            if [ -n "$FILE" ]; then
                echo "Error: Multiple filenames provided ('$FILE' and '$arg')." >&2
                exit 1
            fi
            FILE="$arg"
            ;;
    esac
done

if [ -z "$FILE" ]; then
    echo "Error: No file provided."
    echo "Usage: whisp <filename> [-sum] [--no-diarize]"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found."
    exit 1
fi

# Pick up HF_TOKEN (needed for diarization) from a local .env, if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN is not set. Put it in $SCRIPT_DIR/.env as HF_TOKEN=... (see README.md)."
    exit 1
fi

OUTPUT_DIR="$(cd "$(dirname "$FILE")" && pwd)"

# Tunables. The defaults reproduce the original hardcoded behaviour; override
# them in the environment or by adding the same names to .env (which is sourced
# above, so values set there win over the caller's environment).
MODEL="${WHISP_MODEL:-turbo}"
LANG_CODE="${WHISP_LANG:-ru}"
COMPUTE_TYPE="${WHISP_COMPUTE_TYPE:-int8}"
# The WHISP_SUMMARY_* knobs (backend, model, chunk size, timeout) are read by
# the summarizer straight from the environment, the same way WHISP_DEVICE and
# WHISP_BATCH_SIZE are -- .env is sourced with `set -a` above, so they reach it.

DIARIZE_ARGS=()
if [ "$DIARIZE" -eq 0 ]; then
    DIARIZE_ARGS+=(--no-diarize)
fi

# The whisperx CLI takes one --device for every stage, which pins diarization
# to the CPU because CTranslate2 has no Metal backend. This driver picks a
# device per stage and runs diarization alongside ASR instead.
#
# "${DIARIZE_ARGS[@]+"${DIARIZE_ARGS[@]}"}" rather than "${DIARIZE_ARGS[@]}":
# macOS ships bash 3.2, where expanding an empty array under `set -u` aborts
# with "unbound variable". The array is empty on the default path -- every run
# that keeps diarization on -- so the plain form would break normal use.
PYTHONPATH="$SCRIPT_DIR" "$SCRIPT_DIR/.venv/bin/python" -m whisp "$FILE" \
    --model "$MODEL" \
    --language "$LANG_CODE" \
    --compute-type "$COMPUTE_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    "${DIARIZE_ARGS[@]+"${DIARIZE_ARGS[@]}"}"

BASENAME="$(basename "$FILE")"
BASENAME="${BASENAME%.*}"
TRANSCRIPT_FILE="$OUTPUT_DIR/$BASENAME.txt"
SUMMARY_FILE="$OUTPUT_DIR/$BASENAME-summary.txt"

if [ "$SUMMARIZE" -eq 1 ]; then
    echo "Generating summary..."

    # Summarization runs as its own process, started only after the
    # transcription driver above has exited. torch-MPS and MLX each keep an
    # independent caching allocator over the same unified memory with no
    # cross-pressure signalling, so loading a language model beside a live
    # torch runtime is an out-of-memory or a swap storm on a 16 GB machine.
    #
    # The summarizer owns its own deadline (WHISP_SUMMARY_TIMEOUT) and writes
    # its progress to the inherited stdout, so a local run that takes minutes
    # still reports per-chunk stage lines into the automation log instead of
    # going silent. That is why this does not go through run_with_timeout,
    # which buffers both streams into a file shown only on failure.
    SUMMARY_TMP="$(mktemp)"
    trap 'rm -f "$SUMMARY_TMP"' EXIT

    if PYTHONPATH="$SCRIPT_DIR" "$SCRIPT_DIR/.venv/bin/python" -m whisp.summarize \
            "$TRANSCRIPT_FILE" \
            --output "$SUMMARY_TMP" \
            --language "$LANG_CODE"; then
        # Guard the move: a zero-byte result must never replace a good summary
        # from an earlier run.
        if [ -s "$SUMMARY_TMP" ]; then
            mv "$SUMMARY_TMP" "$SUMMARY_FILE"
            echo "Summary saved to $SUMMARY_FILE"
        else
            echo "Warning: the summarizer exited cleanly but wrote nothing; the transcript itself is unaffected and is saved at $TRANSCRIPT_FILE." >&2
        fi
    else
        echo "Warning: summary generation failed; the transcript itself is unaffected and is saved at $TRANSCRIPT_FILE." >&2
    fi
fi

# Audible completion signal (macOS system sound). Silently ignored if
# afplay/audio output isn't available (e.g. locked screen, no audio device).
afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 || true
