#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure ffmpeg (Homebrew) and claude (~/.local/bin) are on PATH regardless of
# caller's environment (e.g. the Folder Actions dispatcher runs with a minimal PATH).
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

# Run "$@" with stdin from $3, redirecting output to $2, killing it if it
# runs longer than $1 seconds. Hand-rolled because no timeout/gtimeout binary
# exists on this Mac. Stdin must be passed in and redirected right here (not
# inherited from the caller) -- bash drops an inherited stdin redirect to
# /dev/null for a backgrounded (&) command unless the redirect is written on
# that exact command line.
run_with_timeout() {
    local timeout_secs="$1" out_file="$2" in_file="$3"
    shift 3
    "$@" <"$in_file" >"$out_file" 2>&1 &
    local cmd_pid=$!
    ( sleep "$timeout_secs"; kill -TERM "$cmd_pid" 2>/dev/null ) &
    local watcher_pid=$!
    local status=0
    wait "$cmd_pid" || status=$?
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
    return "$status"
}

# --- Argument parsing ---
FILE=""
SUMMARIZE=0

for arg in "$@"; do
    case "$arg" in
        -sum)
            SUMMARIZE=1
            ;;
        -*)
            echo "Error: Unknown option '$arg'" >&2
            echo "Usage: whisp <filename> [-sum]" >&2
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
    echo "Usage: whisp <filename> [-sum]"
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

# Run WhisperX
"$SCRIPT_DIR/.venv/bin/whisperx" "$FILE" \
    --model "$MODEL" \
    --language "$LANG_CODE" \
    --compute_type "$COMPUTE_TYPE" \
    --diarize \
    --hf_token "$HF_TOKEN" \
    --output_dir "$OUTPUT_DIR" \
    --output_format txt

BASENAME="$(basename "$FILE")"
BASENAME="${BASENAME%.*}"
TRANSCRIPT_FILE="$OUTPUT_DIR/$BASENAME.txt"
SUMMARY_FILE="$OUTPUT_DIR/$BASENAME-summary.txt"

if [ "$SUMMARIZE" -eq 1 ]; then
    echo "Generating summary..."

    SUMMARY_SYSTEM_PROMPT="You are a transcript summarizer for Russian-language business call/meeting recordings. You will receive a diarized transcript on stdin, with lines like [SPEAKER_00]: text. Respond ONLY with a structured summary in Russian, using exactly this format:

## Основные темы
(bullet list of key topics discussed)

## Решения
(bullet list of decisions made; write 'Нет' if none)

## Задачи
(bullet list of action items, with owner/speaker if identifiable; write 'Нет' if none)

Do not include any preamble, disclaimers, meta-commentary about the transcript format, or text outside this structure. Do not use any tools. Output plain text only, no markdown code fences."

    SUMMARY_TMP="$(mktemp)"

    if run_with_timeout 300 "$SUMMARY_TMP" "$TRANSCRIPT_FILE" claude -p \
            --system-prompt "$SUMMARY_SYSTEM_PROMPT" \
            --model sonnet; then
        mv "$SUMMARY_TMP" "$SUMMARY_FILE"
        echo "Summary saved to $SUMMARY_FILE"
    else
        echo "Warning: summary generation failed or timed out after 300s; the transcript itself is unaffected and is saved at $TRANSCRIPT_FILE." >&2
        cat "$SUMMARY_TMP" >&2 || true
        rm -f "$SUMMARY_TMP"
    fi
fi

# Audible completion signal (macOS system sound). Silently ignored if
# afplay/audio output isn't available (e.g. locked screen, no audio device).
afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 || true
