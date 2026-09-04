#!/bin/bash
# Called by the Folder Action (see whisp-folder-action.applescript) for every
# item added to the whisp folder. Silently ignores non-audio/video files. Guarantees
# strictly sequential processing across all invocations via a shlock-based
# lock, even if several files land in the folder at once.
set -uo pipefail

WHISPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$WHISPER_DIR/automation/whisp-automation.log"
LOCK_FILE="/tmp/whisp-automation.lock"

FILE="$1"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_FILE"
}

notify() {
    osascript -e 'on run argv' \
              -e 'display notification (item 2 of argv) with title (item 1 of argv)' \
              -e 'end run' \
              "$1" "$2" >/dev/null 2>&1 || true
}

ext="${FILE##*.}"
ext_lower="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"
case "$ext_lower" in
    mp3|m4a|wav|aac|flac|ogg|wma|mp4|mov|mkv|avi|webm|m4v|flv) ;;
    *) exit 0 ;;
esac

[ -f "$FILE" ] || exit 0

# Wait until the file size is stable for a few checks in a row, so we don't
# grab a half-written copy/download.
prev_size=-1
stable_count=0
while [ "$stable_count" -lt 3 ]; do
    size=$(stat -f%z "$FILE" 2>/dev/null || echo -1)
    if [ "$size" = "$prev_size" ] && [ "$size" != "-1" ]; then
        stable_count=$((stable_count + 1))
    else
        stable_count=0
    fi
    prev_size="$size"
    sleep 2
done

# Never run more than one transcription at a time.
while ! shlock -f "$LOCK_FILE" -p $$; do
    sleep 2
done
trap 'rm -f "$LOCK_FILE"' EXIT

log "Starting transcription: $FILE"
if "$WHISPER_DIR/whisp.sh" "$FILE" -sum >>"$LOG_FILE" 2>&1; then
    log "Finished transcription: $FILE"
    notify "whisp: transcription complete" "$(basename "$FILE")"
else
    log "FAILED transcription: $FILE"
    notify "whisp: transcription failed" "$(basename "$FILE") (see automation/whisp-automation.log)"
fi
