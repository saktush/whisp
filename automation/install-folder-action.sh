#!/bin/bash
# One-time (or re-run-after-editing) setup: compiles the Folder Action
# AppleScript and attaches it to this repo folder, so any audio/video file
# dropped in gets transcribed automatically.
set -euo pipefail

AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$(cd "$AUTOMATION_DIR/.." && pwd)"
SOURCE_SCRIPT="$AUTOMATION_DIR/whisp-folder-action.applescript"
PROCESS_SCRIPT="$AUTOMATION_DIR/process-new-file.sh"
SCRIPTS_DIR="$HOME/Library/Scripts/Folder Action Scripts"
COMPILED_SCRIPT="$SCRIPTS_DIR/whisp-folder-action.scpt"

mkdir -p "$SCRIPTS_DIR"

# An AppleScript "property" must be a literal at compile time, so the source
# keeps a __PROCESS_SCRIPT__ placeholder and we bake this checkout's real path
# in here. That keeps the committed script free of any machine-specific path.
TMP_SCRIPT="$(mktemp -t whisp-folder-action)"
trap 'rm -f "$TMP_SCRIPT"' EXIT
awk -v repl="$PROCESS_SCRIPT" \
    '{ gsub(/__PROCESS_SCRIPT__/, repl); print }' \
    "$SOURCE_SCRIPT" >"$TMP_SCRIPT"

if grep -q '__PROCESS_SCRIPT__' "$TMP_SCRIPT"; then
    echo "Error: failed to substitute the process-new-file.sh path." >&2
    exit 1
fi

osacompile -o "$COMPILED_SCRIPT" "$TMP_SCRIPT"
chmod +x "$PROCESS_SCRIPT"

osascript <<OSA
tell application "System Events"
    set folder actions enabled to true

    if not (exists folder action "whisper-transcribe") then
        make new folder action at end of folder actions with properties {name:"whisper-transcribe", path:(POSIX file "$WHISPER_DIR")}
    end if

    tell folder action "whisper-transcribe"
        if not (exists script "whisp-folder-action.scpt") then
            make new script at end of scripts with properties {name:"whisp-folder-action.scpt", path:(POSIX file "$COMPILED_SCRIPT")}
        end if
    end tell
end tell
OSA

echo "Folder Action installed: dropping audio/video files into $WHISPER_DIR will now auto-transcribe."
echo "Logs: $AUTOMATION_DIR/whisp-automation.log"
