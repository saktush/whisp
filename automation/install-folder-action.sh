#!/bin/bash
# One-time (or re-run-after-editing) setup: compiles the Folder Action
# AppleScript and attaches it to a folder, so any audio/video file dropped in
# gets transcribed automatically.
#
# Usage: install-folder-action.sh [FOLDER]
#   FOLDER defaults to this repo. Pass a dedicated inbox -- e.g. a folder you
#   drop recordings into -- to keep the repo free of media files.
#
# Safe to run for several folders: the compiled script is folder-agnostic (the
# handler receives the folder from the event), so they all share one copy.
set -euo pipefail

AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$(cd "$AUTOMATION_DIR/.." && pwd)"
SOURCE_SCRIPT="$AUTOMATION_DIR/whisp-folder-action.applescript"
PROCESS_SCRIPT="$AUTOMATION_DIR/process-new-file.sh"
SCRIPTS_DIR="$HOME/Library/Scripts/Folder Action Scripts"
COMPILED_SCRIPT="$SCRIPTS_DIR/whisp-folder-action.scpt"
SCRIPT_NAME="whisp-folder-action.scpt"

TARGET_DIR="${1:-$WHISPER_DIR}"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a directory." >&2
    echo "Usage: $(basename "$0") [FOLDER]   (defaults to $WHISPER_DIR)" >&2
    exit 1
fi
# Resolve to the *physical* absolute path, with no trailing slash. System
# Events reports folder actions with symlinks already resolved (/tmp/x comes
# back as /private/tmp/x), so comparing a logical path would never match and
# we would attach a second action to the same folder.
TARGET_DIR="$(cd "$TARGET_DIR" && pwd -P)"
TARGET_NAME="$(basename "$TARGET_DIR")"

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

# Escape for embedding in an AppleScript string literal. Folder names routinely
# contain spaces, which are harmless, but a quote or backslash would end the
# literal early.
esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

osascript <<OSA
tell application "System Events"
    set folder actions enabled to true
    set targetPath to "$(esc "$TARGET_DIR")"

    -- Match on path, not on name: several folders may be watched at once, and
    -- two of them could easily share a name.
    set existing to missing value
    repeat with candidate in folder actions
        try
            set candidatePath to (path of candidate) as text
            if candidatePath ends with "/" then
                set candidatePath to text 1 thru -2 of candidatePath
            end if
            if candidatePath is targetPath then set existing to candidate
        end try
    end repeat

    if existing is missing value then
        set existing to make new folder action at end of folder actions with properties {name:"$(esc "$TARGET_NAME")", path:(POSIX file targetPath)}
    end if

    tell existing
        set enabled to true
        if not (exists script "$SCRIPT_NAME") then
            make new script at end of scripts with properties {name:"$SCRIPT_NAME", path:(POSIX file "$(esc "$COMPILED_SCRIPT")")}
        end if
    end tell
end tell
OSA

echo "Folder Action installed: dropping audio/video files into $TARGET_DIR will now auto-transcribe."
echo "Logs: $AUTOMATION_DIR/whisp-automation.log"
