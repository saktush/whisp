#!/bin/bash
# Detaches the whisp Folder Action and removes the compiled script.
# Does not disable Folder Actions globally, in case other folder actions rely on it.
set -euo pipefail

SCRIPTS_DIR="$HOME/Library/Scripts/Folder Action Scripts"
COMPILED_SCRIPT="$SCRIPTS_DIR/whisp-folder-action.scpt"

osascript <<'OSA'
tell application "System Events"
    if exists folder action "whisper-transcribe" then
        delete folder action "whisper-transcribe"
    end if
end tell
OSA

rm -f "$COMPILED_SCRIPT"
echo "Folder Action removed. Automatic transcription on file add is now disabled."
