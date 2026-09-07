#!/bin/bash
# Detaches the whisp Folder Action from a folder.
#
# Usage: uninstall-folder-action.sh [FOLDER]
#   FOLDER defaults to this repo.
#
# Removes only our own script from that folder's action. If the action is left
# with no scripts it is deleted too, but a folder action carrying somebody
# else's scripts is kept. Folder Actions are not disabled globally, in case
# other automations rely on them.
set -euo pipefail

AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$(cd "$AUTOMATION_DIR/.." && pwd)"
SCRIPTS_DIR="$HOME/Library/Scripts/Folder Action Scripts"
COMPILED_SCRIPT="$SCRIPTS_DIR/whisp-folder-action.scpt"
SCRIPT_NAME="whisp-folder-action.scpt"

TARGET_DIR="${1:-$WHISPER_DIR}"
# The folder may already be gone -- detaching should still work, so this does
# not require the directory to exist. When it does exist, resolve the physical
# path: System Events reports folder actions with symlinks resolved, so a
# logical path such as /tmp/x would never match its /private/tmp/x entry and
# the detach would silently report "nothing to do".
if [ -d "$TARGET_DIR" ]; then
    TARGET_DIR="$(cd "$TARGET_DIR" && pwd -P)"
fi
TARGET_DIR="${TARGET_DIR%/}"

esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

RESULT="$(osascript <<OSA
tell application "System Events"
    set targetPath to "$(esc "$TARGET_DIR")"

    -- Find first, mutate afterwards. Deleting a folder action through the
    -- loop variable of a "repeat with x in folder actions" fails, and the
    -- failure is easy to miss because the script deletion above it succeeds.
    set matchIndex to 0
    repeat with i from 1 to (count of folder actions)
        try
            set candidatePath to (path of folder action i) as text
            if candidatePath ends with "/" then
                set candidatePath to text 1 thru -2 of candidatePath
            end if
            if candidatePath is targetPath then
                set matchIndex to i
                exit repeat
            end if
        end try
    end repeat

    if matchIndex is 0 then return "not-attached"

    set hadScript to false
    tell folder action matchIndex
        if exists script "$SCRIPT_NAME" then
            delete script "$SCRIPT_NAME"
            set hadScript to true
        end if
        set remaining to count of scripts
    end tell

    -- Decide the outcome before the optional cleanup below, so that failing to
    -- remove a now-empty action never misreports the script removal itself.
    if hadScript then
        if remaining is 0 then
            set outcome to "removed"
        else
            set outcome to "removed-kept-action"
        end if
    else
        if remaining is 0 then
            set outcome to "removed-empty"
        else
            set outcome to "not-attached"
        end if
    end if

    if remaining is 0 then
        try
            delete folder action matchIndex
        on error
            set outcome to outcome & " (the empty folder action could not be removed)"
        end try
    end if

    return outcome
end tell
OSA
)"

# The compiled script is shared by every watched folder, so it may only be
# deleted once nothing references it any more.
STILL_USED="$(osascript <<'OSA'
tell application "System Events"
    set n to 0
    repeat with candidate in folder actions
        try
            if exists script "whisp-folder-action.scpt" of candidate then set n to n + 1
        end try
    end repeat
    return n
end tell
OSA
)"

case "$RESULT" in
    removed)
        echo "Folder Action removed from $TARGET_DIR. Automatic transcription there is now disabled." ;;
    removed-kept-action)
        echo "whisp script removed from $TARGET_DIR; its folder action was kept because other scripts are still attached." ;;
    removed-empty)
        echo "Removed an empty folder action on $TARGET_DIR (it had no scripts attached)." ;;
    *)
        echo "No whisp Folder Action was attached to $TARGET_DIR; nothing to do." ;;
esac

if [ "$STILL_USED" -eq 0 ]; then
    rm -f "$COMPILED_SCRIPT"
else
    echo "Kept $COMPILED_SCRIPT: still used by $STILL_USED other watched folder(s)."
fi
