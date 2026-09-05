#!/bin/bash
# Run: bash tests/test_whisp_lib.sh
set -uo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/whisp-lib.sh"
# shellcheck source=/dev/null
source "$LIB"

fail=0
check() {
    if [ "$2" = "$3" ]; then
        echo "ok   - $1"
    else
        echo "FAIL - $1 (expected '$3', got '$2')"
        fail=1
    fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
: >"$tmp/in"

# A command that fails immediately must not be reported as a timeout.
run_with_timeout 30 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'echo boom >&2; exit 3'
check "fast failure returns the command status" "$?" "3"
check "fast failure leaves the flag empty" "$(cat "$tmp/flag")" ""
check "fast failure captures output" "$(cat "$tmp/out")" "boom"

# A command that outlives the timeout must be reported as a timeout.
run_with_timeout 1 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'sleep 30'
check "timeout is flagged" "$(cat "$tmp/flag")" "timeout"

# A command that succeeds must return 0 and leave the flag empty.
run_with_timeout 30 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'echo fine'
check "success returns 0" "$?" "0"
check "success leaves the flag empty" "$(cat "$tmp/flag")" ""

exit "$fail"
