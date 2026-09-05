#!/bin/bash
# Shared helpers for whisp.sh. Kept in its own file so tests can source it.

# Run "$5..." with stdin from $3, output to $2, killing it after $1 seconds and
# writing "timeout" into $4 if the watchdog is what stopped it. Hand-rolled
# because no timeout/gtimeout binary exists on a stock Mac. Stdin must be
# passed in and redirected right here (not inherited from the caller) -- bash
# drops an inherited stdin redirect to /dev/null for a backgrounded (&)
# command unless the redirect is written on that exact command line.
#
# The flag file exists so callers can tell a timeout from a fast failure. They
# used to be indistinguishable, and an instant auth error was reported to the
# log as "timed out after 300s" for weeks.
run_with_timeout() {
    local timeout_secs="$1" out_file="$2" in_file="$3" flag_file="$4"
    shift 4
    : >"$flag_file"
    "$@" <"$in_file" >"$out_file" 2>&1 &
    local cmd_pid=$!
    ( sleep "$timeout_secs"
      if kill -TERM "$cmd_pid" 2>/dev/null; then printf 'timeout' >"$flag_file"; fi ) &
    local watcher_pid=$!
    local status=0
    wait "$cmd_pid" || status=$?
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
    return "$status"
}
