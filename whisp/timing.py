"""Per-stage timing.

Stage lines default to stdout -- the same stream whisperx's own verbose
per-segment transcript goes to. automation/process-new-file.sh redirects
both into one log file; sharing whisperx's stream keeps ordering correct by
construction instead of racing it as a second, independently-buffered
stream (stderr vs. stdout can flush at different, arbitrary byte
boundaries, which was observed welding a stage line onto the end of an
unrelated transcript line). Each line is written as a single write() call
that includes its own trailing newline, followed by an explicit flush:
unlike print()'s separate write-then-newline, one write() cannot be split
by another write landing in the middle of it.
"""

import contextlib
import sys
import threading
import time


class StageLog:
    def __init__(self, audio_seconds: float, stream=None):
        self.audio_seconds = audio_seconds
        self.stream = stream if stream is not None else sys.stdout
        self.stages: dict[str, float] = {}
        # whisp.pipeline runs diarization on a background thread while ASR
        # continues on the main thread, so record() can be called from both
        # at once. Without this lock two concurrent emits could interleave
        # their write() calls into a garbled log line.
        self._lock = threading.Lock()

    def _format(self, label: str, seconds: float) -> str:
        line = f"whisp: {label} {seconds:.1f}s"
        if self.audio_seconds > 0:
            line += f" RTF {seconds / self.audio_seconds:.3f}"
        return line

    def _emit(self, line: str) -> str:
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()
        return line

    def record(self, name: str, seconds: float) -> str:
        self.stages[name] = seconds
        return self._emit(self._format(name, seconds))

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - started)

    def total(self, seconds: float) -> str:
        return self._emit(self._format("TOTAL", seconds))
