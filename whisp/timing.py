"""Per-stage timing.

The automation log is the only place stage durations are recorded, so the
lines are flushed as they happen: whisperx block-buffers its own output when
redirected to a file, which made the log lag reality by minutes.
"""

import contextlib
import sys
import time


class StageLog:
    def __init__(self, audio_seconds: float, stream=None):
        self.audio_seconds = audio_seconds
        self.stream = stream if stream is not None else sys.stderr
        self.stages: dict[str, float] = {}

    def _format(self, label: str, seconds: float) -> str:
        line = f"whisp: {label} {seconds:.1f}s"
        if self.audio_seconds > 0:
            line += f" RTF {seconds / self.audio_seconds:.3f}"
        return line

    def _emit(self, line: str) -> str:
        print(line, file=self.stream, flush=True)
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
