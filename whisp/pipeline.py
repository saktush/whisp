"""Pipeline orchestration.

Diarization does not depend on the transcript, and it runs on the GPU while
ASR runs on the CPU, so the two are started together and the wall time is
max(ASR, diarization) instead of their sum. On an 80-minute recording that
takes diarization off the critical path entirely: it finishes around minute 6
while ASR runs to minute 17.

Stages are injected so the concurrency, merge and degradation logic can be
tested without loading any model.
"""

import dataclasses
import pathlib
import threading
import time
from typing import Any, Callable

from whisp import stages as default_stages
from whisp.timing import StageLog


@dataclasses.dataclass
class Stages:
    decode: Callable[..., Any]
    duration: Callable[..., float]
    load_asr: Callable[..., Any]
    transcribe: Callable[..., dict]
    align: Callable[..., dict]
    diarize: Callable[..., Any]
    assign: Callable[..., dict]
    write: Callable[..., Any]


def default() -> Stages:
    import whisperx
    from whisperx.utils import get_writer

    def write(result, audio_path, output_dir):
        writer = get_writer("txt", output_dir)
        writer(
            result,
            audio_path,
            {"highlight_words": False, "max_line_count": None, "max_line_width": None},
        )

    return Stages(
        decode=default_stages.decode,
        duration=default_stages.duration_seconds,
        load_asr=default_stages.load_asr,
        transcribe=default_stages.transcribe,
        align=default_stages.align_segments,
        diarize=default_stages.diarize,
        assign=whisperx.assign_word_speakers,
        write=write,
    )


def run(
    audio_path: str,
    output_dir: str,
    language: str,
    model_name: str,
    compute_type: str,
    device: str,
    hf_token: str,
    batch_size: int,
    diarize_batch_size: int,
    asr_thread_count: int,
    diarize_enabled: bool,
    parallel: bool,
    stages: Stages | None = None,
    log: StageLog | None = None,
    diarize_timeout: float | None = None,
) -> pathlib.Path:
    stages = stages or default()
    started = time.monotonic()

    audio = stages.decode(audio_path)
    audio_seconds = stages.duration(audio)
    log = log or StageLog(audio_seconds=audio_seconds)
    log.record("decode", time.monotonic() - started)

    # Bound how long we wait for diarization at the join point below, which
    # is *after* transcription has already finished. Diarization starts at
    # the same moment as transcription; on the GPU path it finishes in
    # roughly a third of transcription's time, and even on the slowest
    # measured path -- CPU-only diarization at RTF 0.89 against
    # transcription's 0.21 -- it needs well under one further
    # realtime-equivalent after transcription ends. One times the audio
    # duration is therefore generous headroom, and the 300-second floor
    # protects short recordings.
    if diarize_timeout is None:
        diarize_timeout = max(300.0, audio_seconds)

    diarization: dict[str, Any] = {}

    def diarize_worker():
        try:
            worker_started = time.monotonic()
            diarization["df"] = stages.diarize(
                audio, device, hf_token, diarize_batch_size
            )
            log.record("diarize", time.monotonic() - worker_started)
        except Exception as exc:  # transcript matters more than speaker labels
            diarization["error"] = exc

    worker = None
    if diarize_enabled:
        if parallel:
            # daemon=True: the transcript must never wait on a stuck worker.
            # If diarization hangs past the join timeout below, the process
            # still needs to be able to exit once the transcript is written.
            # A daemon thread doing torch/MPS work that gets torn down
            # abruptly at interpreter exit may print shutdown noise -- an
            # accepted trade against hanging forever.
            worker = threading.Thread(
                target=diarize_worker, name="whisp-diarize", daemon=True
            )
            worker.start()
        else:
            diarize_worker()

    with log.stage("asr"):
        asr_model = stages.load_asr(
            model_name, compute_type, language, asr_thread_count, hf_token
        )
        result = stages.transcribe(asr_model, audio, batch_size, language)
    del asr_model

    timed_out = False
    if worker is not None:
        worker.join(timeout=diarize_timeout)
        timed_out = worker.is_alive()

    if diarize_enabled and "df" in diarization:
        try:
            with log.stage("align"):
                result = stages.align(result["segments"], audio, language, device)
            result = stages.assign(diarization["df"], result)
        except Exception as exc:  # transcript matters more than speaker labels
            print(
                f"whisp: alignment failed ({type(exc).__name__}: {exc}); "
                "writing the transcript without speaker labels",
                flush=True,
            )
    elif diarize_enabled and timed_out:
        print(
            f"whisp: diarization timed out after {diarize_timeout:.0f}s; "
            "writing the transcript without speaker labels",
            flush=True,
        )
    elif diarize_enabled:
        exc = diarization.get("error")
        print(
            f"whisp: diarization failed ({type(exc).__name__}: {exc}); "
            "writing the transcript without speaker labels",
            flush=True,
        )

    stages.write(result, audio_path, output_dir)
    log.total(time.monotonic() - started)

    stem = pathlib.Path(audio_path).stem
    return pathlib.Path(output_dir) / f"{stem}.txt"
