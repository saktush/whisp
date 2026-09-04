import io
import threading

import pytest

from whisp import pipeline
from whisp.timing import StageLog


def make_stages(**overrides):
    """Fake stages: fast, deterministic, no models."""
    calls = []

    def record(name, result=None):
        def fn(*args, **kwargs):
            calls.append(name)
            return result
        return fn

    stages = pipeline.Stages(
        decode=record("decode", "AUDIO"),
        duration=lambda audio: 100.0,
        load_asr=record("load_asr", "ASR_MODEL"),
        transcribe=record("transcribe", {"segments": [{"text": "hi"}]}),
        align=record("align", {"segments": [{"text": "hi", "words": []}]}),
        diarize=record("diarize", "DIARIZE_DF"),
        assign=record("assign", {"segments": [{"text": "hi", "speaker": "SPEAKER_00"}]}),
        write=record("write"),
    )
    for key, value in overrides.items():
        setattr(stages, key, value)
    stages.calls = calls
    return stages


def run(stages, **kwargs):
    params = dict(
        audio_path="/tmp/x.wav",
        output_dir="/tmp",
        language="ru",
        model_name="turbo",
        compute_type="int8",
        device="mps",
        hf_token="tok",
        batch_size=8,
        diarize_batch_size=64,
        asr_thread_count=4,
        diarize_enabled=True,
        parallel=True,
        stages=stages,
        log=StageLog(audio_seconds=100.0, stream=io.StringIO()),
    )
    params.update(kwargs)
    return pipeline.run(**params)


def test_diarization_runs_concurrently_with_asr():
    """The whole point of the change: diarization must not wait for ASR."""
    diarize_started = threading.Event()
    asr_may_finish = threading.Event()

    def slow_diarize(*args, **kwargs):
        diarize_started.set()
        return "DIARIZE_DF"

    def transcribe_waiting_for_diarize(*args, **kwargs):
        # Fails the test rather than hanging forever if the stages are serial.
        assert diarize_started.wait(timeout=5), "diarization did not start during ASR"
        asr_may_finish.set()
        return {"segments": [{"text": "hi"}]}

    stages = make_stages(diarize=slow_diarize, transcribe=transcribe_waiting_for_diarize)
    run(stages)
    assert asr_may_finish.is_set()


def test_serial_mode_still_produces_a_transcript():
    stages = make_stages()
    run(stages, parallel=False)
    assert "diarize" in stages.calls
    assert "assign" in stages.calls
    assert "write" in stages.calls


def test_diarization_failure_does_not_lose_the_transcript():
    def failing_diarize(*args, **kwargs):
        raise RuntimeError("MPS out of memory")

    stages = make_stages(diarize=failing_diarize)
    run(stages)
    assert "write" in stages.calls
    assert "assign" not in stages.calls


def test_diarize_timeout_does_not_block_the_transcript():
    """A hung diarize() must not hold up the transcript forever.

    Uses an Event that is never set, so diarize() blocks indefinitely; the
    worker thread is a daemon and is simply abandoned. A tiny diarize_timeout
    keeps this test well under a second instead of waiting minutes.
    """

    def hanging_diarize(*args, **kwargs):
        threading.Event().wait()
        return "DIARIZE_DF"  # never reached

    stages = make_stages(diarize=hanging_diarize)
    run(stages, diarize_timeout=0.01)
    assert "write" in stages.calls
    assert "assign" not in stages.calls


def test_no_diarize_skips_diarization_and_alignment():
    stages = make_stages()
    run(stages, diarize_enabled=False)
    assert "diarize" not in stages.calls
    assert "align" not in stages.calls
    assert "write" in stages.calls


def test_returns_transcript_path():
    stages = make_stages()
    result = run(stages, audio_path="/tmp/meeting.webm", output_dir="/out")
    assert str(result) == "/out/meeting.txt"
