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


def test_alignment_failure_does_not_lose_the_transcript():
    """F1: an align/assign exception must not discard a finished ASR result.

    Reproduces the reviewer's repro case: align() raises after transcription
    has already produced a complete result. Before the fix this propagated
    out of run() and stages.write was never called, discarding a finished
    transcript.
    """

    def failing_align(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory")

    stages = make_stages(align=failing_align)
    run(stages)
    assert "write" in stages.calls
    assert "assign" not in stages.calls


def test_assign_failure_does_not_lose_the_transcript():
    """Same guarantee when assign (not align) is the one that raises."""

    def failing_assign(*args, **kwargs):
        raise KeyError("speaker")

    stages = make_stages(assign=failing_assign)
    run(stages)
    assert "write" in stages.calls
    assert "align" in stages.calls


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


def test_speaker_labels_survive_the_align_then_assign_order():
    """Pins pipeline.run's stage order: align must run before assign.

    whisperx's real align_segments (whisperx/alignment.py) rebuilds every
    output segment as a fresh {text, start, end, words} dict, which drops
    any 'speaker' key an earlier assign_word_speakers call had set. If the
    two calls in pipeline.run were ever swapped, align would silently erase
    every speaker label assign had just added -- the run would still exit 0
    and write a plausible-looking transcript, just with no speaker names in
    it. The fakes below mirror that discarding behaviour (fake_align always
    returns segments without 'speaker'; fake_assign is what adds it) so this
    test fails if the order regresses, instead of only checking that both
    stages ran somewhere.
    """
    written = {}

    def fake_align(segments, *args, **kwargs):
        return {
            "segments": [
                {"text": s["text"], "start": 0.0, "end": 1.0, "words": []}
                for s in segments
            ]
        }

    def fake_assign(diarize_df, result):
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
        return result

    def fake_write(result, audio_path, output_dir):
        written["result"] = result

    stages = make_stages(align=fake_align, assign=fake_assign, write=fake_write)
    run(stages)

    segments = written["result"]["segments"]
    assert segments, "no segments reached write()"
    assert all("speaker" in segment for segment in segments)
