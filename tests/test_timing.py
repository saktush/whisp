import io
import sys

from whisp import timing


def test_default_stream_is_stdout():
    """whisperx's own verbose transcript also goes to stdout; sharing that
    stream is what keeps `whisp:` lines from landing mid-line inside it when
    both are redirected into one log file (see the module docstring)."""
    assert timing.StageLog(audio_seconds=1.0).stream is sys.stdout


def test_record_formats_line_with_rtf():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=1200.0, stream=out)
    line = log.record("diarize", 86.1)
    assert line == "whisp: diarize 86.1s RTF 0.072"
    assert out.getvalue() == line + "\n"


def test_record_without_audio_duration_omits_rtf():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=0.0, stream=out)
    assert log.record("decode", 8.0) == "whisp: decode 8.0s"


def test_stages_are_accumulated():
    log = timing.StageLog(audio_seconds=100.0, stream=io.StringIO())
    log.record("asr", 21.4)
    log.record("align", 3.1)
    assert log.stages == {"asr": 21.4, "align": 3.1}


def test_stage_context_manager_records_elapsed():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=10.0, stream=out)
    with log.stage("asr"):
        pass
    assert "asr" in log.stages
    assert log.stages["asr"] >= 0.0
    assert out.getvalue().startswith("whisp: asr ")


def test_total_line():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=4840.0, stream=out)
    assert log.total(1215.0) == "whisp: TOTAL 1215.0s RTF 0.251"
