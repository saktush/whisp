from unittest import mock

from tests.whisperx_source import whisperx_faster_whisper_threads_default
from whisp import devices


def test_select_device_honours_explicit_request():
    assert devices.select_device("cpu") == "cpu"
    assert devices.select_device("mps") == "mps"


def test_select_device_auto_prefers_mps_when_available():
    with mock.patch.object(devices.torch.backends.mps, "is_available", return_value=True):
        assert devices.select_device("auto") == "mps"
        assert devices.select_device(None) == "mps"


def test_select_device_auto_falls_back_to_cpu():
    with mock.patch.object(devices.torch.backends.mps, "is_available", return_value=False):
        assert devices.select_device("auto") == "cpu"


def test_faster_whisper_threads_extraction_finds_the_constant():
    """Guards the guard: a whisperx refactor could make the parse silently
    return nothing, and the comparison below would then hold vacuously."""
    assert whisperx_faster_whisper_threads_default() is not None


def test_asr_threads_replicates_whisperx_cli_default():
    # whisperx CLI: `--threads 0` means "use its own faster_whisper_threads
    # default", not "use all cores". Compared against the installed
    # whisperx's own source (see tests/whisperx_source.py) rather than a
    # hardcoded literal, because an upgrade changing that default would
    # silently change the transcript otherwise.
    assert devices.asr_threads(0) == whisperx_faster_whisper_threads_default()
    assert devices.asr_threads(6) == 6
