from unittest import mock

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


def test_asr_threads_replicates_whisperx_cli_default():
    # whisperx CLI: `--threads 0` means 4 CTranslate2 threads, not "all cores".
    assert devices.asr_threads(0) == 4
    assert devices.asr_threads(6) == 6
