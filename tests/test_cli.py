import pytest

from whisp import __main__ as cli


def test_defaults_come_from_environment():
    args = cli.parse_args(
        ["audio.webm"],
        env={
            "WHISP_LANG": "en",
            "WHISP_MODEL": "large-v3",
            "WHISP_COMPUTE_TYPE": "float32",
            "WHISP_BATCH_SIZE": "16",
            "WHISP_DIARIZE_BATCH_SIZE": "32",
            "WHISP_PARALLEL": "0",
            "WHISP_DEVICE": "cpu",
            "WHISP_ASR_THREADS": "6",
        },
    )
    assert args.language == "en"
    assert args.model == "large-v3"
    assert args.compute_type == "float32"
    assert args.batch_size == 16
    assert args.diarize_batch_size == 32
    assert args.parallel is False
    assert args.device == "cpu"
    assert args.threads == 6


def test_defaults_without_environment_match_current_behaviour():
    args = cli.parse_args(["audio.webm"], env={})
    assert args.language == "ru"
    assert args.model == "turbo"
    assert args.compute_type == "int8"
    assert args.batch_size == 8
    assert args.diarize_batch_size == 64
    assert args.parallel is True
    assert args.device == "auto"
    assert args.threads == 0
    assert args.diarize is True


def test_no_diarize_flag():
    args = cli.parse_args(["audio.webm", "--no-diarize"], env={})
    assert args.diarize is False


def test_output_dir_defaults_to_the_source_directory():
    args = cli.parse_args(["/recordings/meeting.webm"], env={})
    assert args.output_dir == "/recordings"
