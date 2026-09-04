"""The ASR options must match what the installed whisperx CLI would build.

Comparing against a hardcoded literal would only catch edits to our own
constants. The real risk is a whisperx upgrade changing one of its argparse
defaults: the transcript would silently stop being byte-identical and
nothing would say so. So the expected values are read out of the installed
whisperx instead of being written down here.
"""

import ast
import pathlib

import numpy as np
import whisperx

from whisp import stages


def whisperx_cli_defaults() -> dict:
    """argparse defaults declared by the installed whisperx CLI.

    Parsed statically: whisperx builds its parser inside cli(), which also
    runs the whole pipeline, so it cannot be imported and inspected.
    """
    source = pathlib.Path(whisperx.__file__).with_name("__main__.py").read_text()
    defaults: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        flag = next(
            (
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ),
            None,
        )
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag[2:]] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass  # e.g. --device, whose default is a torch call
    return defaults


def test_extraction_finds_the_arguments_we_depend_on():
    """Guards the guard: a whisperx refactor could make the parse silently
    return nothing, and every assertion below would pass on empty data."""
    found = whisperx_cli_defaults()
    for name in (
        "beam_size",
        "patience",
        "length_penalty",
        "temperature",
        "temperature_increment_on_fallback",
        "compression_ratio_threshold",
        "logprob_threshold",
        "no_speech_threshold",
        "suppress_tokens",
        "chunk_size",
        "vad_onset",
        "vad_offset",
        "diarize_model",
    ):
        assert name in found, f"whisperx no longer declares --{name}"


def test_asr_options_match_the_installed_whisperx_defaults():
    found = whisperx_cli_defaults()
    assert stages.ASR_OPTIONS["beam_size"] == found["beam_size"]
    assert stages.ASR_OPTIONS["patience"] == found["patience"]
    assert stages.ASR_OPTIONS["length_penalty"] == found["length_penalty"]
    assert (
        stages.ASR_OPTIONS["compression_ratio_threshold"]
        == found["compression_ratio_threshold"]
    )
    assert stages.ASR_OPTIONS["log_prob_threshold"] == found["logprob_threshold"]
    assert stages.ASR_OPTIONS["no_speech_threshold"] == found["no_speech_threshold"]
    assert stages.ASR_OPTIONS["initial_prompt"] == found["initial_prompt"]
    assert stages.ASR_OPTIONS["hotwords"] == found["hotwords"]
    assert stages.ASR_OPTIONS["suppress_tokens"] == [
        int(x) for x in found["suppress_tokens"].split(",")
    ]


def test_temperatures_expand_the_fallback_increment():
    """whisperx turns --temperature 0 plus --temperature_increment_on_fallback
    0.2 into a six-value tuple, not [0]. Getting this wrong changes decoding
    and therefore the transcript."""
    found = whisperx_cli_defaults()
    expected = tuple(
        np.arange(
            found["temperature"],
            1.0 + 1e-6,
            found["temperature_increment_on_fallback"],
        )
    )
    assert stages.ASR_OPTIONS["temperatures"] == expected
    assert len(expected) == 6


def test_condition_on_previous_text_is_forced_off():
    """whisperx hardcodes False in transcribe.py regardless of the CLI flag."""
    assert stages.ASR_OPTIONS["condition_on_previous_text"] is False


def test_suppress_numerals_defaults_to_false():
    """A store_true flag, so it declares no argparse default at all."""
    assert "suppress_numerals" not in whisperx_cli_defaults()
    assert stages.ASR_OPTIONS["suppress_numerals"] is False


def test_vad_options_match_the_installed_whisperx_defaults():
    found = whisperx_cli_defaults()
    assert stages.VAD_OPTIONS == {
        "chunk_size": found["chunk_size"],
        "vad_onset": found["vad_onset"],
        "vad_offset": found["vad_offset"],
    }


def test_default_diarization_model_matches_cli():
    assert stages.DIARIZE_MODEL == whisperx_cli_defaults()["diarize_model"]
