"""The ASR options must match what the installed whisperx CLI would build.

Comparing against a hardcoded literal would only catch edits to our own
constants. The real risk is a whisperx upgrade changing one of its argparse
defaults: the transcript would silently stop being byte-identical and
nothing would say so. So the expected values are read out of the installed
whisperx instead of being written down here.
"""

from unittest.mock import MagicMock

import numpy as np
import whisperx

from tests.whisperx_source import whisperx_cli_defaults
from whisp import stages


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


# --- Wrapper call-site tests -------------------------------------------------
#
# Everything above only checks the two constant dicts; none of them ever call
# load_asr/transcribe/align_segments/diarize. That gap is how a missing
# `verbose` or `chunk_size` argument (see whisp/stages.py transcribe()) went
# unnoticed. These tests patch whisperx's entry points and assert the exact
# keyword arguments each wrapper passes through, so a dropped or wrong
# argument fails a test instead of silently changing the transcript. Nothing
# real is constructed: every whisperx call is replaced with a MagicMock
# before the wrapper runs.


def test_load_asr_passes_options_verbatim(monkeypatch):
    fake_model = object()
    load_model = MagicMock(return_value=fake_model)
    monkeypatch.setattr(whisperx, "load_model", load_model)

    result = stages.load_asr(
        model_name="large-v2",
        compute_type="int8",
        language="ru",
        threads=4,
        hf_token="hf-token",
    )

    assert result is fake_model
    load_model.assert_called_once_with(
        "large-v2",
        device="cpu",
        device_index=0,
        compute_type="int8",
        language="ru",
        asr_options=stages.ASR_OPTIONS,
        vad_method="pyannote",
        vad_options=stages.VAD_OPTIONS,
        task="transcribe",
        local_files_only=False,
        threads=4,
        use_auth_token="hf-token",
    )


def test_transcribe_passes_verbose_and_chunk_size(monkeypatch):
    """Regression test for the two dropped arguments: the CLI calls
    model.transcribe(..., chunk_size=..., print_progress=..., verbose=...),
    and its --verbose default is True. Omitting verbose silently falls back
    to FasterWhisperPipeline.transcribe's own default of False, which mutes
    the per-segment progress print that is the only in-flight signal during
    a long ASR run. Omitting chunk_size relies on that method's own default
    (currently 30, same as VAD_OPTIONS["chunk_size"]) instead of tying the
    two together explicitly.
    """
    audio = np.zeros(10, dtype=np.float32)
    fake_result = {"segments": []}
    asr_model = MagicMock()
    asr_model.transcribe.return_value = fake_result

    result = stages.transcribe(asr_model, audio, batch_size=16, language="ru")

    assert result is fake_result
    asr_model.transcribe.assert_called_once_with(
        audio,
        batch_size=16,
        language="ru",
        chunk_size=stages.VAD_OPTIONS["chunk_size"],
        verbose=True,
    )


def test_align_segments_passes_arguments_verbatim(monkeypatch):
    fake_align_model = object()
    fake_metadata = object()
    load_align_model = MagicMock(return_value=(fake_align_model, fake_metadata))
    fake_aligned = {"segments": []}
    align = MagicMock(return_value=fake_aligned)
    monkeypatch.setattr(whisperx, "load_align_model", load_align_model)
    monkeypatch.setattr(whisperx, "align", align)

    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    audio = np.zeros(10, dtype=np.float32)

    result = stages.align_segments(segments, audio, language="ru", device="cpu")

    assert result is fake_aligned
    load_align_model.assert_called_once_with(language_code="ru", device="cpu")
    align.assert_called_once_with(
        segments,
        fake_align_model,
        fake_metadata,
        audio,
        "cpu",
        interpolate_method="nearest",
        return_char_alignments=False,
    )


def test_diarize_sets_batch_sizes_and_passes_arguments_verbatim(monkeypatch):
    fake_pipeline = MagicMock()
    fake_result = MagicMock(name="diarization-dataframe")
    fake_pipeline.return_value = fake_result
    diarization_pipeline_cls = MagicMock(return_value=fake_pipeline)
    monkeypatch.setattr(stages, "DiarizationPipeline", diarization_pipeline_cls)

    audio = np.zeros(10, dtype=np.float32)
    result = stages.diarize(
        audio, device="cpu", hf_token="hf-token", batch_size=64, model_name="custom-model"
    )

    assert result is fake_result
    diarization_pipeline_cls.assert_called_once_with(
        model_name="custom-model", token="hf-token", device="cpu"
    )
    assert fake_pipeline.model.segmentation_batch_size == 64
    assert fake_pipeline.model.embedding_batch_size == 64
    fake_pipeline.assert_called_once_with(audio)


def test_diarize_defaults_to_the_module_diarization_model(monkeypatch):
    fake_pipeline = MagicMock()
    diarization_pipeline_cls = MagicMock(return_value=fake_pipeline)
    monkeypatch.setattr(stages, "DiarizationPipeline", diarization_pipeline_cls)

    audio = np.zeros(10, dtype=np.float32)
    stages.diarize(audio, device="cpu", hf_token=None, batch_size=32)

    diarization_pipeline_cls.assert_called_once_with(
        model_name=stages.DIARIZE_MODEL, token=None, device="cpu"
    )
