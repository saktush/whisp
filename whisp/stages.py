"""Thin wrappers around whisperx, one per pipeline stage.

Every option below is copied verbatim from what whisperx's own CLI assembles
for the arguments whisp.sh passes (see whisperx/transcribe.py and
whisperx/__main__.py). They are reproduced here rather than re-derived
because the transcript must stay byte-identical to the pre-optimisation
output; the acceptance test is an md5 comparison.
"""

import numpy as np
import whisperx
from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.diarize import DiarizationPipeline

DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

# whisperx CLI: --temperature 0 with --temperature_increment_on_fallback 0.2
ASR_OPTIONS = {
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "temperatures": tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    # whisperx hardcodes this to False regardless of the CLI flag.
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "hotwords": None,
    "suppress_tokens": [-1],
    "suppress_numerals": False,
}

VAD_OPTIONS = {
    "chunk_size": 30,
    "vad_onset": 0.500,
    "vad_offset": 0.363,
}


def decode(audio_path: str):
    """Decode once; every stage reuses this array."""
    return load_audio(audio_path)


def duration_seconds(audio) -> float:
    return len(audio) / SAMPLE_RATE


def load_asr(model_name: str, compute_type: str, language: str, threads: int, hf_token: str):
    return whisperx.load_model(
        model_name,
        device="cpu",  # CTranslate2 has no Metal backend
        device_index=0,
        compute_type=compute_type,
        language=language,
        asr_options=dict(ASR_OPTIONS),
        vad_method="pyannote",
        vad_options=dict(VAD_OPTIONS),
        task="transcribe",
        local_files_only=False,
        threads=threads,
        use_auth_token=hf_token,
    )


def transcribe(asr_model, audio, batch_size: int, language: str) -> dict:
    return asr_model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        chunk_size=VAD_OPTIONS["chunk_size"],
        # whisperx CLI --verbose default is True; FasterWhisperPipeline.transcribe's
        # own default is False. This flag gates the per-segment progress print in
        # whisperx/asr.py, which is the only in-flight signal during a long ASR run.
        verbose=True,
    )


def align_segments(segments, audio, language: str, device: str) -> dict:
    model, metadata = whisperx.load_align_model(language_code=language, device=device)
    try:
        return whisperx.align(
            segments,
            model,
            metadata,
            audio,
            device,
            interpolate_method="nearest",
            return_char_alignments=False,
        )
    finally:
        del model


def diarize(audio, device: str, hf_token: str, batch_size: int, model_name: str = DIARIZE_MODEL):
    pipeline = DiarizationPipeline(
        model_name=model_name, token=hf_token, device=device
    )
    # Measured on M1 Pro / 16 GB: batch 64 gives RTF 0.067 against 0.082 at the
    # shipped default of 32. Batch 128 regresses to 0.115 -- it runs out of
    # unified memory -- so 64 is the ceiling, not a starting point.
    pipeline.model.segmentation_batch_size = batch_size
    pipeline.model.embedding_batch_size = batch_size
    try:
        return pipeline(audio)
    finally:
        del pipeline
