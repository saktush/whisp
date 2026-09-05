"""Device selection for the torch-backed stages.

CTranslate2 (the ASR engine) has no Metal backend, so ASR always runs on CPU.
Diarization and alignment are plain torch and do run on MPS, which is why the
pipeline picks devices per stage instead of using one global device.
"""

import torch


def select_device(requested: str | None = None) -> str:
    """Device for diarization and alignment.

    "auto" (the default) picks MPS when the machine has it and falls back to
    CPU otherwise, so the pipeline keeps working on Intel Macs and in CI.
    """
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def asr_threads(requested: int = 0) -> int:
    """CTranslate2 thread count, replicating the whisperx CLI.

    whisperx treats `--threads 0` as "leave the default", which is 4 -- not
    "use every core". Reproducing that exactly is what keeps transcripts
    byte-identical to the pre-optimisation output.
    """
    return requested if requested > 0 else 4
