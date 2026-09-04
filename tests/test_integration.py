"""Opt-in end-to-end check.

Needs cached models and a real recording, so it is marked `integration` and
skipped unless WHISP_TEST_AUDIO points at a file. Run it with:

    WHISP_TEST_AUDIO=/path/to/recording.webm ./.venv/bin/pytest -m integration -v
"""

import hashlib
import os
import pathlib

import pytest

pytestmark = pytest.mark.integration

AUDIO = os.environ.get("WHISP_TEST_AUDIO")
EXPECTED_MD5 = os.environ.get("WHISP_TEST_MD5")


@pytest.mark.skipif(not AUDIO, reason="set WHISP_TEST_AUDIO to run")
def test_pipeline_reproduces_the_reference_transcript(tmp_path):
    from whisp import devices, pipeline

    result = pipeline.run(
        audio_path=AUDIO,
        output_dir=str(tmp_path),
        language=os.environ.get("WHISP_LANG", "ru"),
        model_name=os.environ.get("WHISP_MODEL", "turbo"),
        compute_type=os.environ.get("WHISP_COMPUTE_TYPE", "int8"),
        device=devices.select_device("auto"),
        hf_token=os.environ["HF_TOKEN"],
        batch_size=8,
        diarize_batch_size=64,
        asr_thread_count=devices.asr_threads(0),
        diarize_enabled=True,
        parallel=True,
    )

    produced = pathlib.Path(result)
    assert produced.exists()
    digest = hashlib.md5(produced.read_bytes()).hexdigest()
    if EXPECTED_MD5:
        assert digest == EXPECTED_MD5, (
            f"transcript changed: {digest} != {EXPECTED_MD5}. "
            "The optimisation must not alter the output."
        )
