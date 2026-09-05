"""End-to-end summarization against a real local model.

Opt-in, like tests/test_integration.py: these load ~2.1 GiB of weights and
take minutes. Run with:

    WHISP_TEST_TRANSCRIPT=/path/to/transcript.txt ./.venv/bin/pytest -m integration -v
"""

import os
import pathlib

import pytest

from whisp import summarize

pytestmark = pytest.mark.integration

TRANSCRIPT = os.environ.get("WHISP_TEST_TRANSCRIPT")
MODEL = os.environ.get("WHISP_SUMMARY_MODEL", summarize.DEFAULT_MLX_MODEL)

needs_transcript = pytest.mark.skipif(
    not TRANSCRIPT, reason="set WHISP_TEST_TRANSCRIPT to run"
)

RU_HEADERS = ("## Основные темы", "## Решения", "## Задачи")


def backend():
    from whisp import summary_backends

    return summary_backends.default("mlx", model=MODEL)


def longest_verbatim_run(summary: str, source: str, cap: int = 400) -> int:
    """Longest window of the summary that appears verbatim in the transcript."""
    low, high, best = 0, min(cap, len(summary)), 0
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break
        step = max(1, mid // 4)
        if any(summary[i : i + mid] in source for i in range(0, len(summary) - mid + 1, step)):
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


@needs_transcript
def test_real_model_produces_a_well_formed_russian_summary():
    text = pathlib.Path(TRANSCRIPT).read_text(encoding="utf-8")
    summary = summarize.summarize(
        text, backend(), language="ru", chunk_tokens=6000, max_tokens=2048
    )

    for header in RU_HEADERS:
        assert header in summary, f"missing {header}"
    assert [summary.find(h) for h in RU_HEADERS] == sorted(
        summary.find(h) for h in RU_HEADERS
    ), "sections out of order"

    assert "<think>" not in summary and "</think>" not in summary
    assert "```" not in summary

    letters = sum(1 for ch in summary if ch.isalpha()) or 1
    cyrillic = sum(1 for ch in summary if "Ѐ" <= ch <= "ӿ")
    assert cyrillic / letters > 0.5, "summary is not in Russian"

    # A model that degenerates into echoing the transcript still passes every
    # structural check above, so bound the verbatim overlap explicitly.
    assert longest_verbatim_run(summary, text) < 200


@needs_transcript
def test_the_same_transcript_twice_gives_the_same_summary():
    # Sampling is on (greedy decoding makes Qwen3 loop), so reproducibility
    # rests on the fixed seed in summary_backends.
    text = pathlib.Path(TRANSCRIPT).read_text(encoding="utf-8")
    kwargs = dict(language="ru", chunk_tokens=6000, max_tokens=2048)
    first = summarize.summarize(text, backend(), **kwargs)
    second = summarize.summarize(text, backend(), **kwargs)
    assert first == second


@needs_transcript
def test_a_transcript_without_speaker_labels_still_summarizes():
    source = pathlib.Path(TRANSCRIPT).read_text(encoding="utf-8")
    stripped = "\n".join(
        line.split(": ", 1)[-1] if line.startswith("[SPEAKER_") else line
        for line in source.splitlines()
    )
    summary = summarize.summarize(
        stripped, backend(), language="ru", chunk_tokens=6000, max_tokens=2048
    )
    assert all(header in summary for header in RU_HEADERS)
