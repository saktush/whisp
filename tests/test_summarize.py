"""Transcript chunking.

Chunking is what keeps a long meeting inside the model's context, so its
failure modes are silent: a dropped line is a dropped topic, and a chunk over
budget is a truncated one. The tokenizer is injected so none of this needs a
model.
"""

import pytest

from whisp import summarize


def words(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def test_transcript_under_the_limit_is_a_single_chunk():
    text = "[SPEAKER_00]: раз два\n[SPEAKER_01]: три четыре\n"
    assert summarize.chunk_lines(text, max_tokens=100, count_tokens=words) == [text]


def test_chunks_never_split_a_line():
    text = "".join(f"[SPEAKER_0{i % 2}]: line number {i} here\n" for i in range(40))
    chunks = summarize.chunk_lines(text, max_tokens=20, count_tokens=words)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.splitlines():
            assert line in text.splitlines()


def test_chunks_concatenate_back_to_the_original():
    text = "".join(f"[SPEAKER_00]: utterance {i}\n" for i in range(50))
    chunks = summarize.chunk_lines(text, max_tokens=15, count_tokens=words)
    assert "".join(chunks) == text


def test_chunks_respect_the_token_budget():
    text = "".join(f"[SPEAKER_00]: a b c d e f\n" for _ in range(30))
    budget = 25
    chunks = summarize.chunk_lines(text, max_tokens=budget, count_tokens=words)
    for chunk in chunks:
        assert words(chunk) <= budget


def test_transcript_without_speaker_labels_still_chunks():
    # pipeline.run degrades to a transcript with no [SPEAKER_nn] prefixes when
    # diarization or alignment fails; one such file is 4308 lines long.
    text = "".join(f"plain wrapped line {i}\n" for i in range(60))
    chunks = summarize.chunk_lines(text, max_tokens=20, count_tokens=words)
    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_a_single_line_over_budget_is_emitted_alone_not_dropped():
    long_line = "[SPEAKER_00]: " + " ".join(str(i) for i in range(100)) + "\n"
    text = "short one\n" + long_line + "short two\n"
    chunks = summarize.chunk_lines(text, max_tokens=10, count_tokens=words)
    assert "".join(chunks) == text
    assert any(long_line in c for c in chunks)


def test_empty_input_yields_no_chunks():
    assert summarize.chunk_lines("", max_tokens=100, count_tokens=words) == []
    assert summarize.chunk_lines("   \n\n", max_tokens=100, count_tokens=words) == []


def test_tokenizer_is_called_once_per_line_not_per_growing_prefix():
    # The largest real transcript is 4308 lines; re-encoding the growing chunk
    # would make this quadratic.
    calls = []

    def counting(text: str) -> int:
        calls.append(text)
        return words(text)

    line_count = 60
    text = "".join(f"line {i}\n" for i in range(line_count))
    summarize.chunk_lines(text, max_tokens=10, count_tokens=counting)
    assert len(calls) <= line_count
