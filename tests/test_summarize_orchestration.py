"""Orchestration and output sanitizing, exercised with a fake backend.

Mirrors how tests/test_pipeline.py drives pipeline.run with fake Stages: the
point of the Backend dataclass is that none of this needs a model.
"""

import pytest

from whisp import summarize

FINAL = "## Основные темы\n- тема\n\n## Решения\n- Нет\n\n## Задачи\n- Нет"


def make_backend(reply=None, calls=None, count_tokens=None):
    """Fake backend: records (system, user) and returns a canned reply."""
    calls = [] if calls is None else calls

    def generate(system, user, max_tokens):
        calls.append((system, user))
        return reply(system, user) if callable(reply) else (reply or FINAL)

    return summarize.Backend(
        generate=generate,
        count_tokens=count_tokens or (lambda s: len(s.split())),
        name="fake",
    )


def run(text, backend, **kw):
    kw.setdefault("language", "ru")
    kw.setdefault("chunk_tokens", 1000)
    kw.setdefault("max_tokens", 512)
    return summarize.summarize(text, backend, **kw)


# --- strategy selection ---------------------------------------------------

def test_short_transcript_takes_the_single_pass_path():
    calls = []
    out = run("[SPEAKER_00]: коротко\n", make_backend(calls=calls))
    assert len(calls) == 1
    system, _ = calls[0]
    assert "## Основные темы" in system  # the single-pass template
    assert out == FINAL


def test_long_transcript_maps_every_chunk_then_reduces_once():
    calls = []
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    run(text, make_backend(calls=calls), chunk_tokens=12)
    systems = [s for s, _ in calls]
    assert len(calls) >= 3, "expected several map calls plus a reduce"
    assert all("Фрагмент" in s or "фрагмент" in s.lower() for s in systems[:-1])
    assert "## Основные темы" in systems[-1], "last call must be the reduce template"


def test_map_output_reaches_the_reduce_prompt():
    calls = []
    text = "".join(f"line {i} with several words here\n" for i in range(30))

    def reply(system, user):
        return FINAL if "## Основные темы" in system else "NOTE-MARKER"

    run(text, make_backend(reply=reply, calls=calls), chunk_tokens=12)
    _, reduce_user = calls[-1]
    assert "NOTE-MARKER" in reduce_user


def test_single_chunk_never_uses_the_map_template():
    calls = []
    run("[SPEAKER_00]: коротко\n", make_backend(calls=calls), chunk_tokens=10_000)
    assert len(calls) == 1


# --- sanitizing -----------------------------------------------------------

def test_sanitize_leaves_a_clean_summary_untouched():
    assert summarize.sanitize(FINAL) == FINAL


def test_sanitize_strips_think_blocks():
    assert summarize.sanitize(f"<think>\nвслух рассуждаю\n</think>\n\n{FINAL}") == FINAL


def test_sanitize_strips_code_fences():
    assert summarize.sanitize(f"```markdown\n{FINAL}\n```") == FINAL


def test_sanitize_strips_preamble_before_the_first_section():
    assert summarize.sanitize(f"Конечно! Вот резюме:\n\n{FINAL}") == FINAL


def test_sanitize_does_not_cut_into_a_summary_that_starts_at_position_zero():
    assert summarize.sanitize(FINAL).startswith("## Основные темы")


# --- failure modes --------------------------------------------------------

def test_empty_backend_output_raises_rather_than_writing_an_empty_summary():
    with pytest.raises(summarize.SummaryError):
        run("[SPEAKER_00]: текст\n", make_backend(reply="   \n  "))


def test_output_with_no_sections_raises():
    with pytest.raises(summarize.SummaryError):
        run("[SPEAKER_00]: текст\n", make_backend(reply="I could not do it, sorry."))


def test_a_backend_exception_propagates_and_writes_nothing():
    def boom(system, user, max_tokens):
        raise RuntimeError("model exploded")

    backend = summarize.Backend(generate=boom, count_tokens=lambda s: 1, name="boom")
    with pytest.raises(RuntimeError, match="model exploded"):
        run("[SPEAKER_00]: текст\n", backend)


def test_deadline_exceeded_mid_map_raises_summary_timeout():
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    ticks = iter([0.0, 0.0] + [999.0] * 50)

    with pytest.raises(summarize.SummaryTimeout):
        run(
            text,
            make_backend(),
            chunk_tokens=12,
            deadline=10.0,
            monotonic=lambda: next(ticks),
        )


def test_stage_log_receives_map_and_reduce_lines():
    import io

    from whisp.timing import StageLog

    stream = io.StringIO()
    log = StageLog(audio_seconds=0, stream=stream)
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    run(text, make_backend(), chunk_tokens=12, log=log)

    emitted = stream.getvalue()
    assert "summary-map" in emitted
    assert "summary-reduce" in emitted
    assert "RTF" not in emitted, "no audio duration here, so no RTF column"
