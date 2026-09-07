"""Orchestration and output sanitizing, exercised with a fake backend.

Mirrors how tests/test_pipeline.py drives pipeline.run with fake Stages: the
point of the Backend dataclass is that none of this needs a model.
"""

import pytest

from whisp import summarize

FINAL = "## Участники\n- Иван (аналитик)\n\n## Основные темы\n- тема\n\n## Решения\n- Нет\n\n## Задачи\n- Нет"


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
    assert "## Участники" in system  # the single-pass template
    assert out == FINAL


def test_long_transcript_maps_every_chunk_then_reduces_once():
    calls = []
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    run(text, make_backend(calls=calls), chunk_tokens=12)
    systems = [s for s, _ in calls]
    assert len(calls) >= 3, "expected several map calls plus a reduce"
    assert all("Фрагмент" in s or "фрагмент" in s.lower() for s in systems[:-1])
    assert "## Участники" in systems[-1], "last call must be the reduce template"


def test_map_output_reaches_the_reduce_prompt():
    calls = []
    text = "".join(f"line {i} with several words here\n" for i in range(30))

    def reply(system, user):
        return FINAL if "## Участники" in system else "NOTE-MARKER"

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
    assert summarize.sanitize(FINAL).startswith("## Участники")


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


# --- generation budget ----------------------------------------------------

def test_map_calls_get_a_smaller_budget_than_the_final_summary():
    # Map notes are intermediate, and the map phase runs once per chunk, so it
    # dominates wall time. Measured: 4 chunks allowed 2048 output tokens each
    # cost more than the reduce pass and every prefill combined.
    budgets = []

    def generate(system, user, max_tokens):
        budgets.append(max_tokens)
        return FINAL

    backend = summarize.Backend(
        generate=generate, count_tokens=lambda s: len(s.split()), name="fake"
    )
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    summarize.summarize(
        text, backend, language="ru", chunk_tokens=12, max_tokens=2048
    )

    *map_budgets, reduce_budget = budgets
    assert map_budgets, "expected several map calls"
    assert reduce_budget == 2048
    assert all(b < reduce_budget for b in map_budgets)


def test_a_small_max_tokens_is_never_raised_for_the_map_phase():
    budgets = []

    def generate(system, user, max_tokens):
        budgets.append(max_tokens)
        return FINAL

    backend = summarize.Backend(
        generate=generate, count_tokens=lambda s: len(s.split()), name="fake"
    )
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    summarize.summarize(text, backend, language="ru", chunk_tokens=12, max_tokens=256)
    assert all(b <= 256 for b in budgets)


def test_single_pass_uses_the_full_budget():
    budgets = []

    def generate(system, user, max_tokens):
        budgets.append(max_tokens)
        return FINAL

    backend = summarize.Backend(
        generate=generate, count_tokens=lambda s: len(s.split()), name="fake"
    )
    summarize.summarize(
        "[SPEAKER_00]: коротко\n", backend, language="ru", chunk_tokens=9999, max_tokens=2048
    )
    assert budgets == [2048]


# --- prompt override wiring ----------------------------------------------

def test_summarize_uses_an_override_prompt_when_given_one(tmp_path):
    (tmp_path / "single.txt").write_text("MY OWN PROMPT", encoding="utf-8")
    calls = []
    run("[SPEAKER_00]: коротко\n", make_backend(calls=calls), prompt_dir=tmp_path)
    system, _ = calls[0]
    assert system == "MY OWN PROMPT"


def test_summarize_keeps_our_prompts_for_kinds_the_override_omits(tmp_path):
    (tmp_path / "single.txt").write_text("ONLY SINGLE", encoding="utf-8")
    calls = []
    text = "".join(f"line {i} with several words here\n" for i in range(30))
    run(text, make_backend(calls=calls), chunk_tokens=12, prompt_dir=tmp_path)
    systems = [s for s, _ in calls]
    assert "ONLY SINGLE" not in systems, "single.txt must not leak into map/reduce"
    assert "## Участники" in systems[-1], "reduce keeps the built-in prompt"


# --- truncation ------------------------------------------------------------

TRUNCATED = "## Участники\n- Иван\n\n## Основные темы\n- тема\n\n## Решения\n- начали делать"


def test_a_summary_missing_a_section_is_rejected():
    # A generation that hits max_tokens stops mid-way. The old check only
    # asked whether any "## " was present, so a summary with its last section
    # cut off was written out as if it were complete.
    with pytest.raises(summarize.SummaryError, match="Задачи"):
        run("[SPEAKER_00]: текст\n", make_backend(reply=TRUNCATED))


def test_a_complete_summary_passes_the_section_check():
    assert run("[SPEAKER_00]: текст\n", make_backend()) == FINAL


def test_english_summaries_are_checked_against_english_headers():
    english = ("## Participants\n- Ivan\n\n## Key topics\n- t\n\n"
               "## Decisions\n- None\n\n## Action items\n- None")
    assert run("[SPEAKER_00]: text\n", make_backend(reply=english), language="en") == english


def test_custom_prompts_skip_the_section_check(tmp_path):
    # With the user's own prompt we cannot know the expected headings, so only
    # the non-empty check applies.
    (tmp_path / "single.txt").write_text("MY PROMPT", encoding="utf-8")
    out = run("[SPEAKER_00]: текст\n", make_backend(reply="## Whatever\n- x"),
              prompt_dir=tmp_path)
    assert out == "## Whatever\n- x"
