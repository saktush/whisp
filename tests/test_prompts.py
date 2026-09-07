"""Prompt template selection.

The templates are the entire user-visible contract of the summary, so the
tests pin the things that silently break it: the section headers a downstream
reader greps for, the language fallback, and the map/reduce split (a map
template that emits the final headers makes the model invent decisions from
partial context).
"""

import pytest

from whisp import prompts

KINDS = ("single", "map", "reduce")
RU_HEADERS = ("## Основные темы", "## Решения", "## Задачи")
EN_HEADERS = ("## Key topics", "## Decisions", "## Action items")


@pytest.mark.parametrize("language", ["ru", "RU", "ru-RU", "russian"])
def test_russian_language_codes_select_russian(language):
    assert prompts.select(language, "single").count("## Основные темы") == 1


@pytest.mark.parametrize("language", ["en", "de", "fr", "", None, "zz"])
def test_everything_other_than_russian_falls_back_to_english(language):
    text = prompts.select(language, "single")
    assert "## Key topics" in text
    assert "## Основные темы" not in text


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("language", ["ru", "en"])
def test_every_language_and_kind_combination_exists(language, kind):
    assert prompts.select(language, kind).strip()


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        prompts.select("ru", "nonsense")


@pytest.mark.parametrize(
    "language,headers", [("ru", RU_HEADERS), ("en", EN_HEADERS)]
)
@pytest.mark.parametrize("kind", ["single", "reduce"])
def test_final_output_templates_carry_all_three_headers_in_order(language, kind, headers):
    text = prompts.select(language, kind)
    positions = [text.find(h) for h in headers]
    assert all(p >= 0 for p in positions), f"missing header in {language}/{kind}"
    assert positions == sorted(positions), "headers out of order"


@pytest.mark.parametrize(
    "language,headers", [("ru", RU_HEADERS), ("en", EN_HEADERS)]
)
def test_map_template_does_not_emit_the_final_headers(language, headers):
    # A per-chunk summary in the final format invents decisions from partial
    # context and strips the attribution the reduce pass needs.
    text = prompts.select(language, "map")
    for header in headers:
        assert header not in text


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("language", ["ru", "en"])
def test_no_template_carries_the_claude_cli_tool_guard(language, kind):
    # "Do not use any tools" was a Claude-Code-specific guard; it is noise for
    # a plain local model.
    assert "use any tools" not in prompts.select(language, kind).lower()


@pytest.mark.parametrize("kind", ["single", "reduce"])
@pytest.mark.parametrize("language", ["ru", "en"])
def test_final_templates_pin_the_bullet_syntax(language, kind):
    # Without this the model sometimes returns "Основные темы" as one
    # comma-separated run-on paragraph instead of a list.
    text = prompts.select(language, kind)
    assert '- ' in text
    marker = "отдельной строкой" if language == "ru" else "its own line"
    assert marker in text


@pytest.mark.parametrize("kind", ["single", "reduce"])
@pytest.mark.parametrize("language", ["ru", "en"])
def test_final_templates_forbid_overlapping_sections(language, kind):
    # Measured: without this the model listed the same five items under both
    # Decisions and Action items.
    text = prompts.select(language, kind)
    marker = "не пересекаются" if language == "ru" else "do not overlap"
    assert marker in text
