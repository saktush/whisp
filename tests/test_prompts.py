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
RU_HEADERS = ("## Участники", "## Основные темы", "## Решения", "## Задачи")
EN_HEADERS = ("## Participants", "## Key topics", "## Decisions", "## Action items")


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


@pytest.mark.parametrize("kind", ["single", "reduce"])
@pytest.mark.parametrize("language", ["ru", "en"])
def test_final_templates_ask_for_a_participant_roster_first(language, kind):
    text = prompts.select(language, kind)
    header = "## Участники" if language == "ru" else "## Participants"
    topics = "## Основные темы" if language == "ru" else "## Key topics"
    assert header in text
    assert text.find(header) < text.find(topics), "roster must come first"


@pytest.mark.parametrize("language", ["ru", "en"])
def test_map_template_collects_who_is_speaking(language):
    # The roster is only reconstructable if the map phase records
    # introductions; they occur in the first fragment while the action items
    # land in the last ones.
    text = prompts.select(language, "map")
    marker = "представил" if language == "ru" else "introduce"
    assert marker in text


@pytest.mark.parametrize("language", ["ru", "en"])
def test_final_templates_forbid_inventing_names(language):
    text = prompts.select(language, "single")
    marker = "не выдумывай" if language == "ru" else "never invent"
    assert marker in text.lower()


# --- WHISP_SUMMARY_PROMPT override ---------------------------------------

def test_no_override_returns_the_built_in_template(tmp_path):
    assert prompts.select("ru", "single", None) == prompts.RU_SINGLE


def test_override_directory_replaces_only_the_files_it_contains(tmp_path):
    (tmp_path / "single.txt").write_text("MY SINGLE PROMPT", encoding="utf-8")
    assert prompts.select("ru", "single", tmp_path) == "MY SINGLE PROMPT"
    # The kinds the user did not supply keep our prompts.
    assert prompts.select("ru", "map", tmp_path) == prompts.RU_MAP
    assert prompts.select("ru", "reduce", tmp_path) == prompts.RU_REDUCE


def test_override_can_replace_every_kind(tmp_path):
    for kind in ("single", "map", "reduce"):
        (tmp_path / f"{kind}.txt").write_text(f"CUSTOM {kind}", encoding="utf-8")
    for kind in ("single", "map", "reduce"):
        assert prompts.select("ru", kind, tmp_path) == f"CUSTOM {kind}"


def test_override_applies_regardless_of_language(tmp_path):
    (tmp_path / "single.txt").write_text("ONE PROMPT", encoding="utf-8")
    assert prompts.select("ru", "single", tmp_path) == "ONE PROMPT"
    assert prompts.select("en", "single", tmp_path) == "ONE PROMPT"


def test_an_empty_override_file_is_rejected(tmp_path):
    (tmp_path / "single.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prompts.select("ru", "single", tmp_path)


def test_a_missing_override_directory_is_reported(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        prompts.select("ru", "single", tmp_path / "nope")


def test_pointing_the_override_at_a_file_explains_the_expected_layout(tmp_path):
    plain = tmp_path / "prompt.txt"
    plain.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        prompts.select("ru", "single", plain)
    message = str(excinfo.value)
    assert "directory" in message
    assert "single.txt" in message


@pytest.mark.parametrize("language", ["ru", "en"])
def test_roster_forbids_listing_bare_speaker_labels_as_people(language):
    # Measured: the model emitted "Спикер_00 - представляет команду" alongside
    # the real names, and turned the product "Kaiten" into a participant.
    text = prompts.select(language, "single")
    marker = "метку спикера как отдельного" if language == "ru" else "speaker label as a participant"
    assert marker in text


@pytest.mark.parametrize("kind", ["single", "reduce"])
@pytest.mark.parametrize("language", ["ru", "en"])
def test_topic_list_is_capped(language, kind):
    assert "15" in prompts.select(language, kind)
