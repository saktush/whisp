"""CLI defaults and exit behaviour.

Mirrors tests/test_cli.py: parse_args takes an injectable env dict so the
documented defaults are asserted without touching os.environ.
"""

import subprocess
import sys

import pytest

from whisp import summarize


def parse(argv, env=None):
    return summarize.parse_args(argv, env or {})


# --- defaults -------------------------------------------------------------

def test_defaults_without_any_environment():
    args = parse(["talk.txt", "--output", "out.txt"])
    assert args.backend == "mlx"
    assert args.language == "ru"
    assert args.model == summarize.DEFAULT_MLX_MODEL
    assert args.timeout == 1800.0


def test_every_default_can_come_from_the_environment():
    env = {
        "WHISP_SUMMARY_BACKEND": "claude",
        "WHISP_SUMMARY_MODEL": "opus",
        "WHISP_SUMMARY_LANG": "en",
        "WHISP_SUMMARY_CHUNK_TOKENS": "4321",
        "WHISP_SUMMARY_MAX_TOKENS": "999",
        "WHISP_SUMMARY_TIMEOUT": "60",
    }
    args = parse(["talk.txt", "--output", "out.txt"], env)
    assert args.backend == "claude"
    assert args.model == "opus"
    assert args.language == "en"
    assert args.chunk_tokens == 4321
    assert args.max_tokens == 999
    assert args.timeout == 60.0


def test_summary_language_falls_back_to_the_transcription_language():
    args = parse(["talk.txt", "--output", "out.txt"], {"WHISP_LANG": "en"})
    assert args.language == "en"


def test_explicit_summary_language_wins_over_the_transcription_language():
    env = {"WHISP_LANG": "en", "WHISP_SUMMARY_LANG": "ru"}
    assert parse(["talk.txt", "--output", "out.txt"], env).language == "ru"


def test_claude_backend_defaults_to_sonnet_not_an_mlx_repo():
    args = parse(["talk.txt", "--output", "out.txt"], {"WHISP_SUMMARY_BACKEND": "claude"})
    assert args.model == "sonnet"


def test_command_line_flags_beat_the_environment():
    env = {"WHISP_SUMMARY_BACKEND": "claude", "WHISP_SUMMARY_LANG": "en"}
    args = parse(["talk.txt", "--output", "o.txt", "--backend", "mlx", "--language", "ru"], env)
    assert args.backend == "mlx"
    assert args.language == "ru"


# --- run ------------------------------------------------------------------

def fake_backend(reply):
    return summarize.Backend(
        generate=lambda system, user, max_tokens: reply,
        count_tokens=lambda text: len(text.split()),
        name="fake",
    )


def test_run_writes_the_summary_to_the_output_path(tmp_path):
    transcript = tmp_path / "talk.txt"
    transcript.write_text("[SPEAKER_00]: привет\n", encoding="utf-8")
    out = tmp_path / "talk-summary.txt"
    final = "## Участники\n- Иван (аналитик)\n\n## Основные темы\n- тема\n\n## Решения\n- Нет\n\n## Задачи\n- Нет"

    code = summarize.run(
        parse([str(transcript), "--output", str(out)]), backend=fake_backend(final)
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").rstrip("\n") == final


def test_run_reports_failure_and_writes_nothing(tmp_path, capsys):
    transcript = tmp_path / "talk.txt"
    transcript.write_text("[SPEAKER_00]: привет\n", encoding="utf-8")
    out = tmp_path / "talk-summary.txt"

    code = summarize.run(
        parse([str(transcript), "--output", str(out)]),
        backend=fake_backend("sorry, no sections here"),
    )
    assert code != 0
    assert not out.exists(), "a failed summary must not leave a file behind"
    assert "whisp:" in capsys.readouterr().err


def test_run_rejects_a_missing_transcript(tmp_path, capsys):
    code = summarize.run(
        parse([str(tmp_path / "nope.txt"), "--output", str(tmp_path / "o.txt")]),
        backend=fake_backend("x"),
    )
    assert code != 0
    assert "whisp:" in capsys.readouterr().err


# --- the invariant that keeps the two allocators apart --------------------

def test_importing_the_summarizer_does_not_pull_in_torch():
    # whisp.summarize runs in its own process precisely so torch-MPS and MLX
    # never share one address space. An accidental import here would undo that.
    probe = (
        "import sys, whisp.summarize;"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m);"
        "assert 'whisperx' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, result.stderr


# --- prompt override ------------------------------------------------------

def test_prompt_override_defaults_to_none_so_our_prompts_are_used():
    assert parse(["talk.txt", "--output", "o.txt"]).prompt_dir is None


def test_prompt_override_comes_from_the_environment():
    args = parse(["talk.txt", "--output", "o.txt"], {"WHISP_SUMMARY_PROMPT": "/tmp/mine"})
    assert str(args.prompt_dir) == "/tmp/mine"


def test_prompt_override_flag_beats_the_environment():
    args = parse(
        ["talk.txt", "--output", "o.txt", "--prompt-dir", "/tmp/flag"],
        {"WHISP_SUMMARY_PROMPT": "/tmp/env"},
    )
    assert str(args.prompt_dir) == "/tmp/flag"


def test_a_bad_prompt_override_fails_the_run_without_writing(tmp_path, capsys):
    transcript = tmp_path / "talk.txt"
    transcript.write_text("[SPEAKER_00]: привет\n", encoding="utf-8")
    out = tmp_path / "talk-summary.txt"
    args = parse(
        [str(transcript), "--output", str(out), "--prompt-dir", str(tmp_path / "missing")]
    )
    code = summarize.run(args, backend=fake_backend("x"))
    assert code != 0
    assert not out.exists()
    assert "WHISP_SUMMARY_PROMPT" in capsys.readouterr().err


def test_print_prompt_writes_the_built_in_template_and_exits_ok(capsys):
    from whisp import prompts

    code = summarize.run(parse(["--print-prompt", "map", "--language", "ru"]), backend=None)
    assert code == 0
    assert capsys.readouterr().out.strip() == prompts.RU_MAP.strip()


def test_print_prompt_does_not_need_a_transcript_or_output():
    args = parse(["--print-prompt", "single"])
    assert args.print_prompt == "single"


def test_print_prompt_rejects_an_unknown_kind(capsys):
    code = summarize.run(parse(["--print-prompt", "nonsense"]), backend=None)
    assert code != 0
    assert "single" in capsys.readouterr().err
