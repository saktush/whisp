"""Backend construction, dispatch, and the API contract with mlx-lm."""

import subprocess

import pytest

from whisp import summarize, summary_backends
from tests import mlx_lm_source


class FakeTokenizer:
    chat_template = "{% if enable_thinking is defined %}x{% endif %}"

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        self.template_kwargs = kwargs
        return "PROMPT:" + messages[-1]["content"]

    def encode(self, text):
        return text.split()


# --- dispatch -------------------------------------------------------------

def test_unknown_backend_name_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        summary_backends.default("ollama", model="whatever")
    assert "mlx" in str(excinfo.value) and "claude" in str(excinfo.value)


def test_claude_backend_is_built_without_touching_mlx():
    backend = summary_backends.default("claude", model="sonnet")
    assert backend.name == "claude"


def test_mlx_backend_is_built_lazily_without_loading_the_model(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        summary_backends, "_load_mlx", lambda m: loaded.append(m) or (object(), FakeTokenizer())
    )
    backend = summary_backends.default("mlx", model="some/model")
    assert backend.name == "mlx"
    assert loaded == [], "constructing the backend must not load weights"


# --- mlx backend ----------------------------------------------------------

def test_missing_mlx_lm_produces_an_actionable_message_not_an_import_error(monkeypatch):
    def missing(model_id):
        raise ImportError("No module named 'mlx_lm'")

    monkeypatch.setattr(summary_backends, "_load_mlx", missing)
    backend = summary_backends.default("mlx", model="some/model")

    with pytest.raises(summarize.SummaryError) as excinfo:
        backend.count_tokens("hello")
    message = str(excinfo.value)
    assert "mlx-lm" in message
    assert "WHISP_SUMMARY_BACKEND=claude" in message


def test_mlx_backend_disables_thinking_when_the_template_supports_it(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(summary_backends, "_load_mlx", lambda m: (object(), tokenizer))
    monkeypatch.setattr(summary_backends, "_stream_text", lambda *a, **k: "## Темы\n- x")

    backend = summary_backends.default("mlx", model="some/model")
    backend.generate("system", "user", 128)
    assert tokenizer.template_kwargs == {"enable_thinking": False}


def test_mlx_backend_omits_thinking_kwarg_when_the_template_lacks_it(monkeypatch):
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = "a template with no such switch"
    monkeypatch.setattr(summary_backends, "_load_mlx", lambda m: (object(), tokenizer))
    monkeypatch.setattr(summary_backends, "_stream_text", lambda *a, **k: "## Темы\n- x")

    backend = summary_backends.default("mlx", model="some/model")
    backend.generate("system", "user", 128)
    assert tokenizer.template_kwargs == {}


def test_mlx_backend_loads_the_model_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        summary_backends,
        "_load_mlx",
        lambda m: (calls.append(m), (object(), FakeTokenizer()))[1],
    )
    monkeypatch.setattr(summary_backends, "_stream_text", lambda *a, **k: "## Темы\n- x")

    backend = summary_backends.default("mlx", model="some/model")
    backend.count_tokens("a b c")
    backend.generate("system", "user", 128)
    backend.generate("system", "user", 128)
    assert len(calls) == 1


# --- claude backend -------------------------------------------------------

def test_claude_backend_builds_the_expected_argv_and_detaches_the_child(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="## Темы\n- x", stderr="")

    monkeypatch.setattr(summary_backends.subprocess, "run", fake_run)
    backend = summary_backends.default("claude", model="sonnet")
    out = backend.generate("SYSTEM", "USER", 512)

    assert captured["argv"][:2] == ["claude", "-p"]
    assert "--system-prompt" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--system-prompt") + 1] == "SYSTEM"
    assert captured["argv"][captured["argv"].index("--model") + 1] == "sonnet"
    assert captured["kwargs"]["input"] == "USER"
    # Without a new session the shell's SIGTERM reaches python but orphans claude.
    assert captured["kwargs"]["start_new_session"] is True
    assert out == "## Темы\n- x"


def test_claude_backend_reports_a_nonzero_exit_with_its_stderr(monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="OAuth session expired")

    monkeypatch.setattr(summary_backends.subprocess, "run", fake_run)
    backend = summary_backends.default("claude", model="sonnet")
    with pytest.raises(summarize.SummaryError, match="OAuth session expired"):
        backend.generate("SYSTEM", "USER", 512)


def test_claude_backend_translates_a_subprocess_timeout(monkeypatch):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(summary_backends.subprocess, "run", fake_run)
    backend = summary_backends.default("claude", model="sonnet", timeout=5)
    with pytest.raises(summarize.SummaryTimeout):
        backend.generate("SYSTEM", "USER", 512)


# --- API drift guards -----------------------------------------------------

def test_installed_mlx_lm_still_accepts_the_keywords_we_pass():
    params = mlx_lm_source.generate_step_parameters()
    assert params, "guard the guard: signature parsing found nothing"
    for keyword in ("sampler", "logits_processors", "kv_bits", "kv_group_size",
                    "quantized_kv_start"):
        assert keyword in params, f"mlx-lm no longer accepts {keyword}"


def test_we_never_pass_max_kv_size():
    # max_kv_size selects a RotatingKVCache, whose _trim drops the OLDEST
    # tokens. On a summarizer that silently deletes the start of the meeting
    # and still exits 0. Prose mentioning it is fine; passing it is not.
    for name, source in mlx_lm_source.whisp_sources().items():
        assert "max_kv_size=" not in source, f"{name} must not pass max_kv_size"


def test_mlx_still_exposes_the_memory_helpers_we_call():
    import mlx.core as mx

    for attr in ("clear_cache", "get_peak_memory", "reset_peak_memory"):
        assert hasattr(mx, attr), f"mx.{attr} moved; check the namespace migration"
