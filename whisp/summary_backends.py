"""Summarization backends.

Implementations for the Backend contract declared in whisp.summarize, the way
whisp.stages holds the implementations for whisp.pipeline's Stages. mlx-lm is
imported inside the loader rather than at module scope so this module -- and
the tests -- import cleanly on a machine without it.
"""

import subprocess

from whisp.summarize import Backend, SummaryError, SummaryTimeout

# Sampling comes from the model's own generation_config.json rather than being
# a knob. Greedy decoding (temp 0) is explicitly discouraged for Qwen3: it
# falls into repetition loops. Reproducibility comes from the fixed seed below,
# not from temperature.
_TEMPERATURE = 0.7
_TOP_P = 0.8
_TOP_K = 20
_REPETITION_PENALTY = 1.05
_SEED = 0

# 8-bit KV quantization roughly halves the cache, which is what keeps a
# 23k-token transcript inside the 11.84 GiB working set measured on this
# M1 Pro / 16 GB. Deliberately NOT max_kv_size: that selects a RotatingKVCache
# whose _trim drops the oldest tokens, silently summarizing a meeting with its
# beginning missing.
_KV_BITS = 8
_KV_GROUP_SIZE = 64
_QUANTIZED_KV_START = 1024

_MISSING_MLX = (
    "whisp: mlx-lm is not installed; local summarization needs it on Apple Silicon.\n"
    '       Install it with: ./.venv/bin/pip install "mlx-lm>=0.29,<0.30"\n'
    "       or select the hosted backend with WHISP_SUMMARY_BACKEND=claude"
)


def _load_mlx(model_id: str):
    """Load weights and tokenizer. Separate so tests can substitute it."""
    from mlx_lm import load

    return load(model_id)


def _stream_text(model, tokenizer, prompt, max_tokens: int) -> str:
    """Run one generation to completion and return its text."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    mx.random.seed(_SEED)
    sampler = make_sampler(temp=_TEMPERATURE, top_p=_TOP_P, top_k=_TOP_K)
    processors = make_logits_processors(repetition_penalty=_REPETITION_PENALTY)

    pieces = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
        kv_bits=_KV_BITS,
        kv_group_size=_KV_GROUP_SIZE,
        quantized_kv_start=_QUANTIZED_KV_START,
    ):
        pieces.append(response.text)
    return "".join(pieces)


def _mlx_backend(model_id: str) -> Backend:
    state: dict = {}

    def ensure():
        """Load on first use, once. The model stays resident for the whole run:
        reloading between chunks would cost seconds per chunk for nothing."""
        if not state:
            try:
                model, tokenizer = _load_mlx(model_id)
            except ImportError as exc:
                raise SummaryError(_MISSING_MLX) from exc
            state["model"], state["tokenizer"] = model, tokenizer
        return state["model"], state["tokenizer"]

    def count_tokens(text: str) -> int:
        _, tokenizer = ensure()
        return len(tokenizer.encode(text))

    def generate(system: str, user: str, max_tokens: int) -> str:
        model, tokenizer = ensure()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Qwen3's template guards on `enable_thinking is defined and is false`,
        # so leaving it out leaves thinking ON. Models without the switch (the
        # Instruct-2507 line, Gemma) must not receive it.
        template_kwargs = {}
        if "enable_thinking" in (tokenizer.chat_template or ""):
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **template_kwargs
        )
        text = _stream_text(model, tokenizer, prompt, max_tokens)
        _release_cache()
        return text

    return Backend(generate=generate, count_tokens=count_tokens, name="mlx")


def _release_cache() -> None:
    """Return the prompt cache's buffers to the system between generations.

    MLX retains freed blocks in its own cache, so without this the peak grows
    across chunks on a machine that is already swapping.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # pragma: no cover - never fail a summary over cleanup
        pass


def _claude_backend(model: str, timeout: float | None = None) -> Backend:
    def generate(system: str, user: str, max_tokens: int) -> str:
        argv = ["claude", "-p", "--system-prompt", system, "--model", model]
        try:
            proc = subprocess.run(
                argv,
                input=user,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Own session so the whole process group can be signalled;
                # otherwise a SIGTERM aimed at python orphans claude.
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise SummaryError(
                "whisp: the `claude` CLI was not found on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SummaryTimeout(f"claude did not finish within {timeout}s") from exc
        if proc.returncode != 0:
            raise SummaryError(
                f"claude exited {proc.returncode}: {proc.stderr.strip() or '(no stderr)'}"
            )
        return proc.stdout

    def count_tokens(text: str) -> int:
        # Only used to decide whether chunking is needed. Claude's context is
        # far larger than any transcript here, so a coarse estimate is enough.
        return len(text) // 3

    return Backend(generate=generate, count_tokens=count_tokens, name="claude")


def default(name: str, model: str, timeout: float | None = None) -> Backend:
    """Build a backend by name."""
    if name == "mlx":
        return _mlx_backend(model)
    if name == "claude":
        return _claude_backend(model, timeout)
    raise ValueError(f"unknown summary backend {name!r}; valid names are 'mlx' and 'claude'")
