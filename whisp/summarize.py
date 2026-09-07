"""Transcript summarization.

Runs as its own process (`python -m whisp.summarize`), started by whisp.sh
after the transcription driver has exited. The process boundary is deliberate:
torch-MPS and MLX each run an independent caching allocator over the same
unified memory with no cross-pressure signalling, so a torch cache holding
freed-but-retained blocks while MLX allocates a KV cache is an OOM or a swap
storm that neither allocator can be told to back off from. Running separately
also means a summary can be regenerated from an existing transcript without
re-transcribing.

Nothing in this module's import graph may pull in torch or whisperx; that is
what keeps the two allocators apart, and it is covered by a test.
"""

import argparse
import dataclasses
import os
import pathlib
import re
import sys
import time
from typing import Callable

from whisp import prompts
from whisp.timing import StageLog

# Chosen by measurement on this machine; see
# docs/superpowers/specs/2026-09-05-local-summarization-design.md.
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_CLAUDE_MODEL = "sonnet"
DEFAULT_CHUNK_TOKENS = 6000
# Four sections; 2048 truncated a real 25k-token transcript mid-way through
# the last one. Combined with the 15-bullet cap in the prompt this leaves room.
DEFAULT_MAX_TOKENS = 3072
# Ceiling for a single chunk's intermediate notes; see summarize().
MAP_TOKEN_CAP = 768
# Raised from the 900 that sized a ~60s network call. Model load is timed
# separately and deliberately excluded from this budget.
DEFAULT_TIMEOUT = 1800.0


def chunk_lines(
    text: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """Split a transcript into chunks of at most max_tokens, on line boundaries.

    Lines are the unit because a transcript line is one utterance
    (`[SPEAKER_00]: ...`) -- and because when diarization or alignment fails
    the transcript has no speaker prefixes at all, so splitting on those would
    return one giant chunk for exactly the files that need chunking most.

    Each line is measured once and the total accumulated; measuring the
    growing chunk instead would be quadratic, and the largest real transcript
    is 4308 lines.

    A single line longer than the budget is emitted as its own oversized chunk
    rather than dropped or split mid-utterance.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in text.splitlines(keepends=True):
        line_tokens = count_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append("".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return chunks


class SummaryError(RuntimeError):
    """The backend ran but produced nothing usable."""


class SummaryTimeout(SummaryError):
    """The deadline passed before the summary was finished."""


@dataclasses.dataclass
class Backend:
    """A text generator plus its tokenizer.

    Injected the way whisp.pipeline injects Stages: every decision in this
    module -- chunking, strategy, sanitizing, the deadline -- is then testable
    without loading a model.
    """

    generate: Callable[[str, str, int], str]
    count_tokens: Callable[[str], int]
    name: str


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n?```$", re.MULTILINE)


def sanitize(raw: str) -> str:
    """Strip the wrappers small models add around the requested output.

    enable_thinking=False already keeps <think> out of the generated text on
    the models that support it, but the guard is one line and covers the ones
    that ignore it.
    """
    text = _THINK.sub("", raw).strip()
    text = _FENCE.sub("", text).strip()
    first_section = text.find("## ")
    if first_section > 0:
        text = text[first_section:]
    return text.strip()


def _check_deadline(started, deadline, monotonic, what):
    if deadline is not None and monotonic() - started > deadline:
        raise SummaryTimeout(f"summary deadline of {deadline:.0f}s passed during {what}")


def summarize(
    text: str,
    backend: Backend,
    language: str,
    chunk_tokens: int,
    max_tokens: int,
    log=None,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    prompt_dir=None,
) -> str:
    """Summarize a transcript, single-pass or map-reduce depending on length.

    One knob decides the strategy: if the transcript fits a single chunk it is
    summarized in one call, so short recordings never pay for a reduce pass.
    """
    chunks = chunk_lines(text, chunk_tokens, backend.count_tokens)
    if not chunks:
        raise SummaryError("transcript is empty")

    started = monotonic()

    if len(chunks) == 1:
        started_call = monotonic()
        raw = backend.generate(prompts.select(language, "single", prompt_dir), chunks[0], max_tokens)
        if log is not None:
            log.record("summary-single", monotonic() - started_call)
    else:
        # The map phase runs once per chunk, so its output budget -- not the
        # reduce pass, and not prefill -- dominates wall time on a long
        # transcript. The notes are intermediate and feed a second pass, so
        # they are capped well below the final summary's budget.
        map_max_tokens = min(max_tokens, MAP_TOKEN_CAP)
        map_prompt = prompts.select(language, "map", prompt_dir)
        notes = []
        for index, chunk in enumerate(chunks, 1):
            _check_deadline(started, deadline, monotonic, f"chunk {index}/{len(chunks)}")
            started_call = monotonic()
            notes.append(backend.generate(map_prompt, chunk, map_max_tokens).strip())
            if log is not None:
                log.record(f"summary-map {index}/{len(chunks)}", monotonic() - started_call)

        _check_deadline(started, deadline, monotonic, "reduce")
        joined = "\n\n".join(
            f"[Фрагмент {i}]\n{note}" for i, note in enumerate(notes, 1)
        )
        started_call = monotonic()
        raw = backend.generate(prompts.select(language, "reduce", prompt_dir), joined, max_tokens)
        if log is not None:
            log.record("summary-reduce", monotonic() - started_call)

    summary = sanitize(raw)
    if not summary:
        raise SummaryError("the model returned an empty summary")
    if "## " not in summary:
        raise SummaryError(
            f"the model returned no sections; first 200 chars: {summary[:200]!r}"
        )
    if prompt_dir is None:
        # Only meaningful for our own templates; a custom prompt may ask for
        # any shape at all.
        missing = [h for h in prompts.sections(language) if h not in summary]
        if missing:
            raise SummaryError(
                "the summary stopped before it was finished -- missing "
                + ", ".join(h.removeprefix("## ") for h in missing)
                + ". Raise WHISP_SUMMARY_MAX_TOKENS and try again."
            )
    return summary



def parse_args(argv: list[str], env: dict | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env

    parser = argparse.ArgumentParser(prog="whisp.summarize", description=__doc__)
    parser.add_argument("transcript", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--backend", default=env.get("WHISP_SUMMARY_BACKEND", "mlx"))
    parser.add_argument("--model", default=env.get("WHISP_SUMMARY_MODEL"))
    parser.add_argument(
        "--language",
        default=env.get("WHISP_SUMMARY_LANG") or env.get("WHISP_LANG", "ru"),
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=int(env.get("WHISP_SUMMARY_CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS)),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(env.get("WHISP_SUMMARY_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    )
    parser.add_argument(
        "--print-prompt",
        metavar="KIND",
        help=f"print the built-in prompt for one of {', '.join(prompts.KINDS)} and exit",
    )
    parser.add_argument(
        "--prompt-dir",
        default=env.get("WHISP_SUMMARY_PROMPT") or None,
        help="directory with single.txt / map.txt / reduce.txt overriding our prompts",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(env.get("WHISP_SUMMARY_TIMEOUT", DEFAULT_TIMEOUT)),
    )

    args = parser.parse_args(argv)
    if args.prompt_dir is not None:
        args.prompt_dir = pathlib.Path(args.prompt_dir)
    if args.model is None:
        args.model = (
            DEFAULT_CLAUDE_MODEL if args.backend == "claude" else DEFAULT_MLX_MODEL
        )
    return args


def run(args: argparse.Namespace, backend: Backend | None = None) -> int:
    """Summarize one transcript. Returns a process exit code.

    Never raises for an expected failure: the transcript is the deliverable
    and a missing summary must not take the run down with it.
    """
    if args.print_prompt:
        try:
            print(prompts.select(args.language, args.print_prompt, args.prompt_dir))
        except KeyError:
            print(
                f"whisp: unknown prompt kind {args.print_prompt!r}; "
                f"expected one of {', '.join(prompts.KINDS)}",
                file=sys.stderr,
            )
            return 1
        return 0

    if not args.transcript or not args.output:
        print("whisp: a transcript and --output are required", file=sys.stderr)
        return 1

    transcript_path = pathlib.Path(args.transcript)
    try:
        text = transcript_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"whisp: cannot read transcript {transcript_path}: {exc}", file=sys.stderr)
        return 1

    log = StageLog(audio_seconds=0)

    try:
        if backend is None:
            from whisp import summary_backends

            backend = summary_backends.default(
                args.backend, model=args.model, timeout=args.timeout
            )
        print(f"whisp: summary backend {backend.name} ({args.model})", flush=True)

        # Loading (and, on a first run, downloading ~2.1 GiB of weights) is
        # timed on its own and kept out of the generation deadline.
        started_load = time.monotonic()
        backend.count_tokens("")
        log.record("summary-load", time.monotonic() - started_load)

        started = time.monotonic()
        summary = summarize(
            text,
            backend,
            language=args.language,
            chunk_tokens=args.chunk_tokens,
            max_tokens=args.max_tokens,
            log=log,
            deadline=args.timeout,
            prompt_dir=args.prompt_dir,
        )
        log.total(time.monotonic() - started)
    except SummaryTimeout as exc:
        print(f"whisp: {exc}", file=sys.stderr)
        return 2
    except (SummaryError, ValueError) as exc:
        print(f"whisp: {exc}", file=sys.stderr)
        return 1

    pathlib.Path(args.output).write_text(summary + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
