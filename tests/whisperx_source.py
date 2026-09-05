"""Shared helpers for reading defaults out of the installed whisperx package.

Comparing against a hardcoded literal only catches edits to our own
constants. The real risk this guards against is a whisperx upgrade changing
one of its own defaults out from under us: the transcript would silently
stop being byte-identical and nothing would say so. These helpers parse the
installed whisperx source with `ast` instead of writing the expected values
down here, so a whisperx upgrade that changes them fails the tests that
compare against these helpers.
"""

import ast
import pathlib

import whisperx


def _source(filename: str) -> str:
    return pathlib.Path(whisperx.__file__).with_name(filename).read_text()


def whisperx_cli_defaults() -> dict:
    """argparse defaults declared by the installed whisperx CLI.

    Parsed statically: whisperx builds its parser inside cli(), which also
    runs the whole pipeline, so it cannot be imported and inspected.
    """
    source = _source("__main__.py")
    defaults: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        flag = next(
            (
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ),
            None,
        )
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag[2:]] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass  # e.g. --device, whose default is a torch call
    return defaults


def whisperx_faster_whisper_threads_default() -> int | None:
    """The `faster_whisper_threads = 4` fallback in whisperx/transcribe.py.

    This is the CTranslate2 thread count whisperx uses when `--threads 0`
    (the default) is passed -- 0 means "use the plain assignment's default
    of 4", not "use every core". It is a local variable inside cli(), not an
    argparse default, so it needs its own tiny parse rather than reusing
    whisperx_cli_defaults(): find the assignment to `faster_whisper_threads`
    whose right-hand side is itself a literal (the later conditional
    reassignment `faster_whisper_threads = threads` is a Name, not a
    literal, so it is skipped automatically).
    """
    source = _source("transcribe.py")
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "faster_whisper_threads"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    return None
