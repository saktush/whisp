"""Helpers for reading the API surface out of the installed mlx-lm.

Same reasoning as tests/whisperx_source.py: comparing against a hardcoded
literal only catches edits to our own code. The risk this guards against is an
mlx-lm upgrade moving a keyword we depend on, or MLX's namespace migration
(mx.metal.clear_cache -> mx.clear_cache) landing under us.
"""

import inspect
import pathlib


def generate_step_parameters() -> list[str]:
    from mlx_lm.generate import generate_step

    return list(inspect.signature(generate_step).parameters)


def whisp_sources() -> dict[str, str]:
    root = pathlib.Path(__file__).resolve().parent.parent / "whisp"
    return {p.name: p.read_text(encoding="utf-8") for p in root.glob("*.py")}
