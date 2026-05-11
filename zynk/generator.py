"""Backward-compatible TypeScript generation entry point.

The TypeScript implementation lives in ``zynk.generators.typescript``. This module
exists so existing ``from zynk.generator import generate_typescript`` imports keep
working without putting plugin logic in core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zynk.generators.typescript.legacy import TypeScriptGenerator


def __getattr__(name: str):
    if name == "TypeScriptGenerator":
        from zynk.generators.typescript.legacy import TypeScriptGenerator

        return TypeScriptGenerator
    raise AttributeError(name)


def generate_typescript(output_path: str) -> None:
    import zynk.generators.typescript  # noqa: F401

    from .codegen import generate_client

    generate_client(output_path, language="typescript")


__all__ = ["TypeScriptGenerator", "generate_typescript"]
