"""Built-in and user-importable client generator backends.

A generator module registers itself on import. For example, TypeScript is registered
by importing ``zynk.generators.typescript``. A future Rust backend can follow the
same shape in ``zynk/generators/rust.py`` or in a third-party package.
"""

from zynk.codegen.base import (
    CodeGenerator,
    GeneratedFile,
    GenerationContext,
    GenerationResult,
)
from zynk.codegen.registry import get_generator, list_generators, register_generator


def generate_client(*args, **kwargs):
    from zynk.codegen import generate_client as _generate_client

    return _generate_client(*args, **kwargs)


__all__ = [
    "CodeGenerator",
    "GeneratedFile",
    "GenerationContext",
    "GenerationResult",
    "generate_client",
    "get_generator",
    "list_generators",
    "register_generator",
]
