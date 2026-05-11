"""Language-agnostic code generation API.

This package intentionally contains no language backends. Generator plugins live
outside core and register themselves when imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zynk.registry import get_registry

from .base import CodeGenerator, GeneratedFile, GenerationContext, GenerationResult
from .registry import get_generator, list_generators, register_generator


def generate_client(
    output_path: str,
    *,
    language: str,
    options: Mapping[str, Any] | None = None,
) -> GenerationResult:
    registry = get_registry()
    generator = get_generator(language)
    result = generator.generate(
        Path(output_path),
        GenerationContext(
            graph=registry.get_api_graph(),
            registry=registry,
            options=options or {},
        ),
    )
    result.write()
    return result


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
