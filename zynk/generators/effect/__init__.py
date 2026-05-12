"""Effect connector for Zynk.

Importing this module registers the ``effect`` language with Zynk's code
generation registry. The output is a TypeScript client that returns ``Effect``
values, validates payloads with ``effect/Schema``, supports retry/backoff
strategies configured globally or per-call, and exposes channels as Streams.

Usage:

    >>> from zynk import generate_client
    >>> import zynk.generators.effect  # registers the backend
    >>> generate_client("./frontend/src/api.ts", language="effect")
"""

from __future__ import annotations

from pathlib import Path

from zynk.codegen import (
    GeneratedFile,
    GenerationContext,
    GenerationResult,
    register_generator,
)

from .lower import lower_graph
from .options import EffectGeneratorOptions
from .runtime import INTERNAL_RUNTIME


class EffectClientGenerator:
    language = "effect"

    def generate(
        self,
        output_path: Path,
        context: GenerationContext,
    ) -> GenerationResult:
        options = EffectGeneratorOptions.from_mapping(context.options) if context.options else None
        module = lower_graph(context.graph, context.registry, options)
        runtime_path = output_path.parent / "_effect_internal.ts"
        return GenerationResult(
            files=[
                GeneratedFile(path=runtime_path, content=INTERNAL_RUNTIME),
                GeneratedFile(path=output_path, content=module.render()),
            ]
        )


register_generator(EffectClientGenerator(), replace=True)


def generate_effect(output_path: str) -> None:
    """Convenience helper mirroring ``generate_typescript`` for the Effect backend."""
    from zynk.codegen import generate_client

    generate_client(output_path, language="effect")


__all__ = ["EffectClientGenerator", "generate_effect"]
