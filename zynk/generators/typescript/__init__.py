"""Built-in TypeScript client generator plugin.

Importing this module registers the TypeScript backend. Future language plugins
should mirror this package: keep all language-specific lowering/printing/runtime
logic here and call ``register_generator(...)`` at import time.
"""

from __future__ import annotations

from pathlib import Path

from zynk.codegen import (
    GeneratedFile,
    GenerationContext,
    GenerationResult,
    register_generator,
)

from .legacy import TypeScriptGenerator
from .lower import TypeScriptLowerer
from .printer import TsPrinter


class TypeScriptClientGenerator:
    language = "typescript"

    def __init__(self, legacy_generator: TypeScriptGenerator | None = None):
        self.legacy_generator = legacy_generator or TypeScriptGenerator()

    def generate(self, output_path: Path, context: GenerationContext) -> GenerationResult:
        module = TypeScriptLowerer(self.legacy_generator).lower(
            context.graph,
            context.registry,
        )
        return GenerationResult(
            files=[
                GeneratedFile(
                    path=output_path.parent / "_internal.ts",
                    content=self.legacy_generator._generate_internal_module(),
                ),
                GeneratedFile(
                    path=output_path,
                    content=TsPrinter().print(module),
                ),
            ]
        )


register_generator(TypeScriptClientGenerator(), replace=True)

__all__ = ["TypeScriptClientGenerator"]
