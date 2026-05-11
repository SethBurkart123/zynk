"""Language-agnostic code generation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from zynk.core.ir import ApiGraph


@dataclass(slots=True)
class GenerationContext:
    """Inputs available to every language generator."""

    graph: ApiGraph
    registry: Any
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedFile:
    """One file emitted by a code generator."""

    path: Path
    content: str


@dataclass(slots=True)
class GenerationResult:
    """All files emitted by a generator."""

    files: list[GeneratedFile]

    def write(self) -> None:
        for file in self.files:
            file.path.parent.mkdir(parents=True, exist_ok=True)
            file.path.write_text(file.content)


class CodeGenerator(Protocol):
    """Protocol implemented by language backends."""

    language: str

    def generate(self, output_path: Path, context: GenerationContext) -> GenerationResult:
        """Return generated files without writing them."""
        ...
