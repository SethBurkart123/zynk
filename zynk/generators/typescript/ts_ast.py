"""Tiny TypeScript AST used as the final code generation target."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TsNode:
    pass


@dataclass(slots=True)
class TsRaw(TsNode):
    code: str


@dataclass(slots=True)
class TsImport(TsNode):
    names: list[str]
    module: str
    type_only: bool = False


@dataclass(slots=True)
class TsExport(TsNode):
    names: list[str]
    type_only: bool = False


@dataclass(slots=True)
class TsModule(TsNode):
    banner: str | None = None
    body: list[TsNode] = field(default_factory=list)

    def add(self, node: TsNode | str) -> None:
        self.body.append(TsRaw(node) if isinstance(node, str) else node)
