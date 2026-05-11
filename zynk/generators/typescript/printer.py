"""Printer for the tiny TypeScript AST."""

from __future__ import annotations

from .ts_ast import TsExport, TsImport, TsModule, TsNode, TsRaw


class TsPrinter:
    def print(self, node: TsNode) -> str:
        if isinstance(node, TsModule):
            return self._module(node)
        if isinstance(node, TsImport):
            prefix = "import type" if node.type_only else "import"
            return f'{prefix} {{ {", ".join(node.names)} }} from "{node.module}";'
        if isinstance(node, TsExport):
            prefix = "export type" if node.type_only else "export"
            return f'{prefix} {{ {", ".join(node.names)} }};'
        if isinstance(node, TsRaw):
            return node.code.rstrip()
        raise TypeError(f"Unsupported TS node: {type(node)!r}")

    def _module(self, module: TsModule) -> str:
        parts: list[str] = []
        if module.banner:
            parts.append(module.banner.rstrip())
        for node in module.body:
            text = self.print(node).rstrip()
            if text:
                parts.append(text)
        return "\n\n".join(parts).rstrip() + "\n"
