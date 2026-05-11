"""Map Zynk IR ``TypeRef`` values to TypeScript types and Effect Schema expressions.

The Effect connector emits two parallel pieces for every Pydantic model:

1. A static ``Schema.Struct`` value that validates the server's JSON payload at
   runtime.
2. A derived TypeScript type alias (``Schema.Schema.Type<typeof MyModel>``) that
   is used in the generated function signatures.

Keeping the two derived from the same schema means there is one source of truth
for every API boundary in the generated client.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from zynk.core.ir import TypeRef
from zynk.core.naming import python_name_to_camel_case

_PRIMITIVE_TS: dict[str, str] = {
    "string": "string",
    "number": "number",
    "boolean": "boolean",
    "undefined": "undefined",
}

_PRIMITIVE_SCHEMA: dict[str, str] = {
    "string": "Schema.String",
    "number": "Schema.Number",
    "boolean": "Schema.Boolean",
    "undefined": "Schema.Undefined",
}


@dataclass(slots=True)
class TypeExpr:
    """A pair of strings describing the TypeScript type and matching schema."""

    ts: str
    schema: str


def type_expr(ref: TypeRef | None, model_names: set[str]) -> TypeExpr:
    """Convert a ``TypeRef`` to a TypeScript + Effect Schema expression pair.

    Pydantic model names referenced from this type are accumulated in
    ``model_names`` so the generator can decide which schemas need to be
    declared in the output module.
    """
    if ref is None:
        return TypeExpr("void", "Schema.Void")

    kind = ref.kind

    if kind == "void":
        return TypeExpr("void", "Schema.Void")
    if kind == "any":
        return TypeExpr("unknown", "Schema.Unknown")
    if kind == "primitive":
        ts = _PRIMITIVE_TS.get(ref.name or "unknown", "unknown")
        schema = _PRIMITIVE_SCHEMA.get(ref.name or "unknown", "Schema.Unknown")
        return _maybe_optional(ref, TypeExpr(ts, schema))
    if kind == "literal":
        value = ref.value
        if isinstance(value, str):
            return TypeExpr(f'"{value}"', f'Schema.Literal("{value}")')
        return TypeExpr(repr(value), f"Schema.Literal({_literal_js(value)})")
    if kind == "array":
        inner = type_expr(ref.inner[0] if ref.inner else None, model_names)
        return _maybe_optional(
            ref,
            TypeExpr(f"ReadonlyArray<{inner.ts}>", f"Schema.Array({inner.schema})"),
        )
    if kind == "record":
        key = type_expr(ref.inner[0] if len(ref.inner) >= 1 else None, model_names)
        value = type_expr(ref.inner[1] if len(ref.inner) >= 2 else None, model_names)
        ts_key = key.ts if key.ts in ("string", "number") else "string"
        ts = f"Readonly<Record<{ts_key}, {value.ts}>>"
        schema = f"Schema.Record({{ key: {key.schema}, value: {value.schema} }})"
        return _maybe_optional(ref, TypeExpr(ts, schema))
    if kind == "tuple":
        parts = [type_expr(inner, model_names) for inner in ref.inner]
        ts = "readonly [" + ", ".join(part.ts for part in parts) + "]"
        schema = "Schema.Tuple(" + ", ".join(part.schema for part in parts) + ")"
        return _maybe_optional(ref, TypeExpr(ts, schema))
    if kind == "union":
        parts = [type_expr(inner, model_names) for inner in ref.inner]
        ts = " | ".join(part.ts for part in parts) or "unknown"
        schema = (
            "Schema.Union(" + ", ".join(part.schema for part in parts) + ")"
            if parts
            else "Schema.Unknown"
        )
        return _maybe_optional(ref, TypeExpr(ts, schema))
    if kind == "model":
        if ref.name:
            model_names.add(ref.name)
        ts = ref.name or "unknown"
        schema = ref.name or "Schema.Unknown"
        return _maybe_optional(ref, TypeExpr(ts, schema))
    if kind == "enum":
        return _maybe_optional(ref, _enum_expr(ref))

    return TypeExpr("unknown", "Schema.Unknown")


def _maybe_optional(ref: TypeRef, expr: TypeExpr) -> TypeExpr:
    if not ref.optional:
        return expr
    return TypeExpr(
        f"{expr.ts} | undefined",
        f"Schema.UndefinedOr({expr.schema})",
    )


def _enum_expr(ref: TypeRef) -> TypeExpr:
    py_type = ref.py_type
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        members = list(py_type)
        if all(isinstance(m.value, str) for m in members):
            literals = [f'"{m.value}"' for m in members]
            ts = " | ".join(literals) if literals else "string"
            schema = (
                "Schema.Literal(" + ", ".join(literals) + ")"
                if literals
                else "Schema.String"
            )
            return TypeExpr(ts, schema)
    return TypeExpr("string", "Schema.String")


def _literal_js(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return repr(value)


# ---------------------------------------------------------------------------
# Model rendering
# ---------------------------------------------------------------------------


def render_model_schema(
    model: type[BaseModel],
    model_names: set[str],
) -> tuple[str, str]:
    """Return ``(schema_decl, type_alias)`` strings for a Pydantic model.

    The schema represents the server's snake_case payload; the camelCase
    response shape used by the generated client functions is constructed at the
    call site once decoded.
    """
    from zynk.core.types import strip_optional, type_ref

    field_lines: list[str] = []
    for field_name, field_info in model.model_fields.items():
        ts_name = python_name_to_camel_case(field_name)
        annotation = field_info.annotation
        _, optional = strip_optional(annotation)
        is_optional = optional or (
            not field_info.is_required() and field_info.default is not None
        )
        ref = type_ref(annotation)
        if is_optional:
            ref.optional = True
        expr = type_expr(ref, model_names)
        field_lines.append(f"  {ts_name}: {expr.schema}")

    schema_body = ",\n".join(field_lines)
    schema_decl = (
        f"export const {model.__name__} = Schema.Struct({{\n{schema_body}\n}})"
        if field_lines
        else f"export const {model.__name__} = Schema.Struct({{}})"
    )
    type_alias = (
        f"export type {model.__name__} = "
        f"Schema.Schema.Type<typeof {model.__name__}>"
    )
    return schema_decl, type_alias


def collect_model_refs(refs: Iterable[TypeRef | None]) -> set[str]:
    """Collect every model name reachable from the given type refs."""
    seen: set[str] = set()
    for ref in refs:
        if ref is None:
            continue
        _walk(ref, seen)
    return seen


def _walk(ref: TypeRef, out: set[str]) -> None:
    if ref.kind == "model" and ref.name:
        out.add(ref.name)
    for inner in ref.inner:
        _walk(inner, out)
