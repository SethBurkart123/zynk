"""Small semantic IR for the registered Zynk API graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

EndpointKind = Literal["rpc", "channel", "upload", "static", "ws"]
TypeKind = Literal[
    "primitive",
    "array",
    "record",
    "tuple",
    "union",
    "literal",
    "model",
    "enum",
    "any",
    "void",
]

MISSING = object()


@dataclass(slots=True)
class TypeRef:
    kind: TypeKind
    name: str | None = None
    inner: list[TypeRef] = field(default_factory=list)
    py_type: Any = None
    optional: bool = False
    value: Any = None


@dataclass(slots=True)
class Param:
    py_name: str
    ts_name: str
    type: TypeRef
    required: bool
    default: Any = MISSING


@dataclass(slots=True)
class Field:
    py_name: str
    ts_name: str
    type: TypeRef
    required: bool
    doc: str | None = None
    default: Any = MISSING


@dataclass(slots=True)
class ModelDef:
    name: str
    py_type: type
    fields: list[Field]
    doc: str | None = None


@dataclass(slots=True)
class EnumDef:
    name: str
    py_type: type
    values: list[Any]
    doc: str | None = None


@dataclass(slots=True)
class Endpoint:
    name: str
    kind: EndpointKind
    func: Callable[..., Any]
    is_async: bool
    module: str
    doc: str | None
    params: list[Param]
    returns: TypeRef
    channel_item: TypeRef | None = None
    file_param: str | None = None
    multi_file: bool = False
    max_size: int | None = None
    allowed_types: list[str] = field(default_factory=list)
    server_events: dict[str, TypeRef] = field(default_factory=dict)
    client_events: dict[str, TypeRef] = field(default_factory=dict)
    deps: list[Callable[..., Any]] = field(default_factory=list)
    raw: Any = None


@dataclass(slots=True)
class ApiGraph:
    endpoints: dict[str, Endpoint] = field(default_factory=dict)
    models: dict[str, ModelDef] = field(default_factory=dict)
    enums: dict[str, EnumDef] = field(default_factory=dict)
