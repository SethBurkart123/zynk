"""Python typing helpers that lower annotations into Zynk IR."""

from __future__ import annotations

import types
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from .ir import Field, ModelDef, TypeRef
from .naming import python_name_to_camel_case

PYTHON_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    bytes: "string",
    type(None): "undefined",
}


def is_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def strip_optional(type_hint: Any) -> tuple[Any, bool]:
    origin = get_origin(type_hint)
    if not is_union(origin):
        return type_hint, False
    args = get_args(type_hint)
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0], True
    return type_hint, False


def type_ref(type_hint: Any) -> TypeRef:
    if type_hint is None:
        return TypeRef(kind="void", py_type=None)
    if type_hint is Any:
        return TypeRef(kind="any", py_type=type_hint)
    if type_hint in PYTHON_PRIMITIVES:
        return TypeRef(kind="primitive", name=PYTHON_PRIMITIVES[type_hint], py_type=type_hint)

    actual, optional = strip_optional(type_hint)
    if optional:
        ref = type_ref(actual)
        ref.optional = True
        ref.py_type = type_hint
        return ref

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Literal:
        if len(args) == 1:
            return TypeRef(kind="literal", value=args[0], py_type=type_hint)
        return TypeRef(
            kind="union",
            inner=[TypeRef(kind="literal", value=arg, py_type=type(arg)) for arg in args],
            py_type=type_hint,
        )
    if is_union(origin):
        return TypeRef(
            kind="union",
            inner=[type_ref(arg) for arg in args],
            py_type=type_hint,
        )
    if origin is list:
        item = type_ref(args[0]) if args else TypeRef(kind="any")
        return TypeRef(kind="array", inner=[item], py_type=type_hint)
    if origin is dict:
        key = type_ref(args[0]) if len(args) >= 1 else TypeRef(kind="primitive", name="string")
        value = type_ref(args[1]) if len(args) >= 2 else TypeRef(kind="any")
        return TypeRef(kind="record", inner=[key, value], py_type=type_hint)
    if origin is tuple:
        return TypeRef(kind="tuple", inner=[type_ref(arg) for arg in args], py_type=type_hint)
    if origin is set:
        item = type_ref(args[0]) if args else TypeRef(kind="any")
        return TypeRef(kind="array", inner=[item], py_type=type_hint)

    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return TypeRef(kind="model", name=type_hint.__name__, py_type=type_hint)
    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return TypeRef(kind="enum", name=type_hint.__name__, py_type=type_hint)
    if isinstance(type_hint, type):
        return TypeRef(kind="any", name=type_hint.__name__, py_type=type_hint)

    return TypeRef(kind="any", py_type=type_hint)


def model_def(model: type[BaseModel]) -> ModelDef:
    fields: list[Field] = []
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        _, optional = strip_optional(annotation)
        required = field_info.is_required() and not optional
        fields.append(
            Field(
                py_name=field_name,
                ts_name=python_name_to_camel_case(field_name),
                type=type_ref(annotation),
                required=required,
                doc=field_info.description,
                default=None if field_info.is_required() else field_info.default,
            )
        )
    return ModelDef(name=model.__name__, py_type=model, fields=fields, doc=model.__doc__)
