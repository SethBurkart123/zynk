"""Runtime coercion and serialization helpers."""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def _camel_to_snake(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def instantiate_model(data: Any, type_hint: Any) -> Any:
    if data is None:
        return None

    origin = get_origin(type_hint) if type_hint else None
    args = get_args(type_hint) if type_hint else ()

    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return instantiate_model(data, non_none_args[0])

    if isinstance(data, list) and origin is list and args:
        item_type = args[0]
        return [instantiate_model(item, item_type) for item in data]

    if isinstance(data, dict):
        if type_hint and isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            converted = {}
            fields_by_wire_name = {
                _camel_to_snake(field_name): (field_name, field_info)
                for field_name, field_info in type_hint.model_fields.items()
            }
            for key, value in data.items():
                field = type_hint.model_fields.get(key)
                field_name = key
                if field is None:
                    wire_field = fields_by_wire_name.get(key)
                    if wire_field is not None:
                        field_name, field = wire_field
                converted[field_name] = instantiate_model(value, field.annotation) if field else value
            return type_hint(**converted)

    return data


def coerce_query_value(value: Any, type_hint: Any) -> Any:
    if type_hint is int:
        return int(value)
    if type_hint is float:
        return float(value)
    if type_hint is bool:
        return str(value).lower() in ("true", "1", "yes")
    return value


def serialize_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, list):
        return [item.model_dump() if isinstance(item, BaseModel) else item for item in result]
    return result
