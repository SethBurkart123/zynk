"""Canonical JSON serialization for the Zynk API graph."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .ir import MISSING, ApiGraph, Endpoint, EnumDef, Field, ModelDef, Param, TypeRef
from .naming import python_name_to_camel_case

JsonObject = dict[str, Any]


def dump_api_graph_json(graph: ApiGraph) -> str:
    """Serialize a Python API graph into the Rust ``zynk-schema`` JSON shape."""
    canonical = {
        "endpoints": {
            name: _endpoint_to_schema(endpoint)
            for name, endpoint in sorted(graph.endpoints.items())
        },
        "enums": {
            name: _enum_to_schema(enum_def)
            for name, enum_def in sorted(graph.enums.items())
        },
        "models": {
            name: _model_to_schema(model)
            for name, model in sorted(graph.models.items())
        },
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _endpoint_to_schema(endpoint: Endpoint) -> JsonObject:
    schema: JsonObject = {
        "kind": endpoint.kind,
        "module": endpoint.module,
        "multiFile": endpoint.multi_file,
        "name": endpoint.name,
        "returns": _type_ref_to_schema(endpoint.returns),
    }
    _insert_optional(schema, "doc", endpoint.doc)
    _insert_non_empty(schema, "params", [_param_to_schema(param) for param in endpoint.params])
    _insert_optional(schema, "channelItem", _optional_type_ref(endpoint.channel_item))
    _insert_optional(schema, "fileParam", endpoint.file_param)
    _insert_optional(schema, "maxSize", endpoint.max_size)
    _insert_non_empty(schema, "allowedTypes", sorted(endpoint.allowed_types))
    _insert_non_empty(
        schema,
        "serverEvents",
        _event_params_to_schema(endpoint.server_events),
    )
    _insert_non_empty(
        schema,
        "clientEvents",
        _event_params_to_schema(endpoint.client_events),
    )
    return schema


def _model_to_schema(model: ModelDef) -> JsonObject:
    schema: JsonObject = {
        "fields": [_field_to_schema(field) for field in model.fields],
        "name": model.name,
    }
    _insert_optional(schema, "doc", model.doc)
    return schema


def _enum_to_schema(enum_def: EnumDef | type[Enum]) -> JsonObject:
    if isinstance(enum_def, type) and issubclass(enum_def, Enum):
        name = enum_def.__name__
        values = [_jsonable(member.value) for member in enum_def]
        doc = enum_def.__doc__
    else:
        name = enum_def.name
        values = [_jsonable(value) for value in enum_def.values]
        doc = enum_def.doc
    schema: JsonObject = {"name": name, "values": values}
    _insert_optional(schema, "doc", doc)
    return schema


def _param_to_schema(param: Param) -> JsonObject:
    schema: JsonObject = {
        "required": param.required,
        "sourceName": param.py_name,
        "ty": _type_ref_to_schema(
            param.type,
            optional=param.type.optional or not param.required,
            nullable=param.type.nullable,
        ),
        "wireName": param.ts_name,
    }
    _insert_jsonable_default(schema, param.default)
    return schema


def _field_to_schema(field: Field) -> JsonObject:
    optional = field.type.optional or not field.required
    nullable = field.type.nullable
    schema: JsonObject = {
        "nullable": nullable,
        "optional": optional,
        "required": field.required,
        "sourceName": field.py_name,
        "ty": _type_ref_to_schema(field.type, optional=optional, nullable=nullable),
        "wireName": field.ts_name,
    }
    _insert_optional(schema, "doc", field.doc)
    if not field.required:
        _insert_jsonable_default(schema, field.default)
    return schema


def _type_ref_to_schema(
    ref: TypeRef,
    *,
    optional: bool | None = None,
    nullable: bool | None = None,
) -> JsonObject:
    schema: JsonObject = {
        "kind": ref.kind,
        "nullable": ref.nullable if nullable is None else nullable,
        "optional": ref.optional if optional is None else optional,
    }
    _insert_optional(schema, "name", ref.name)
    _insert_non_empty(
        schema,
        "inner",
        [_type_ref_to_schema(inner) for inner in ref.inner],
    )
    if ref.kind == "literal":
        schema["value"] = _jsonable(ref.value)
    return schema


def _optional_type_ref(ref: TypeRef | None) -> JsonObject | None:
    if ref is None:
        return None
    return _type_ref_to_schema(ref)


def _event_params_to_schema(events: dict[str, TypeRef]) -> list[JsonObject]:
    return [
        _param_to_schema(
            Param(
                py_name=name,
                ts_name=python_name_to_camel_case(name),
                type=ty,
                required=True,
            )
        )
        for name, ty in sorted(events.items())
    ]


def _insert_optional(schema: JsonObject, key: str, value: Any) -> None:
    if value is not None:
        schema[key] = value


def _insert_non_empty(schema: JsonObject, key: str, value: list[Any]) -> None:
    if value:
        schema[key] = value


def _insert_jsonable_default(schema: JsonObject, default: Any) -> None:
    if default is MISSING or default is PydanticUndefined:
        return
    try:
        schema["default"] = _jsonable(default)
    except TypeError:
        return


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return _jsonable(value.value)
    json.dumps(value)
    return value
