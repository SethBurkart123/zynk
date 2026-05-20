"""Shared callable introspection for Zynk decorators."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from zynk.core.ir import Endpoint, EndpointKind, Param, TypeRef
from zynk.core.naming import python_name_to_camel_case
from zynk.core.types import type_ref


def safe_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(fn)
    except Exception:
        return {}


def build_params(
    fn: Callable[..., Any],
    hints: dict[str, Any],
    *,
    skip: set[str] | None = None,
) -> tuple[list[Param], set[str], dict[str, Any]]:
    skip = skip or set()
    params: list[Param] = []
    optional: set[str] = set()
    raw: dict[str, Any] = {}

    for name, param in inspect.signature(fn).parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, Any)
        raw[name] = annotation
        required = param.default is inspect.Parameter.empty
        if not required:
            optional.add(name)
        params.append(
            Param(
                py_name=name,
                ts_name=python_name_to_camel_case(name),
                type=type_ref(annotation),
                required=required,
                default=param.default,
            )
        )
    return params, optional, raw


def build_endpoint(
    *,
    fn: Callable[..., Any],
    name: str,
    kind: EndpointKind,
    params: list[Param],
    returns: TypeRef,
    deps: list[Callable[..., Any]] | None = None,
    **kwargs: Any,
) -> Endpoint:
    return Endpoint(
        name=name,
        kind=kind,
        func=fn,
        is_async=inspect.iscoroutinefunction(fn),
        module=fn.__module__,
        doc=fn.__doc__,
        params=params,
        returns=returns,
        deps=deps or [],
        **kwargs,
    )
