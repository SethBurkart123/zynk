"""Naming helpers shared by runtime and code generation."""

from __future__ import annotations


def python_name_to_camel_case(name: str) -> str:
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def python_name_to_pascal_case(name: str) -> str:
    return "".join(x.title() for x in name.split("_"))
