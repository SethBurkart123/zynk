"""Registry for pluggable language code generators."""

from __future__ import annotations

from collections.abc import Iterable

from .base import CodeGenerator

_generators: dict[str, CodeGenerator] = {}


def register_generator(generator: CodeGenerator, *, replace: bool = False) -> None:
    language = generator.language.lower()
    if language in _generators and not replace:
        raise ValueError(f"Generator already registered for language: {language}")
    _generators[language] = generator


def get_generator(language: str) -> CodeGenerator:
    key = language.lower()
    try:
        return _generators[key]
    except KeyError:
        available = ", ".join(sorted(_generators)) or "none"
        raise ValueError(f"No generator registered for language '{language}'. Available: {available}") from None


def list_generators() -> Iterable[str]:
    return tuple(sorted(_generators))
