"""Configuration options for the Effect code generator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Surface = Literal["effect", "promise"]

_VALID_KEYS = {"default", "commands", "uploads", "statics", "channels", "websockets"}


@dataclass(slots=True)
class EffectGeneratorOptions:
    """Controls what surface each endpoint kind exposes.

    ``"effect"`` emits raw ``Effect.Effect`` / ``Stream.Stream`` return types.
    ``"promise"`` wraps each call in ``runPromise`` so callers use plain ``await``.

    Per-kind fields default to ``None`` which inherits from ``default``.
    """

    default: Surface = "effect"
    commands: Surface | None = None
    uploads: Surface | None = None
    statics: Surface | None = None
    channels: Surface | None = None
    websockets: Surface | None = None

    def resolve(self, kind: str) -> Surface:
        override = getattr(self, kind, None)
        return override if override is not None else self.default

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EffectGeneratorOptions:
        unknown = set(raw.keys()) - _VALID_KEYS
        if unknown:
            raise ValueError(f"Unknown effect generator options: {', '.join(sorted(unknown))}")
        return cls(
            default=raw.get("default", "effect"),
            commands=raw.get("commands"),
            uploads=raw.get("uploads"),
            statics=raw.get("statics"),
            channels=raw.get("channels"),
            websockets=raw.get("websockets"),
        )
