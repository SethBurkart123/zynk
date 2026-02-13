"""
Static file serving for Zynk. Resolver functions return StaticFile; params become query args.
"""

from __future__ import annotations

import inspect
import logging
import mimetypes
import types
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class StaticFile:
    """File to serve statically; path with optional content_type and headers."""

    def __init__(
        self,
        path: Path | str,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.path = Path(path).resolve()
        self.content_type = content_type
        self.headers = headers or {}

        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Not a regular file: {self.path}")

    @classmethod
    def from_directory(
        cls,
        base_dir: Path | str,
        filename: str,
        content_type: str | None = None,
    ) -> "StaticFile":
        """Resolve a file under base_dir; raises ValueError if path escapes base."""
        base = Path(base_dir).resolve()
        target = (base / filename).resolve()

        if not target.is_relative_to(base):
            raise ValueError("Path traversal detected")

        return cls(path=target, content_type=content_type)

    def guess_content_type(self) -> str:
        """Guess the MIME type from the file extension."""
        if self.content_type:
            return self.content_type

        mime_type, _ = mimetypes.guess_type(str(self.path))
        return mime_type or "application/octet-stream"

    def __repr__(self) -> str:
        return f"StaticFile({self.path!r}, content_type={self.content_type!r})"


@dataclass
class StaticInfo:
    """Metadata for a registered static file handler."""

    name: str
    func: Callable[..., Any]
    params: dict[str, Any]
    return_type: Any
    is_async: bool
    docstring: str | None
    module: str
    optional_params: set[str]


def static(
    fn: F | None = None,
    *,
    name: str | None = None,
) -> F | Callable[[F], F]:
    """Register a function as a static file handler; params become URL query args."""

    def decorator(func: F) -> F:
        from .registry import get_registry

        handler_name = name or func.__name__
        is_async = inspect.iscoroutinefunction(func)
        docstring = func.__doc__
        module = func.__module__

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        sig = inspect.signature(func)

        params: dict[str, Any] = {}
        optional_params: set[str] = set()

        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, param.annotation)

            # Handle Optional types
            origin = get_origin(param_type)
            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                args = get_args(param_type)
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    param_type = non_none_args[0]

            params[param_name] = param_type
            if param.default is not inspect.Parameter.empty:
                optional_params.add(param_name)

        return_type = hints.get("return", None)

        static_info = StaticInfo(
            name=handler_name,
            func=func,
            params=params,
            return_type=return_type,
            is_async=is_async,
            docstring=docstring,
            module=module,
            optional_params=optional_params,
        )

        registry = get_registry()
        registry.register_static(static_info)

        for param_type in params.values():
            registry.collect_models_from_type(param_type)

        logger.debug(f"Registered static handler: {handler_name}")

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_async:
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    if fn is not None:
        return decorator(fn)
    return decorator
