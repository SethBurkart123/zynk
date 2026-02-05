"""
Static File Serving Module

Provides static file serving support for Zynk with type-safe URL generation.
Allows serving files through resolver functions with full control over access.
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
    """
    Represents a file to be served statically.

    Wraps a file path with optional content type override.
    Includes security helpers for safe path resolution.

    Attributes:
        path: The resolved file path.
        content_type: MIME type (None = auto-detect from extension).
        headers: Additional headers to include in response.
    """

    def __init__(
        self,
        path: Path | str,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.path = Path(path).resolve()
        self.content_type = content_type
        self.headers = headers or {}

        # Security: ensure file exists and is a regular file
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
        """
        Safely resolve a file within a directory.

        Prevents path traversal attacks by ensuring the resolved path
        is within the base directory.

        Args:
            base_dir: The base directory to serve from.
            filename: The filename (may include subdirectories).
            content_type: Optional MIME type override.

        Returns:
            A StaticFile instance.

        Raises:
            ValueError: If path traversal is detected.
            FileNotFoundError: If the file doesn't exist.
        """
        base = Path(base_dir).resolve()
        target = (base / filename).resolve()

        # Security: prevent path traversal
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
    """
    Information about a registered static file handler.

    Stores metadata about static handlers for routing and TypeScript generation.
    """

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
    """
    Decorator to register a function as a static file handler.

    The decorated function should accept parameters and return a StaticFile.
    The parameters become query string arguments in the generated URL.

    Args:
        fn: The function to decorate (when used without parentheses).
        name: Custom name for the handler. Defaults to function name.

    Returns:
        The decorated function.

    Example:
        @static
        async def agent_file(agent_id: str, file_type: str) -> StaticFile:
            path = get_file_path(agent_id, file_type)
            return StaticFile(path=path)

        # Safe directory-based serving:
        @static
        async def user_avatar(user_id: str) -> StaticFile:
            return StaticFile.from_directory(
                base_dir=AVATARS_DIR,
                filename=f"{user_id}.png"
            )
    """

    def decorator(func: F) -> F:
        from .registry import get_registry

        handler_name = name or func.__name__
        is_async = inspect.iscoroutinefunction(func)
        docstring = func.__doc__
        module = func.__module__

        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        sig = inspect.signature(func)

        # Extract parameters
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

        # Get return type
        return_type = hints.get("return", None)

        # Create static info
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

        # Register with the registry
        registry = get_registry()
        registry.register_static(static_info)

        # Collect models from params
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
