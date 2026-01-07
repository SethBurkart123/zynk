"""
Upload Module

Provides file upload support for Zynk with XMLHttpRequest progress tracking.
Supports single and multiple file uploads with validation.
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
import re
import types
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from fastapi import UploadFile as FastAPIUploadFile

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def parse_size(size: str | int) -> int:
    """
    Parse a size string like "10MB" or "500KB" to bytes.

    Args:
        size: Size as int (bytes) or string with unit (KB, MB, GB).

    Returns:
        Size in bytes.

    Raises:
        ValueError: If the size string is invalid.
    """
    if isinstance(size, int):
        return size

    size = size.strip().upper()

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
    }

    match = re.match(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)?$", size)
    if not match:
        raise ValueError(f"Invalid size format: {size}")

    value = float(match.group(1))
    unit = match.group(2) or "B"

    return int(value * units[unit])


def matches_mime_type(mime_type: str, patterns: list[str]) -> bool:
    """
    Check if a MIME type matches any of the patterns.

    Supports wildcards like "image/*" or exact matches like "application/pdf".

    Args:
        mime_type: The MIME type to check.
        patterns: List of patterns to match against.

    Returns:
        True if the MIME type matches any pattern.
    """
    if not patterns:
        return True

    for pattern in patterns:
        if fnmatch.fnmatch(mime_type, pattern):
            return True

    return False


class UploadFile:
    """
    Represents an uploaded file.

    Wraps FastAPI's UploadFile with additional metadata and convenience methods.

    Attributes:
        filename: Original filename from the client.
        content_type: MIME type of the file.
        size: Size of the file in bytes.
    """

    def __init__(
        self,
        fastapi_file: FastAPIUploadFile,
        size: int | None = None,
    ):
        self._file = fastapi_file
        self._size = size
        self._content: bytes | None = None

    @property
    def filename(self) -> str:
        """Original filename from the client."""
        return self._file.filename or "unknown"

    @property
    def content_type(self) -> str:
        """MIME type of the file."""
        return self._file.content_type or "application/octet-stream"

    @property
    def size(self) -> int | None:
        """Size of the file in bytes (if known)."""
        return self._size

    async def read(self) -> bytes:
        """
        Read the entire file content.

        Returns:
            The file content as bytes.
        """
        if self._content is None:
            self._content = await self._file.read()
            if self._size is None:
                self._size = len(self._content)
        return self._content

    async def seek(self, offset: int) -> None:
        """Seek to a position in the file."""
        await self._file.seek(offset)

    async def close(self) -> None:
        """Close the file."""
        await self._file.close()

    def __repr__(self) -> str:
        size_str = f"{self._size} bytes" if self._size else "unknown size"
        return f"UploadFile({self.filename!r}, {self.content_type}, {size_str})"


@dataclass
class UploadInfo:
    """
    Information about a registered upload handler.

    Stores metadata about upload handlers for routing and TypeScript generation.
    """

    name: str
    func: Callable[..., Any]
    params: dict[str, Any]  # Non-file parameters
    file_param: str | None  # Name of the UploadFile parameter
    is_multi_file: bool  # True if list[UploadFile], False if single UploadFile
    return_type: Any
    is_async: bool
    docstring: str | None
    module: str
    optional_params: set[str]

    # Validation options
    max_size: int | None = None  # Max file size in bytes
    allowed_types: list[str] = field(default_factory=list)  # Allowed MIME types

    def validate_file(self, file: UploadFile) -> None:
        """
        Validate a file against the upload constraints.

        Args:
            file: The file to validate.

        Raises:
            ValueError: If validation fails.
        """
        if self.max_size is not None and file.size is not None:
            if file.size > self.max_size:
                max_mb = self.max_size / (1024 * 1024)
                file_mb = file.size / (1024 * 1024)
                raise ValueError(
                    f"File '{file.filename}' ({file_mb:.1f}MB) exceeds "
                    f"maximum size ({max_mb:.1f}MB)"
                )

        if self.allowed_types:
            if not matches_mime_type(file.content_type, self.allowed_types):
                raise ValueError(
                    f"File '{file.filename}' has type '{file.content_type}' "
                    f"which is not in allowed types: {self.allowed_types}"
                )


def upload(
    fn: F | None = None,
    *,
    name: str | None = None,
    max_size: str | int | None = None,
    allowed_types: list[str] | None = None,
) -> F | Callable[[F], F]:
    """
    Decorator to register a function as an upload handler.

    The decorated function should accept an `UploadFile` or `list[UploadFile]`
    parameter for the uploaded files, plus any additional parameters.

    Args:
        fn: The function to decorate (when used without parentheses).
        name: Custom name for the upload handler. Defaults to function name.
        max_size: Maximum file size (e.g., "10MB", "500KB", or bytes as int).
        allowed_types: List of allowed MIME types (supports wildcards like "image/*").

    Returns:
        The decorated function.

    Example:
        @upload
        async def upload_avatar(file: UploadFile, user_id: str) -> AvatarResult:
            content = await file.read()
            # ... process file ...
            return AvatarResult(url="...")

        @upload(max_size="10MB", allowed_types=["image/*"])
        async def upload_images(files: list[UploadFile]) -> list[ImageResult]:
            results = []
            for file in files:
                # ... process each file ...
                results.append(ImageResult(...))
            return results
    """

    def decorator(func: F) -> F:
        from .registry import get_registry

        # Get function metadata
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

        # Find the UploadFile parameter and other params
        file_param: str | None = None
        is_multi_file = False
        params: dict[str, Any] = {}
        optional_params: set[str] = set()

        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, param.annotation)

            # Check if this is an UploadFile parameter
            origin = get_origin(param_type)
            args = get_args(param_type)

            # Handle Optional[UploadFile] or Optional[list[UploadFile]]
            actual_type = param_type
            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]
                    origin = get_origin(actual_type)
                    args = get_args(actual_type)

            # Check for list[UploadFile]
            if origin is list and args:
                item_type = args[0]
                if item_type is UploadFile or (
                    isinstance(item_type, type) and item_type.__name__ == "UploadFile"
                ):
                    file_param = param_name
                    is_multi_file = True
                    continue

            # Check for single UploadFile
            if actual_type is UploadFile or (
                isinstance(actual_type, type)
                and getattr(actual_type, "__name__", None) == "UploadFile"
            ):
                file_param = param_name
                is_multi_file = False
                continue

            # Regular parameter
            params[param_name] = param_type
            if param.default is not inspect.Parameter.empty:
                optional_params.add(param_name)

        if file_param is None:
            raise TypeError(
                f"Upload handler '{handler_name}' must have a parameter of type "
                f"UploadFile or list[UploadFile]"
            )

        # Get return type
        return_type = hints.get("return", None)

        # Parse max_size
        parsed_max_size = None
        if max_size is not None:
            parsed_max_size = parse_size(max_size)

        # Create upload info
        upload_info = UploadInfo(
            name=handler_name,
            func=func,
            params=params,
            file_param=file_param,
            is_multi_file=is_multi_file,
            return_type=return_type,
            is_async=is_async,
            docstring=docstring,
            module=module,
            optional_params=optional_params,
            max_size=parsed_max_size,
            allowed_types=allowed_types or [],
        )

        # Register with the registry
        registry = get_registry()
        registry.register_upload(upload_info)

        # Collect models from params and return type
        for param_type in params.values():
            registry.collect_models_from_type(param_type)
        if return_type:
            registry.collect_models_from_type(return_type)

        logger.debug(
            f"Registered upload handler: {handler_name} "
            f"(file_param={file_param}, multi={is_multi_file})"
        )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_async:
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    # Handle both @upload and @upload(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator
