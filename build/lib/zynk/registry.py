"""
Command Registry Module

Provides the @command and @message decorators and global registry for Zynk.
Commands and message handlers are registered automatically when decorated,
enabling zero-config setup.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from .websocket import MessageHandlerInfo, WebSocket, _extract_event_types

if TYPE_CHECKING:
    from .upload import UploadInfo


class CommandInfo:
    """Stores metadata about a registered command."""

    def __init__(
        self,
        name: str,
        func: Callable,
        params: dict[str, type],
        return_type: type | None,
        is_async: bool,
        docstring: str | None,
        module: str,
        has_channel: bool = False,
        optional_params: set[str] | None = None,
    ):
        self.name = name
        self.func = func
        self.params = params
        self.return_type = return_type
        self.is_async = is_async
        self.docstring = docstring
        self.module = module
        self.has_channel = has_channel
        self.optional_params = optional_params or set()

    def __repr__(self) -> str:
        return f"CommandInfo(name={self.name!r}, module={self.module!r})"


class CommandRegistry:
    """
    Global registry for all Zynk commands and message handlers.

    Maintains a flat namespace of commands/handlers and ensures no duplicates.
    Collects Pydantic models used in signatures for TS generation.
    """

    _instance: CommandRegistry | None = None

    def __new__(cls) -> CommandRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._commands: dict[str, CommandInfo] = {}
            cls._instance._models: dict[str, type[BaseModel]] = {}
            cls._instance._message_handlers: dict[str, MessageHandlerInfo] = {}
            cls._instance._uploads: dict[str, "UploadInfo"] = {}
            cls._instance._initialized = True
        return cls._instance

    @classmethod
    def get_instance(cls) -> CommandRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        if cls._instance is not None:
            cls._instance._commands.clear()
            cls._instance._models.clear()
            cls._instance._message_handlers.clear()
            cls._instance._uploads.clear()

    def register(self, cmd: CommandInfo) -> None:
        """
        Register a command.

        Raises:
            ValueError: If a command with the same name already exists.
        """
        if cmd.name in self._commands:
            existing = self._commands[cmd.name]
            raise ValueError(
                f"Command name conflict: '{cmd.name}' is defined in both "
                f"'{existing.module}' and '{cmd.module}'. "
                f"Command names must be unique across all modules."
            )
        self._commands[cmd.name] = cmd

    def register_model(self, model: type[BaseModel]) -> None:
        """Register a Pydantic model for TypeScript generation."""
        if model.__name__ not in self._models:
            self._models[model.__name__] = model

    def get_command(self, name: str) -> CommandInfo | None:
        """Get a command by name."""
        return self._commands.get(name)

    def get_all_commands(self) -> dict[str, CommandInfo]:
        """Get all registered commands."""
        return self._commands.copy()

    def get_all_models(self) -> dict[str, type[BaseModel]]:
        """Get all registered Pydantic models."""
        return self._models.copy()

    def register_message_handler(self, handler: MessageHandlerInfo) -> None:
        """
        Register a message handler.

        Raises:
            ValueError: If a handler with the same name already exists.
        """
        if handler.name in self._message_handlers:
            existing = self._message_handlers[handler.name]
            raise ValueError(
                f"Message handler name conflict: '{handler.name}' is defined in both "
                f"'{existing.module}' and '{handler.module}'. "
                f"Handler names must be unique across all modules."
            )
        self._message_handlers[handler.name] = handler

    def get_message_handler(self, name: str) -> MessageHandlerInfo | None:
        """Get a message handler by name."""
        return self._message_handlers.get(name)

    def get_all_message_handlers(self) -> dict[str, MessageHandlerInfo]:
        """Get all registered message handlers."""
        return self._message_handlers.copy()

    def register_upload(self, upload_info: "UploadInfo") -> None:
        """
        Register an upload handler.

        Raises:
            ValueError: If an upload handler with the same name already exists.
        """
        if upload_info.name in self._uploads:
            existing = self._uploads[upload_info.name]
            raise ValueError(
                f"Upload handler name conflict: '{upload_info.name}' is defined in both "
                f"'{existing.module}' and '{upload_info.module}'. "
                f"Handler names must be unique across all modules."
            )
        self._uploads[upload_info.name] = upload_info

    def get_upload(self, name: str) -> "UploadInfo | None":
        """Get an upload handler by name."""
        return self._uploads.get(name)

    def get_all_uploads(self) -> dict[str, "UploadInfo"]:
        """Get all registered upload handlers."""
        return self._uploads.copy()

    def collect_models_from_type(self, type_hint: Any) -> None:
        """
        Recursively collect Pydantic models from a type hint.

        Handles Optional, List, Dict, and nested models.
        """
        if type_hint is None:
            return

        origin = get_origin(type_hint)

        # Handle Union types (both old Union[T, None] and new T | None syntax)
        if origin is Union or origin is types.UnionType:
            args = get_args(type_hint)
            for arg in args:
                if arg is not type(None):  # Skip None types in unions
                    self.collect_models_from_type(arg)
            return

        if origin is not None:
            args = get_args(type_hint)
            for arg in args:
                self.collect_models_from_type(arg)
            return

        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            if type_hint.__name__ not in self._models:
                self._models[type_hint.__name__] = type_hint
                for field_name, field_info in type_hint.model_fields.items():
                    self.collect_models_from_type(field_info.annotation)


# Global registry instance
_registry = CommandRegistry.get_instance()


def command(func: Callable = None, *, name: str | None = None) -> Callable:
    """
    Decorator to register a function as a Zynk command.

    Usage:
        @command
        async def get_user(user_id: int) -> User:
            ...

        @command(name="custom_name")
        def my_function() -> str:
            ...

    Args:
        func: The function to decorate.
        name: Optional custom command name (defaults to function name).

    Returns:
        The decorated function.
    """

    def decorator(fn: Callable) -> Callable:
        cmd_name = name or fn.__name__

        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        sig = inspect.signature(fn)
        params: dict[str, type] = {}
        optional_params: set[str] = set()
        has_channel = False

        for param_name, param in sig.parameters.items():
            if param_name == "channel":
                has_channel = True
                channel_type = hints.get("channel")
                if channel_type:
                    from typing import get_args

                    channel_args = get_args(channel_type)
                    if channel_args:
                        _registry.collect_models_from_type(channel_args[0])
                continue

            param_type = hints.get(param_name, Any)
            params[param_name] = param_type
            _registry.collect_models_from_type(param_type)

            # Track params with defaults as optional
            if param.default is not inspect.Parameter.empty:
                optional_params.add(param_name)

        return_type = hints.get("return", None)
        _registry.collect_models_from_type(return_type)

        is_async = asyncio.iscoroutinefunction(fn)
        module = fn.__module__

        cmd_info = CommandInfo(
            name=cmd_name,
            func=fn,
            params=params,
            return_type=return_type,
            is_async=is_async,
            docstring=fn.__doc__,
            module=module,
            has_channel=has_channel,
            optional_params=optional_params,
        )

        _registry.register(cmd_info)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            if is_async:
                return await fn(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        async_wrapper._zynk_command = cmd_info
        return async_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def get_registry() -> CommandRegistry:
    """Get the global command registry."""
    return _registry


def message(func: Callable = None, *, name: str | None = None) -> Callable:
    """
    Decorator to register a function as a Zynk WebSocket message handler.

    Usage:
        class ServerEvents:
            chat_message: ChatMessage
            status: StatusUpdate

        class ClientEvents:
            chat_message: ChatMessage
            typing: TypingIndicator

        @message
        async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            @ws.on("chat_message")
            async def handle_chat(data: ChatMessage):
                await ws.send("chat_message", data)

            await ws.listen()

        @message(name="custom_name")
        async def my_handler(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            ...

    Args:
        func: The function to decorate.
        name: Optional custom handler name (defaults to function name).

    Returns:
        The decorated function.
    """

    def decorator(fn: Callable) -> Callable:
        handler_name = name or fn.__name__

        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        # Find the WebSocket parameter and extract its type arguments
        sig = inspect.signature(fn)
        server_events: type | None = None
        client_events: type | None = None
        ws_param_name: str | None = None

        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name)
            if param_type is not None:
                origin = get_origin(param_type)
                # Check if it's WebSocket[ServerEvents, ClientEvents]
                if origin is WebSocket:
                    ws_param_name = param_name
                    args = get_args(param_type)
                    if len(args) >= 2:
                        server_events = args[0]
                        client_events = args[1]
                    break

        if ws_param_name is None:
            raise ValueError(
                f"Message handler '{handler_name}' must have a WebSocket parameter. "
                f"Example: async def {handler_name}(ws: WebSocket[ServerEvents, ClientEvents])"
            )

        # Extract event types from the event classes
        server_event_types = _extract_event_types(server_events)
        client_event_types = _extract_event_types(client_events)

        # Collect Pydantic models from event types
        for event_type in server_event_types.values():
            _registry.collect_models_from_type(event_type)
        for event_type in client_event_types.values():
            _registry.collect_models_from_type(event_type)

        module = fn.__module__

        handler_info = MessageHandlerInfo(
            name=handler_name,
            func=fn,
            server_events=server_events,
            client_events=client_events,
            server_event_types=server_event_types,
            client_event_types=client_event_types,
            docstring=fn.__doc__,
            module=module,
        )

        _registry.register_message_handler(handler_info)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        async_wrapper._zynk_message_handler = handler_info
        return async_wrapper

    if func is not None:
        return decorator(func)
    return decorator
