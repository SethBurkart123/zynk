"""
WebSocket Module

Provides bidirectional WebSocket communication for Zynk with full type safety.
Allows Python functions to define typed message handlers for real-time communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, get_type_hints

from fastapi import WebSocket as FastAPIWebSocket
from fastapi import WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variables for server/client event classes
ServerEvents = TypeVar("ServerEvents")
ClientEvents = TypeVar("ClientEvents")


class WebSocketStatus(Enum):
    """Status of a WebSocket connection."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """A message sent through the WebSocket."""

    event: str
    data: Any

    def to_json(self) -> str:
        """Convert to JSON string for sending."""
        return json.dumps({"event": self.event, "data": self.data})

    @classmethod
    def from_json(cls, json_str: str) -> WebSocketMessage:
        """Parse a JSON string into a WebSocketMessage."""
        parsed = json.loads(json_str)
        return cls(event=parsed.get("event", "message"), data=parsed.get("data", {}))


def _extract_event_types(events_class: type | None) -> dict[str, type]:
    """
    Extract event name -> type mappings from an events class.

    Supports both class attributes with type annotations and __annotations__.
    """
    if events_class is None:
        return {}

    event_types: dict[str, type] = {}

    # Get type hints from the class
    try:
        hints = get_type_hints(events_class)
        for name, type_hint in hints.items():
            if not name.startswith("_"):
                event_types[name] = type_hint
    except Exception:
        pass

    # Also check __annotations__ directly
    if hasattr(events_class, "__annotations__"):
        for name, type_hint in events_class.__annotations__.items():
            if not name.startswith("_") and name not in event_types:
                event_types[name] = type_hint

    return event_types


class WebSocket(Generic[ServerEvents, ClientEvents]):
    """
    A type-safe WebSocket connection for bidirectional communication.

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

    The WebSocket is automatically created and managed by Zynk.
    """

    def __init__(
        self,
        websocket: FastAPIWebSocket,
        server_events: type[ServerEvents] | None = None,
        client_events: type[ClientEvents] | None = None,
    ):
        self._websocket = websocket
        self._status = WebSocketStatus.CONNECTING
        self._handlers: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self._server_events = server_events
        self._client_events = client_events
        self._server_event_types = _extract_event_types(server_events)
        self._client_event_types = _extract_event_types(client_events)
        self._closed_event = asyncio.Event()

    @property
    def status(self) -> WebSocketStatus:
        """Get the current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is connected."""
        return self._status == WebSocketStatus.CONNECTED

    async def accept(self) -> None:
        """Accept the WebSocket connection."""
        await self._websocket.accept()
        self._status = WebSocketStatus.CONNECTED
        logger.debug("WebSocket connection accepted")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the WebSocket connection."""
        if self._status == WebSocketStatus.CONNECTED:
            await self._websocket.close(code=code, reason=reason)
            self._status = WebSocketStatus.DISCONNECTED
            self._closed_event.set()
            logger.debug(f"WebSocket closed: {reason or 'normal closure'}")

    async def send(self, event: str, data: Any) -> None:
        """
        Send a message to the client.

        Args:
            event: The event name (must be defined in ServerEvents).
            data: The data to send. Pydantic models are automatically serialized.

        Raises:
            RuntimeError: If the WebSocket is not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Cannot send on disconnected WebSocket")

        # Serialize Pydantic models
        if isinstance(data, BaseModel):
            serialized = data.model_dump()
        elif isinstance(data, dict):
            serialized = data
        else:
            serialized = data

        message = WebSocketMessage(event=event, data=serialized)
        await self._websocket.send_text(message.to_json())
        logger.debug(f"WebSocket sent: {event}")

    def on(
        self, event: str
    ) -> Callable[[Callable[[Any], Awaitable[None]]], Callable[[Any], Awaitable[None]]]:
        """
        Register an event handler.

        Usage:
            @ws.on("chat_message")
            async def handle_chat(data: ChatMessage):
                await ws.send("response", Response(...))

        Args:
            event: The event name to handle (must be defined in ClientEvents).

        Returns:
            A decorator that registers the handler function.
        """

        def decorator(
            handler: Callable[[Any], Awaitable[None]],
        ) -> Callable[[Any], Awaitable[None]]:
            self._handlers[event] = handler
            logger.debug(f"WebSocket handler registered: {event}")
            return handler

        return decorator

    async def _handle_message(self, raw_message: str) -> None:
        """Handle an incoming message."""
        try:
            message = WebSocketMessage.from_json(raw_message)
            event = message.event
            data = message.data

            # Get the expected type for this event
            expected_type = self._client_event_types.get(event)

            # Instantiate Pydantic model if we have a type
            if (
                expected_type
                and isinstance(expected_type, type)
                and issubclass(expected_type, BaseModel)
            ):
                if isinstance(data, dict):
                    data = expected_type(**data)

            # Call the handler if registered
            handler = self._handlers.get(event)
            if handler:
                await handler(data)
            else:
                logger.warning(f"No handler registered for event: {event}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception:
            raise

    async def listen(self) -> None:
        """
        Start listening for messages.

        This method blocks until the connection is closed.
        Call this after registering all event handlers.
        """
        if not self.is_connected:
            await self.accept()

        try:
            while self.is_connected:
                try:
                    raw_message = await self._websocket.receive_text()
                    await self._handle_message(raw_message)
                except WebSocketDisconnect:
                    logger.debug("WebSocket client disconnected")
                    break
                except Exception:
                    raise
        finally:
            self._status = WebSocketStatus.DISCONNECTED
            self._closed_event.set()

    async def wait_closed(self) -> None:
        """Wait until the WebSocket is closed."""
        await self._closed_event.wait()


@dataclass
class MessageHandlerInfo:
    """Stores metadata about a registered message handler."""

    name: str
    func: Callable
    server_events: type | None
    client_events: type | None
    server_event_types: dict[str, type]
    client_event_types: dict[str, type]
    docstring: str | None
    module: str

    def __repr__(self) -> str:
        return f"MessageHandlerInfo(name={self.name!r}, module={self.module!r})"
