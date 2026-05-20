"""
Tests for WebSocket support and @message decorator.
"""

import pytest
from pydantic import BaseModel

from zynk import message, WebSocket, get_registry
from zynk.registry import CommandRegistry
from zynk.websocket import _extract_event_types, MessageHandlerInfo


# Test models
class ChatMessage(BaseModel):
    user: str
    text: str


class TypingIndicator(BaseModel):
    user: str
    is_typing: bool


class StatusUpdate(BaseModel):
    count: int


# Test event classes
class ServerEvents:
    chat_message: ChatMessage
    typing: TypingIndicator
    status: StatusUpdate


class ClientEvents:
    chat_message: ChatMessage
    typing: TypingIndicator


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test."""
    CommandRegistry.reset()
    yield


class TestExtractEventTypes:
    """Test the _extract_event_types function."""

    def test_extract_from_class(self):
        """Test extracting event types from a class."""
        event_types = _extract_event_types(ServerEvents)

        assert "chat_message" in event_types
        assert "typing" in event_types
        assert "status" in event_types
        assert event_types["chat_message"] is ChatMessage
        assert event_types["typing"] is TypingIndicator
        assert event_types["status"] is StatusUpdate

    def test_extract_from_none(self):
        """Test extracting from None returns empty dict."""
        event_types = _extract_event_types(None)
        assert event_types == {}

    def test_excludes_private_attributes(self):
        """Test that private attributes are excluded."""

        class EventsWithPrivate:
            public_event: ChatMessage
            _private_event: TypingIndicator

        event_types = _extract_event_types(EventsWithPrivate)
        assert "public_event" in event_types
        assert "_private_event" not in event_types


class TestMessageDecorator:
    """Test the @message decorator."""

    def test_basic_registration(self):
        """Test that @message registers the handler."""

        @message
        async def test_handler(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        registry = get_registry()
        handler = registry.get_message_handler("test_handler")

        assert handler is not None
        assert handler.name == "test_handler"
        assert handler.server_events is ServerEvents
        assert handler.client_events is ClientEvents

    def test_custom_name(self):
        """Test @message with custom name."""

        @message(name="custom_ws")
        async def my_handler(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        registry = get_registry()
        handler = registry.get_message_handler("custom_ws")

        assert handler is not None
        assert handler.name == "custom_ws"

    def test_extracts_event_types(self):
        """Test that event types are extracted correctly."""

        @message
        async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        registry = get_registry()
        handler = registry.get_message_handler("chat")

        assert "chat_message" in handler.server_event_types
        assert "typing" in handler.server_event_types
        assert "status" in handler.server_event_types

        assert "chat_message" in handler.client_event_types
        assert "typing" in handler.client_event_types
        assert "status" not in handler.client_event_types

    def test_collects_pydantic_models(self):
        """Test that Pydantic models are collected for TS generation."""

        @message
        async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        registry = get_registry()
        models = registry.get_all_models()

        assert "ChatMessage" in models
        assert "TypingIndicator" in models
        assert "StatusUpdate" in models

    def test_requires_websocket_param(self):
        """Test that handlers must have a WebSocket parameter."""
        with pytest.raises(ValueError, match="must have a WebSocket parameter"):

            @message
            async def bad_handler(x: int) -> None:
                pass

    def test_duplicate_handler_raises_error(self):
        """Test that duplicate handler names raise an error."""

        @message
        async def same_name(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        with pytest.raises(ValueError, match="name conflict"):

            @message
            async def same_name(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
                pass

    def test_preserves_docstring(self):
        """Test that the docstring is preserved."""

        @message
        async def documented_handler(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            """This is a documented handler."""
            pass

        registry = get_registry()
        handler = registry.get_message_handler("documented_handler")

        assert handler.docstring == "This is a documented handler."


class TestGetAllMessageHandlers:
    """Test the get_all_message_handlers method."""

    def test_returns_all_handlers(self):
        """Test that all registered handlers are returned."""

        @message
        async def handler1(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        @message
        async def handler2(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
            pass

        registry = get_registry()
        handlers = registry.get_all_message_handlers()

        assert len(handlers) == 2
        assert "handler1" in handlers
        assert "handler2" in handlers
