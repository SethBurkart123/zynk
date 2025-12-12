"""
Channel Module

Provides streaming support for Zynk commands via Server-Sent Events (SSE).
Allows Python functions to send multiple updates to the frontend over time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ChannelStatus(Enum):
    """Status of a channel."""
    OPEN = "open"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ChannelMessage:
    """A message sent through a channel."""
    event: str
    data: Any
    channel_id: str

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        data_str = json.dumps(self.data) if not isinstance(self.data, str) else self.data
        return f"event: {self.event}\ndata: {data_str}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event,
            "data": self.data,
            "channelId": self.channel_id,
        }


class Channel(Generic[T]):
    """
    A channel for streaming data from Python to the frontend.

    Usage:
        @command
        async def stream_data(query: str, channel: Channel[DataPoint]) -> None:
            for i in range(10):
                await channel.send(DataPoint(value=i))
                await asyncio.sleep(0.1)

    The channel is automatically created and managed by Zynk.
    """

    def __init__(self, channel_id: str | None = None):
        self.id = channel_id or str(uuid.uuid4())
        self._queue: asyncio.Queue[ChannelMessage] = asyncio.Queue()
        self._status = ChannelStatus.OPEN
        self._closed_event = asyncio.Event()

    @property
    def status(self) -> ChannelStatus:
        """Get the current channel status."""
        return self._status

    @property
    def is_open(self) -> bool:
        """Check if the channel is open."""
        return self._status == ChannelStatus.OPEN

    async def send(self, data: T) -> None:
        """
        Send data through the channel.

        Args:
            data: The data to send. If it's a Pydantic model, it will be
                  automatically serialized to JSON.

        Raises:
            RuntimeError: If the channel is closed.
        """
        if not self.is_open:
            raise RuntimeError(f"Cannot send on closed channel {self.id}")

        # Serialize Pydantic models
        from pydantic import BaseModel
        if isinstance(data, BaseModel):
            serialized = data.model_dump()
        else:
            serialized = data

        message = ChannelMessage(
            event="message",
            data=serialized,
            channel_id=self.id,
        )
        await self._queue.put(message)
        logger.debug(f"Channel {self.id}: sent message")

    async def send_error(self, error: str) -> None:
        """
        Send an error through the channel.

        Args:
            error: The error message.
        """
        message = ChannelMessage(
            event="error",
            data={"error": error},
            channel_id=self.id,
        )
        await self._queue.put(message)
        self._status = ChannelStatus.ERROR

    async def close(self) -> None:
        """Close the channel."""
        if self._status == ChannelStatus.OPEN:
            message = ChannelMessage(
                event="close",
                data={"channelId": self.id},
                channel_id=self.id,
            )
            await self._queue.put(message)
            self._status = ChannelStatus.CLOSED
            self._closed_event.set()
            logger.debug(f"Channel {self.id}: closed")

    async def receive(self) -> ChannelMessage | None:
        """
        Receive the next message from the channel.

        Returns:
            The next message, or None if the channel is closed.
        """
        try:
            message = await asyncio.wait_for(self._queue.get(), timeout=30.0)
            return message
        except asyncio.TimeoutError:
            # Send keepalive
            return ChannelMessage(
                event="keepalive",
                data={},
                channel_id=self.id,
            )

    async def __aiter__(self):
        """Async iterator for receiving messages."""
        while self.is_open or not self._queue.empty():
            message = await self.receive()
            if message:
                yield message
                if message.event in ("close", "error"):
                    break


@dataclass
class ChannelManager:
    """
    Manages all active channels.

    Provides channel creation, lookup, and cleanup.
    """

    _channels: dict[str, Channel] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def create(self, channel_id: str | None = None) -> Channel:
        """Create a new channel."""
        async with self._lock:
            channel = Channel(channel_id)
            self._channels[channel.id] = channel
            logger.debug(f"ChannelManager: created channel {channel.id}")
            return channel

    async def get(self, channel_id: str) -> Channel | None:
        """Get a channel by ID."""
        return self._channels.get(channel_id)

    async def remove(self, channel_id: str) -> None:
        """Remove a channel."""
        async with self._lock:
            if channel_id in self._channels:
                channel = self._channels.pop(channel_id)
                if channel.is_open:
                    await channel.close()
                logger.debug(f"ChannelManager: removed channel {channel_id}")

    async def cleanup_closed(self) -> None:
        """Remove all closed channels."""
        async with self._lock:
            closed = [
                cid for cid, ch in self._channels.items()
                if ch.status != ChannelStatus.OPEN
            ]
            for cid in closed:
                del self._channels[cid]
            if closed:
                logger.debug(f"ChannelManager: cleaned up {len(closed)} channels")

    def get_active_count(self) -> int:
        """Get the number of active channels."""
        return len([ch for ch in self._channels.values() if ch.is_open])


# Global channel manager
channel_manager = ChannelManager()
