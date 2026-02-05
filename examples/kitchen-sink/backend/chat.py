"""
Chat Module

Demonstrates WebSocket bidirectional communication with typed messages.
"""

from datetime import datetime

from pydantic import BaseModel

from zynk import WebSocket, message


# Message types
class ChatMessage(BaseModel):
    """A chat message sent by a user."""

    user: str
    text: str
    timestamp: str


class TypingIndicator(BaseModel):
    """Indicates a user is typing."""

    user: str
    is_typing: bool


class UserJoined(BaseModel):
    """Notification that a user joined the chat."""

    user: str
    timestamp: str


class UserLeft(BaseModel):
    """Notification that a user left the chat."""

    user: str
    timestamp: str


class ServerStatus(BaseModel):
    """Server status information."""

    connected_users: int
    uptime_seconds: float


# Event schemas - these define what messages can be sent/received
class ServerEvents:
    """Messages the server can send to the client."""

    chat_message: ChatMessage
    typing: TypingIndicator
    user_joined: UserJoined
    user_left: UserLeft
    status: ServerStatus


class ClientEvents:
    """Messages the client can send to the server."""

    chat_message: ChatMessage
    typing: TypingIndicator
    join: UserJoined
    leave: UserLeft


_connected_clients: set["WebSocket[ServerEvents, ClientEvents]"] = set()
_connected_users: set[str] = set()
_start_time = datetime.now()


async def broadcast(event: str, data: BaseModel) -> None:
    """Broadcast a message to all connected clients."""
    for client in list(_connected_clients):
        if client.is_connected:
            await client.send(event, data)


@message
async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
    """
    Real-time chat WebSocket endpoint.

    Supports:
    - Sending and receiving chat messages
    - Typing indicators
    - Join/leave notifications
    - Server status updates
    """
    current_user: str | None = None
    _connected_clients.add(ws)

    @ws.on("join")
    async def on_join(data: UserJoined):
        nonlocal current_user
        current_user = data.user
        _connected_users.add(data.user)

        await broadcast(
            "user_joined",
            UserJoined(user=data.user, timestamp=datetime.now().isoformat()),
        )

        uptime = (datetime.now() - _start_time).total_seconds()
        await broadcast(
            "status",
            ServerStatus(connected_users=len(_connected_users), uptime_seconds=uptime),
        )

    @ws.on("leave")
    async def on_leave(data: UserLeft):
        nonlocal current_user
        if data.user in _connected_users:
            _connected_users.discard(data.user)
            current_user = None

            await broadcast(
                "user_left",
                UserLeft(user=data.user, timestamp=datetime.now().isoformat()),
            )

    @ws.on("chat_message")
    async def on_chat(data: ChatMessage):
        await broadcast(
            "chat_message",
            ChatMessage(
                user=data.user, text=data.text, timestamp=datetime.now().isoformat()
            ),
        )

    @ws.on("typing")
    async def on_typing(data: TypingIndicator):
        await broadcast("typing", data)

    await ws.listen()

    _connected_clients.discard(ws)
    if current_user:
        _connected_users.discard(current_user)
        await broadcast(
            "user_left",
            UserLeft(user=current_user, timestamp=datetime.now().isoformat()),
        )