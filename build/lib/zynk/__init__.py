"""
Zynk - Bridge Python to TypeScript Frontends

A library for creating type-safe bridges between Python backends
and TypeScript frontends with automatic client generation.

Example usage:

    # main.py
    from zynk import Bridge, command
    from pydantic import BaseModel

    class User(BaseModel):
        id: int
        name: str

    @command
    async def get_user(user_id: int) -> User:
        return User(id=user_id, name="Alice")

    app = Bridge(
        generate_ts="../frontend/src/api.ts",
        port=8000
    )

    if __name__ == "__main__":
        app.run(dev=True)

For streaming support:

    from zynk import command, Channel

    @command
    async def stream_data(channel: Channel[dict]) -> None:
        for i in range(10):
            await channel.send({"value": i})
            await asyncio.sleep(0.1)

For WebSocket bidirectional communication:

    from zynk import message, WebSocket
    from pydantic import BaseModel

    class ChatMessage(BaseModel):
        text: str

    class ServerEvents:
        chat_message: ChatMessage

    class ClientEvents:
        chat_message: ChatMessage

    @message
    async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
        @ws.on("chat_message")
        async def on_message(data: ChatMessage):
            await ws.send("chat_message", data)

        await ws.listen()

For file uploads with progress tracking:

    from zynk import upload, UploadFile
    from pydantic import BaseModel

    class UploadResult(BaseModel):
        filename: str
        size: int
        url: str

    @upload(max_size="10MB", allowed_types=["image/*", "application/pdf"])
    async def upload_file(file: UploadFile) -> UploadResult:
        content = await file.read()
        # ... save file ...
        return UploadResult(filename=file.filename, size=file.size, url="...")

    # For multiple files:
    @upload
    async def upload_files(files: list[UploadFile]) -> list[UploadResult]:
        results = []
        for file in files:
            content = await file.read()
            results.append(UploadResult(...))
        return results

For static file serving:

    from zynk import static, StaticFile

    @static
    async def user_avatar(user_id: str) -> StaticFile:
        return StaticFile.from_directory(
            base_dir=AVATARS_DIR,
            filename=f"{user_id}.png"
        )

    @static
    async def agent_file(agent_id: str, file_type: str) -> StaticFile:
        path = resolve_agent_file(agent_id, file_type)
        return StaticFile(path=path)
"""

__version__ = "0.1.3"
__author__ = "Zynk Team"

# Core exports
from .bridge import Bridge
from .channel import Channel, ChannelManager, channel_manager
from .errors import (
    BridgeError,
    ChannelError,
    CommandExecutionError,
    CommandNotFoundError,
    InternalError,
    MessageHandlerNotFoundError,
    StaticHandlerNotFoundError,
    UploadHandlerNotFoundError,
    UploadValidationError,
    ValidationError,
    WebSocketError,
)
from .generator import generate_typescript
from .registry import CommandInfo, CommandRegistry, command, get_registry, message
from .runner import run
from .static import StaticFile, StaticInfo, static
from .upload import UploadFile, UploadInfo, upload
from .websocket import WebSocket, MessageHandlerInfo

__all__ = [
    # Version
    "__version__",
    # Core
    "Bridge",
    "command",
    "message",
    "upload",
    "static",
    "run",
    # Streaming
    "Channel",
    "ChannelManager",
    "channel_manager",
    # WebSocket
    "WebSocket",
    "MessageHandlerInfo",
    # Upload
    "UploadFile",
    "UploadInfo",
    # Static
    "StaticFile",
    "StaticInfo",
    # Registry
    "get_registry",
    "CommandRegistry",
    "CommandInfo",
    # TypeScript generation
    "generate_typescript",
    # Errors
    "BridgeError",
    "ValidationError",
    "CommandNotFoundError",
    "CommandExecutionError",
    "ChannelError",
    "InternalError",
    "WebSocketError",
    "MessageHandlerNotFoundError",
    "UploadHandlerNotFoundError",
    "UploadValidationError",
    "StaticHandlerNotFoundError",
]
