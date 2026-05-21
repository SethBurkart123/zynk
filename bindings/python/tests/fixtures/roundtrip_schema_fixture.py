from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from zynk import (
    Bridge,
    Channel,
    StaticFile,
    UploadFile,
    WebSocket,
    command,
    message,
    static,
    upload,
)
from zynk.registry import CommandRegistry

CommandRegistry.reset()


class RoundtripPriority(str, Enum):
    LOW = "low"
    HIGH = "high"


class RoundtripMetadata(BaseModel):
    tags: list[str]
    attributes: dict[str, str] = {}


class RoundtripProfile(BaseModel):
    id: int
    metadata: RoundtripMetadata
    priority: RoundtripPriority
    display_name: str = "Anonymous"
    nickname: str | None = None
    status: Literal["active"] = "active"


class RoundtripServerEvents:
    profile_updated: RoundtripProfile
    heartbeat: Literal["pong"]


class RoundtripClientEvents:
    subscribe: RoundtripPriority


@command
async def update_profile(
    profile: RoundtripProfile,
    include_deleted: bool = False,
) -> RoundtripProfile:
    return profile


@command
async def stream_profiles(
    query: str,
    channel: Channel[RoundtripProfile],
) -> None:
    await channel.close()


@upload(max_size="5MB", allowed_types=["image/*"])
async def upload_avatar(
    file: UploadFile,
    profile_id: int,
) -> RoundtripProfile:
    return RoundtripProfile(
        id=profile_id,
        metadata=RoundtripMetadata(tags=[]),
        priority=RoundtripPriority.LOW,
    )


@static
async def avatar_file(profile_id: int, size: str = "small") -> StaticFile:
    return StaticFile(Path(__file__))


@message
async def profile_socket(
    ws: WebSocket[RoundtripServerEvents, RoundtripClientEvents],
) -> None:
    await ws.close()


bridge = Bridge()


if __name__ == "__main__":
    print(bridge.dump_schema_json())
