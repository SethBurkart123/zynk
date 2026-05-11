"""Demo commands for the Effect connector example."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from zynk import Channel, command


class User(BaseModel):
    id: int
    name: str
    is_admin: bool = False


class Tick(BaseModel):
    n: int
    label: str


@command
async def get_user(user_id: int) -> User:
    """Return a single user by id."""
    return User(id=user_id, name=f"user-{user_id}")


@command
async def list_users(active_only: bool = True) -> list[User]:
    """List all users."""
    return [
        User(id=1, name="alice", is_admin=True),
        User(id=2, name="bob"),
    ]


@command
async def stream_ticks(label: str, channel: Channel[Tick]) -> None:
    """Stream ten ticks to demonstrate Effect Streams."""
    for n in range(10):
        await channel.send(Tick(n=n, label=label))
        await asyncio.sleep(0.2)
