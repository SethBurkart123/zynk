"""Tests for Bridge.dump_schema_json()."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel

from zynk import Bridge, Channel, WebSocket, command, message
from zynk.registry import CommandRegistry


class SchemaPriority(str, Enum):
    LOW = "low"
    HIGH = "high"


class SchemaProfile(BaseModel):
    id: int
    display_name: str = "Anonymous"
    nickname: str | None = None
    status: Literal["active"] = "active"
    priority: SchemaPriority


class SchemaServerEvents:
    profile_updated: SchemaProfile
    heartbeat: Literal["pong"]


class SchemaClientEvents:
    subscribe: SchemaPriority


@pytest.fixture(autouse=True)
def reset_registry():
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


def register_schema_fixture() -> None:
    @command
    async def update_profile(profile: SchemaProfile) -> SchemaProfile:
        return profile

    @command
    async def stream_profiles(query: str, channel: Channel[SchemaProfile]) -> None:
        await channel.close()

    @message
    async def profile_socket(ws: WebSocket[SchemaServerEvents, SchemaClientEvents]) -> None:
        await ws.close()


def test_dump_schema_json_method_exists_returns_str_and_valid_json() -> None:
    register_schema_fixture()
    bridge = Bridge()

    dumped = bridge.dump_schema_json()

    assert isinstance(dumped, str)
    parsed = json.loads(dumped)
    assert set(parsed) == {"endpoints", "enums", "models"}
    assert "update_profile" in parsed["endpoints"]


def test_dump_schema_json_is_byte_stable_and_sorted() -> None:
    register_schema_fixture()
    bridge = Bridge()

    first = bridge.dump_schema_json()
    second = bridge.dump_schema_json()

    assert first == second
    parsed = json.loads(first)
    assert list(parsed["endpoints"].keys()) == sorted(parsed["endpoints"].keys())
    assert list(parsed["models"].keys()) == sorted(parsed["models"].keys())
    assert list(parsed["enums"].keys()) == sorted(parsed["enums"].keys())


def test_dump_schema_json_matches_canonical_rust_shape() -> None:
    register_schema_fixture()
    bridge = Bridge()

    dumped = bridge.dump_schema_json()
    parsed = json.loads(dumped)

    serialized = json.dumps(parsed)
    for forbidden in ("py_name", "ts_name", "is_async", "func", "deps", "raw", "py_type"):
        assert forbidden not in serialized

    expected = {
        "endpoints": {
            "profile_socket": {
                "clientEvents": [
                    {
                        "required": True,
                        "sourceName": "subscribe",
                        "ty": {
                            "kind": "enum",
                            "name": "SchemaPriority",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "subscribe",
                    }
                ],
                "kind": "ws",
                "module": __name__,
                "multiFile": False,
                "name": "profile_socket",
                "returns": {"kind": "void", "nullable": False, "optional": False},
                "serverEvents": [
                    {
                        "required": True,
                        "sourceName": "heartbeat",
                        "ty": {
                            "kind": "literal",
                            "nullable": False,
                            "optional": False,
                            "value": "pong",
                        },
                        "wireName": "heartbeat",
                    },
                    {
                        "required": True,
                        "sourceName": "profile_updated",
                        "ty": {
                            "kind": "model",
                            "name": "SchemaProfile",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "profileUpdated",
                    },
                ],
            },
            "stream_profiles": {
                "channelItem": {
                    "kind": "model",
                    "name": "SchemaProfile",
                    "nullable": False,
                    "optional": False,
                },
                "kind": "channel",
                "module": __name__,
                "multiFile": False,
                "name": "stream_profiles",
                "params": [
                    {
                        "required": True,
                        "sourceName": "query",
                        "ty": {
                            "kind": "primitive",
                            "name": "string",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "query",
                    }
                ],
                "returns": {"kind": "primitive", "name": "undefined", "nullable": False, "optional": False},
            },
            "update_profile": {
                "kind": "rpc",
                "module": __name__,
                "multiFile": False,
                "name": "update_profile",
                "params": [
                    {
                        "required": True,
                        "sourceName": "profile",
                        "ty": {
                            "kind": "model",
                            "name": "SchemaProfile",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "profile",
                    }
                ],
                "returns": {
                    "kind": "model",
                    "name": "SchemaProfile",
                    "nullable": False,
                    "optional": False,
                },
            },
        },
        "enums": {
            "SchemaPriority": {
                "name": "SchemaPriority",
                "values": ["low", "high"],
            }
        },
        "models": {
            "SchemaProfile": {
                "fields": [
                    {
                        "required": True,
                        "sourceName": "id",
                        "ty": {
                            "kind": "primitive",
                            "name": "number",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "id",
                        "optional": False,
                        "nullable": False,
                    },
                    {
                        "default": "Anonymous",
                        "required": False,
                        "sourceName": "display_name",
                        "ty": {
                            "kind": "primitive",
                            "name": "string",
                            "nullable": False,
                            "optional": True,
                        },
                        "wireName": "displayName",
                        "optional": True,
                        "nullable": False,
                    },
                    {
                        "default": None,
                        "required": False,
                        "sourceName": "nickname",
                        "ty": {
                            "kind": "primitive",
                            "name": "string",
                            "nullable": True,
                            "optional": True,
                        },
                        "wireName": "nickname",
                        "optional": True,
                        "nullable": True,
                    },
                    {
                        "default": "active",
                        "required": False,
                        "sourceName": "status",
                        "ty": {
                            "kind": "literal",
                            "nullable": False,
                            "optional": True,
                            "value": "active",
                        },
                        "wireName": "status",
                        "optional": True,
                        "nullable": False,
                    },
                    {
                        "required": True,
                        "sourceName": "priority",
                        "ty": {
                            "kind": "enum",
                            "name": "SchemaPriority",
                            "nullable": False,
                            "optional": False,
                        },
                        "wireName": "priority",
                        "optional": False,
                        "nullable": False,
                    },
                ],
                "name": "SchemaProfile",
            }
        },
    }
    assert parsed == expected
    assert parsed["endpoints"]["profile_socket"]["serverEvents"][0]["sourceName"] == "heartbeat"
    assert isinstance(parsed["endpoints"]["profile_socket"]["serverEvents"], list)
    assert isinstance(parsed["endpoints"]["profile_socket"]["clientEvents"], list)
