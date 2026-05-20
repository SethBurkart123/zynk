"""Tests for the Python runtime cutover away from bundled codegen."""

from __future__ import annotations

import inspect
import json

import pytest

import zynk
from zynk import Bridge, server
from zynk.registry import CommandRegistry, command
from zynk.runner import run


@pytest.fixture(autouse=True)
def reset_registry():
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


def test_bridge_rejects_generate_ts_keyword() -> None:
    with pytest.raises(TypeError):
        Bridge(generate_ts=True)


@pytest.mark.parametrize("target", [Bridge, Bridge()])
def test_bridge_generate_typescript_client_attribute_removed(target: object) -> None:
    with pytest.raises(AttributeError):
        getattr(target, "generate_typescript_client")


def test_public_codegen_exports_removed() -> None:
    removed = {
        "CodeGenerator",
        "GeneratedFile",
        "GenerationContext",
        "GenerationResult",
        "generate_client",
        "generate_typescript",
        "list_generators",
        "register_generator",
    }

    assert removed.isdisjoint(zynk.__all__)
    for name in removed:
        assert not hasattr(zynk, name)


def test_dump_schema_json_survives_cutover() -> None:
    @command
    async def ping(value: str) -> str:
        return value

    payload = Bridge().dump_schema_json()
    graph = json.loads(payload)

    assert isinstance(payload, str)
    assert "ping" in graph["endpoints"]
    assert graph["endpoints"]["ping"]["kind"] == "rpc"


def test_reload_and_runner_surfaces_do_not_accept_generate_ts() -> None:
    assert "generate_ts" not in inspect.signature(Bridge.__init__).parameters
    assert "generate_ts" not in inspect.signature(Bridge.run).parameters
    assert "generate_ts" not in inspect.signature(server.set_config).parameters
    assert "generate_ts" not in inspect.signature(server.create_app).parameters
    assert "generate_ts" not in inspect.signature(run).parameters

    with pytest.raises(TypeError):
        server.set_config(generate_ts="api.ts")
    with pytest.raises(TypeError):
        run(generate_ts="api.ts")


def test_server_config_payload_does_not_include_generate_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZYNK_SERVER_CONFIG", raising=False)
    assert "generate_ts" not in server.get_config()

    server.set_config(host="0.0.0.0", port=9999)
    assert "generate_ts" not in server.get_config()
