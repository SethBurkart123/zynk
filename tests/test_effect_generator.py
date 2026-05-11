"""Tests for the Effect TypeScript connector."""

from __future__ import annotations

import os
import tempfile

import pytest
from pydantic import BaseModel

# Importing the package registers the generator with Zynk's plugin registry.
import zynk.generators.effect  # noqa: F401
from zynk.channel import Channel
from zynk.codegen import generate_client, list_generators
from zynk.registry import CommandRegistry, command
from zynk.static import StaticFile, static
from zynk.upload import UploadFile, upload
from zynk.websocket import WebSocket


class _User(BaseModel):
    id: int
    name: str
    is_admin: bool = False


class _Address(BaseModel):
    street: str
    city: str


class _Person(BaseModel):
    name: str
    address: _Address | None = None
    tags: list[str] = []


class _UploadResult(BaseModel):
    url: str
    size: int


class _ChatMessage(BaseModel):
    text: str
    sender: str


class _ServerEvents:
    chat_message: _ChatMessage
    status: str


class _ClientEvents:
    chat_message: _ChatMessage


@pytest.fixture(autouse=True)
def _reset_registry():
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _generate(tmp: str) -> tuple[str, str]:
    out = os.path.join(tmp, "api.ts")
    generate_client(out, language="effect")
    runtime = os.path.join(tmp, "_effect_internal.ts")
    return out, runtime


def test_generator_registered():
    assert "effect" in list_generators()


def test_emits_two_files(temp_dir):
    @command
    async def ping() -> str:
        return "pong"

    client_path, runtime_path = _generate(temp_dir)
    assert os.path.exists(client_path)
    assert os.path.exists(runtime_path)


def test_runtime_contains_core_helpers(temp_dir):
    @command
    async def ping() -> str:
        return "pong"

    _, runtime_path = _generate(temp_dir)
    runtime = open(runtime_path).read()

    for symbol in (
        "class ZynkClient",
        "callCommand",
        "callChannel",
        "callUpload",
        "buildStaticUrl",
        "ZynkNetworkError",
        "ZynkTimeoutError",
        "ZynkHttpError",
        "ZynkDecodeError",
        "RetryOptions",
        "BackoffStrategy",
        "buildSchedule",
        "Schedule.exponential",
        "Schedule.jittered",
    ):
        assert symbol in runtime, f"runtime missing {symbol!r}"


def test_command_returns_effect_with_schema(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "export const _User = Schema.Struct({" in content
    assert 'id: Schema.Number' in content
    assert "export type _User = Schema.Schema.Type<typeof _User>" in content
    assert (
        "export const getUser = (args: { userId: number }, options?: CallOptions)"
        in content
    )
    assert "Effect.Effect<_User, ZynkError, ZynkClient>" in content
    assert 'callCommand("get_user", { user_id: args.userId }, _User, options)' in content


def test_optional_param_becomes_question_mark(temp_dir):
    @command
    async def list_users(active_only: bool = True) -> list[_User]:
        return []

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()
    assert "args: { activeOnly?: boolean }" in content
    assert "ReadonlyArray<_User>" in content


def test_optional_return_yields_undefined_or(temp_dir):
    @command
    async def find_person(name: str) -> _Person | None:
        return None

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()
    assert "_Person | undefined" in content
    assert "Schema.UndefinedOr(_Person)" in content


def test_no_args_command(temp_dir):
    @command
    async def get_version() -> str:
        return "1.0.0"

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()
    assert "export const getVersion = (options?: CallOptions)" in content


def test_channel_returns_stream(temp_dir):
    @command
    async def stream_users(channel: Channel[_User]) -> None:
        pass

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "Stream.Stream<_User, ZynkError, ZynkClient>" in content
    assert 'callChannel("stream_users"' in content


def test_upload_signature(temp_dir):
    @upload
    async def upload_file(file: UploadFile, label: str = "") -> _UploadResult:
        return _UploadResult(url="x", size=0)

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "file: File" in content
    assert "options?: UploadOptions" in content
    assert "Effect.Effect<_UploadResult, ZynkError, ZynkClient>" in content
    assert 'callUpload("upload_file", [args.file]' in content


def test_static_url_helper(temp_dir):
    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "export const avatarUrl" in content
    assert "Effect.Effect<string, never, ZynkClient>" in content
    assert 'buildStaticUrl("avatar", { user_id: args.userId })' in content


def test_websocket_generates_typed_socket(temp_dir):
    from zynk.registry import message

    @message
    async def chat(ws: WebSocket[_ServerEvents, _ClientEvents]) -> None:
        await ws.listen()

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "ChatServerEvents" in content
    assert "ChatClientEvents" in content
    assert "interface ChatSocket" in content
    assert "openWebSocket(\"chat\")" in content
    assert "Stream.async" in content


def test_retry_options_exported_through_runtime(temp_dir):
    @command
    async def ping() -> str:
        return "x"

    client_path, runtime_path = _generate(temp_dir)
    runtime = open(runtime_path).read()
    client = open(client_path).read()

    # Runtime provides the types and defaults.
    assert "DEFAULT_RETRY" in runtime
    assert "kind: \"exponential\"" in runtime
    assert "Schedule.exponential" in runtime
    assert "Schedule.jittered" in runtime
    assert "Schedule.recurs" in runtime
    assert "mergeRetry" in runtime
    # Per-call CallOptions/RetryOptions/BackoffStrategy are re-exported.
    assert "type RetryOptions" in client
    assert "type BackoffStrategy" in client
    assert "type CallOptions" in client


def test_global_defaults_in_init(temp_dir):
    @command
    async def ping() -> str:
        return "x"

    _, runtime_path = _generate(temp_dir)
    runtime = open(runtime_path).read()

    # ZynkClientConfig accepts global timeout/retry/headers.
    assert "interface ZynkClientConfig" in runtime
    assert "readonly timeout?: DurationInput" in runtime
    assert "readonly retry?: RetryOptions" in runtime
    assert "readonly headers?: Readonly<Record<string, string>>" in runtime
    assert "initZynk" in runtime


def test_nested_models_are_emitted(temp_dir):
    @command
    async def get_person(name: str) -> _Person:
        return _Person(name=name)

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()

    assert "export const _Address = Schema.Struct" in content
    assert "export const _Person = Schema.Struct" in content
    # Nested optional model resolves to UndefinedOr(_Address)
    assert "Schema.UndefinedOr(_Address)" in content
    assert "Schema.Array(Schema.String)" in content


def test_call_options_are_forwarded(temp_dir):
    @command
    async def echo(value: str) -> str:
        return value

    client_path, _ = _generate(temp_dir)
    content = open(client_path).read()
    # The last argument of every callCommand emission is the user's options.
    assert "options" in content
    assert "callCommand(\"echo\", { value: args.value }, Schema.String, options)" in content
