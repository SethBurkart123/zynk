"""Tests for the Effect generator promise surface options."""

from __future__ import annotations

import os
import tempfile

import pytest
from pydantic import BaseModel

import zynk.generators.effect  # noqa: F401
from zynk.channel import Channel
from zynk.codegen import generate_client
from zynk.registry import CommandRegistry, command
from zynk.static import StaticFile, static
from zynk.upload import UploadFile, upload


class _User(BaseModel):
    id: int
    name: str


class _UploadResult(BaseModel):
    url: str
    size: int


@pytest.fixture(autouse=True)
def _reset_registry():
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _generate(tmp: str, **options) -> str:
    out = os.path.join(tmp, "api.ts")
    generate_client(out, language="effect", options=options)
    return open(out).read()


def test_no_options_emits_effect(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    content = _generate(temp_dir)
    assert "Effect.Effect<_User, ZynkError, ZynkClient>" in content
    assert "Promise<_User>" not in content


def test_commands_promise(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    content = _generate(temp_dir, commands="promise")
    assert "Promise<_User>" in content
    assert "runPromise(callCommand(" in content
    assert "Effect.Effect<_User" not in content


def test_commands_promise_no_args(temp_dir):
    @command
    async def get_version() -> str:
        return "1.0.0"

    content = _generate(temp_dir, commands="promise")
    assert "Promise<string>" in content
    assert "options?: CallOptions): Promise<string>" in content


def test_default_promise_affects_all(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    content = _generate(temp_dir, default="promise")
    assert "Promise<_User>" in content
    assert "Promise<string>" in content
    assert "runPromise(" in content


def test_default_promise_override_commands_effect(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    content = _generate(temp_dir, default="promise", commands="effect")
    assert "Effect.Effect<_User, ZynkError, ZynkClient>" in content
    assert "avatarUrl" in content
    assert "Promise<string>" in content


def test_uploads_promise(temp_dir):
    @upload
    async def upload_file(file: UploadFile) -> _UploadResult:
        return _UploadResult(url="x", size=0)

    content = _generate(temp_dir, uploads="promise")
    assert "Promise<_UploadResult>" in content
    assert "runPromise(callUpload(" in content


def test_statics_promise(temp_dir):
    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    content = _generate(temp_dir, statics="promise")
    assert "Promise<string>" in content
    assert "runPromise(buildStaticUrl(" in content


def test_channels_promise(temp_dir):
    @command
    async def stream_users(channel: Channel[_User]) -> None:
        pass

    content = _generate(temp_dir, channels="promise")
    assert "AsyncIterable<_User>" in content
    assert "toAsyncIterable(callChannel(" in content


def test_runtime_has_run_promise(temp_dir):
    @command
    async def ping() -> str:
        return "x"

    _generate(temp_dir, commands="promise")
    runtime_path = os.path.join(temp_dir, "_effect_internal.ts")
    runtime = open(runtime_path).read()
    assert "export const runPromise" in runtime
    assert "setDefaultLayer" in runtime
    assert "ManagedRuntime.make" in runtime


def test_init_zynk_sets_default_layer(temp_dir):
    @command
    async def ping() -> str:
        return "x"

    _generate(temp_dir, commands="promise")
    runtime_path = os.path.join(temp_dir, "_effect_internal.ts")
    runtime = open(runtime_path).read()
    assert "setDefaultLayer(layer)" in runtime


def test_promise_imports_run_promise(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    content = _generate(temp_dir, commands="promise")
    assert "runPromise," in content


def test_effect_does_not_import_run_promise(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    content = _generate(temp_dir)
    assert "runPromise" not in content


def test_unknown_option_raises(temp_dir):
    @command
    async def ping() -> str:
        return "x"

    with pytest.raises(ValueError, match="Unknown effect generator options"):
        _generate(temp_dir, bogus="value")


def test_schemas_still_emitted_with_promise(temp_dir):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="alice")

    content = _generate(temp_dir, commands="promise")
    assert "export const _User = Schema.Struct({" in content
    assert "export type _User = Schema.Schema.Type<typeof _User>" in content


def test_call_options_forwarded_in_promise(temp_dir):
    @command
    async def echo(value: str) -> str:
        return value

    content = _generate(temp_dir, commands="promise")
    assert "options?: CallOptions): Promise<string>" in content
    assert 'callCommand("echo", { value: args.value }, Schema.String, options)' in content
