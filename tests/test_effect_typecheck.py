"""Type-check the emitted Effect runtime against a real ``effect`` install.

This catches mistakes where the generated TypeScript uses APIs that don't
exist in the runtime ``effect`` library (eg. wrong helper names, wrong
parameter shapes). Skips if a usable ``tsc`` + ``effect`` install can't be
found alongside the repo (typical in CI without a JS toolchain).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

import zynk.generators.effect  # noqa: F401
from zynk.channel import Channel
from zynk.codegen import generate_client
from zynk.registry import CommandRegistry, command
from zynk.static import StaticFile, static
from zynk.upload import UploadFile, upload


REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_effect_install() -> tuple[Path, Path] | None:
    """Locate a project that has both ``effect`` and a ``tsc`` binary.

    Searches a few likely siblings of the zynk checkout. Returns
    ``(node_modules_dir, tsc_binary)`` or None if nothing usable found.
    """
    candidates = [
        REPO_ROOT,
        REPO_ROOT.parent / "agno-app-electron",
    ]
    for root in candidates:
        node_modules = root / "node_modules"
        tsc = node_modules / ".bin" / "tsc"
        effect_pkg = node_modules / "effect" / "package.json"
        if tsc.exists() and effect_pkg.exists():
            return node_modules, tsc
    return None


_INSTALL = _find_effect_install()

pytestmark = pytest.mark.skipif(
    _INSTALL is None,
    reason="no neighbouring node_modules with effect+tsc found",
)


class _User(BaseModel):
    id: int
    name: str


class _UploadResult(BaseModel):
    url: str
    size: int


class _ZLeaf(BaseModel):
    """Alphabetically last but referenced by ``_AContainer`` below."""

    id: int


class _AContainer(BaseModel):
    """Alphabetically first; depends on ``_ZLeaf`` which sorts after it."""

    leaf: _ZLeaf
    items: list[_ZLeaf]


class _MNested(BaseModel):
    """Middle alphabetically; depends on both leaf and container."""

    a: _AContainer
    z: _ZLeaf


class _Tree(BaseModel):
    """Self-recursive; needs ``Schema.suspend`` wrapping."""

    label: str
    children: list["_Tree"] | None = None


_Tree.model_rebuild()


@pytest.fixture(autouse=True)
def _reset_registry():
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


def _typecheck(workdir: Path) -> subprocess.CompletedProcess:
    assert _INSTALL is not None
    node_modules, tsc = _INSTALL

    # Symlink node_modules so the generated files resolve `effect`.
    (workdir / "node_modules").symlink_to(node_modules)

    (workdir / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "target": "ES2022",
                    "module": "ESNext",
                    "moduleResolution": "Bundler",
                    "strict": True,
                    "noEmit": True,
                    "skipLibCheck": True,
                    "lib": ["ES2022", "DOM", "DOM.Iterable"],
                    "types": [],
                },
                "include": ["*.ts"],
            }
        )
    )
    return subprocess.run(
        [str(tsc), "--project", str(workdir / "tsconfig.json")],
        capture_output=True,
        text=True,
        cwd=workdir,
    )


def _generate(workdir: Path, **options) -> None:
    out = workdir / "api.ts"
    generate_client(str(out), language="effect", options=options)


def _assert_compiles(workdir: Path) -> None:
    result = _typecheck(workdir)
    if result.returncode != 0:
        msg = "tsc failed:\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr
        pytest.fail(msg)


def test_effect_only_default_typechecks(tmp_path):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="x")

    @command
    async def stream_users(channel: Channel[_User]) -> None:
        pass

    @upload
    async def upload_file(file: UploadFile) -> _UploadResult:
        return _UploadResult(url="x", size=0)

    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    _generate(tmp_path)
    _assert_compiles(tmp_path)


def test_all_promise_typechecks(tmp_path):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="x")

    @upload
    async def upload_file(file: UploadFile) -> _UploadResult:
        return _UploadResult(url="x", size=0)

    @static
    async def avatar(user_id: str) -> StaticFile:
        return StaticFile(path="x")

    _generate(tmp_path, default="promise")
    _assert_compiles(tmp_path)


def test_mixed_promise_commands_effect_channels_typechecks(tmp_path):
    @command
    async def get_user(user_id: int) -> _User:
        return _User(id=user_id, name="x")

    @command
    async def stream_users(channel: Channel[_User]) -> None:
        pass

    _generate(tmp_path, default="effect", commands="promise")
    _assert_compiles(tmp_path)


def test_promise_channels_typechecks(tmp_path):
    @command
    async def stream_users(channel: Channel[_User]) -> None:
        pass

    _generate(tmp_path, channels="promise")
    _assert_compiles(tmp_path)


def test_schemas_emitted_in_dependency_order(tmp_path):
    """``Schema.Struct`` references other schemas as values, so a schema must
    appear after every schema it depends on. Models are intentionally named so
    that alphabetical order would break (``_AContainer`` references ``_ZLeaf``).
    """

    @command
    async def fetch_nested() -> _MNested:
        return _MNested(a=_AContainer(leaf=_ZLeaf(id=0), items=[]), z=_ZLeaf(id=0))

    _generate(tmp_path)
    text = (tmp_path / "api.ts").read_text()

    def position(name: str) -> int:
        token = f"export const {name} = Schema.Struct"
        idx = text.find(token)
        assert idx != -1, f"missing schema declaration for {name}"
        return idx

    assert position("_ZLeaf") < position("_AContainer")
    assert position("_AContainer") < position("_MNested")
    _assert_compiles(tmp_path)


def test_self_recursive_schema_typechecks(tmp_path):
    """A model that references itself must be wrapped in ``Schema.suspend``."""

    @command
    async def fetch_tree() -> _Tree:
        return _Tree(label="root")

    _generate(tmp_path)
    text = (tmp_path / "api.ts").read_text()
    assert "Schema.suspend" in text
    _assert_compiles(tmp_path)
