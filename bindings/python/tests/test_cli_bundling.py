from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import tomllib  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_PATH = REPO_ROOT / "bindings" / "python" / "pyproject.toml"
CLI_CARGO_PATH = REPO_ROOT / "crates" / "zynk-cli" / "Cargo.toml"


def test_cli_shim_execs_bundled_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zynk import _cli

    binary_name = "zynk.exe" if os.name == "nt" else "zynk"
    bundled = tmp_path / binary_name
    bundled.write_text("#!/bin/sh\necho zynk 0.1.0\n", encoding="utf-8")
    bundled.chmod(bundled.stat().st_mode | stat.S_IXUSR)

    exec_calls: list[tuple[str, list[str]]] = []

    def fake_execv(path: str, args: list[str]) -> NoReturn:
        exec_calls.append((path, args))
        raise SystemExit(42)

    monkeypatch.setattr(_cli, "_bundled_binary_path", lambda: bundled)
    monkeypatch.setattr(os, "execv", fake_execv)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(["--version"])

    assert exc_info.value.code == 42
    assert exec_calls == [(str(bundled), [str(bundled), "--version"])]


def test_cli_shim_prints_actionable_guidance_without_bundled_binary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from zynk import _cli

    monkeypatch.setattr(_cli, "_bundled_binary_path", lambda: None)
    monkeypatch.setattr(_cli, "_find_system_binary", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(["gen", "typescript"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "No prebuilt zynk binary" in captured.err
    assert sys.platform in captured.err
    assert "cargo install zynk-cli" in captured.err


def test_cli_shim_falls_back_to_system_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zynk import _cli

    binary_name = "zynk.exe" if os.name == "nt" else "zynk"
    system_bin = tmp_path / binary_name
    # Write a real binary header (not a #! shim) so the fallback picks it up.
    system_bin.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 100)
    system_bin.chmod(system_bin.stat().st_mode | stat.S_IXUSR)

    exec_calls: list[tuple[str, list[str]]] = []

    def fake_execv(path: str, args: list[str]) -> NoReturn:
        exec_calls.append((path, args))
        raise SystemExit(42)

    monkeypatch.setattr(_cli, "_bundled_binary_path", lambda: None)
    monkeypatch.setattr(_cli, "_find_system_binary", lambda: system_bin)
    monkeypatch.setattr(os, "execv", fake_execv)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(["--version"])

    assert exc_info.value.code == 42
    assert exec_calls == [(str(system_bin), [str(system_bin), "--version"])]


def test_pyproject_aligns_version_entrypoint_and_cibuildwheel_targets() -> None:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    cli_cargo = tomllib.loads(CLI_CARGO_PATH.read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == cli_cargo["package"]["version"]
    assert pyproject["project"]["scripts"]["zynk"] == "zynk._cli:main"

    package_data = pyproject["tool"]["setuptools"]["package-data"]["zynk"]
    assert "bin/zynk" in package_data
    assert "bin/zynk.exe" in package_data

    cibuildwheel = pyproject["tool"]["cibuildwheel"]
    build_targets = set(cibuildwheel["build"].split())
    assert {
        "cp*-macosx_arm64",
        "cp*-manylinux_x86_64",
        "cp*-manylinux_aarch64",
        "cp*-win_amd64",
    }.issubset(build_targets)
    assert "cp*-macosx_x86_64" not in build_targets
    assert cibuildwheel["environment"]["ZYNK_BUNDLE_CLI"] == "1"
    assert "ZYNK_BUNDLE_CLI" in cibuildwheel["before-build"]
