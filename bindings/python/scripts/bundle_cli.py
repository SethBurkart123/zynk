from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "Cargo.toml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find repo root (no Cargo.toml in any parent directory)"
    )


def _binary_name() -> str:
    return "zynk.exe" if sys.platform == "win32" else "zynk"


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def main() -> None:
    repo_root = _repo_root()
    binary = repo_root / "target" / "release" / _binary_name()
    explicit = _truthy(os.environ.get("ZYNK_BUNDLE_CLI"))
    skip = _truthy(os.environ.get("ZYNK_SKIP_BUNDLE_CLI"))

    if skip:
        print("ZYNK_SKIP_BUNDLE_CLI is set; building a pure Python wheel")
        return

    # Build the CLI when the caller asks for it, or when we have no local binary
    # to copy yet. If a pre-built binary already exists we reuse it so iterative
    # `uv sync` cycles don't pay the cargo-build cost.
    if explicit or not binary.is_file():
        try:
            subprocess.run(
                ["cargo", "build", "--release", "-p", "zynk-cli"],
                cwd=repo_root,
                check=True,
            )
        except FileNotFoundError:
            if explicit:
                raise
            print("cargo not found; building a pure Python wheel")
            return
        except subprocess.CalledProcessError:
            if explicit:
                raise
            print("cargo build failed; building a pure Python wheel")
            return

    if not binary.is_file():
        if explicit:
            raise FileNotFoundError(f"expected zynk CLI binary at {binary}")
        print(f"no zynk CLI binary at {binary}; building a pure Python wheel")
        return

    configured_destination = os.environ.get("ZYNK_CLI_DESTINATION")
    destination_dir = (
        Path(configured_destination)
        if configured_destination is not None
        else repo_root / "bindings" / "python" / "zynk" / "bin"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / _binary_name()
    shutil.copy2(binary, destination)
    if sys.platform != "win32":
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Bundled zynk CLI binary at {destination}")


if __name__ == "__main__":
    main()
