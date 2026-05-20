from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _binary_name() -> str:
    return "zynk.exe" if sys.platform == "win32" else "zynk"


def main() -> None:
    if os.environ.get("ZYNK_BUNDLE_CLI") not in {"1", "true", "TRUE", "yes", "YES"}:
        print("ZYNK_BUNDLE_CLI is not enabled; building a pure Python wheel")
        return

    repo_root = _repo_root()
    subprocess.run(
        ["cargo", "build", "--release", "-p", "zynk-cli"],
        cwd=repo_root,
        check=True,
    )

    binary = repo_root / "target" / "release" / _binary_name()
    if not binary.is_file():
        raise FileNotFoundError(f"expected zynk CLI binary at {binary}")

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
