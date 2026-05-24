from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

_BINARY_NAMES = ("zynk.exe", "zynk") if os.name == "nt" else ("zynk", "zynk.exe")


def _bundled_binary_path() -> Path | None:
    bin_dir = Path(__file__).resolve().parent / "bin"
    for name in _BINARY_NAMES:
        candidate = bin_dir / name
        if candidate.is_file():
            return candidate
    return None


def _platform_label() -> str:
    machine = platform.machine() or "unknown-arch"
    return f"{sys.platform}-{machine}"


def _find_system_binary() -> Path | None:
    path_dirs = os.get_exec_path()
    for d in path_dirs:
        for name in _BINARY_NAMES:
            candidate = Path(d) / name
            if not candidate.is_file():
                continue
            # Skip Python entry-point shims (text scripts); we want a real binary.
            try:
                with open(candidate, "rb") as f:
                    if f.read(2) == b"#!":
                        continue
            except OSError:
                continue
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> NoReturn:
    binary = _bundled_binary_path() or _find_system_binary()
    if binary is None:
        print(
            "No prebuilt zynk binary for "
            f"{_platform_label()}. Install the standalone CLI with "
            "`cargo install zynk-cli`, or download a zynk release artifact "
            "for your platform.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    args = list(sys.argv[1:] if argv is None else argv)
    os.execv(str(binary), [str(binary), *args])
    raise SystemExit(1)
