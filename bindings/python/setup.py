from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup  # type: ignore[import-untyped]
from setuptools.command.build_py import (
    build_py as _build_py,  # type: ignore[import-untyped]
)
from wheel.bdist_wheel import (
    bdist_wheel as _bdist_wheel,  # type: ignore[import-not-found]
)


def _bundle_cli_enabled() -> bool:
    return os.environ.get("ZYNK_BUNDLE_CLI") in {"1", "true", "TRUE", "yes", "YES"}


def _bundle_cli(destination: Path) -> None:
    script = Path(__file__).resolve().parent / "scripts" / "bundle_cli.py"
    env = {**os.environ, "ZYNK_CLI_DESTINATION": str(destination)}
    subprocess.run([sys.executable, str(script)], check=True, env=env)


class BuildPy(_build_py):  # type: ignore[misc]
    def run(self) -> None:
        if _bundle_cli_enabled():
            package_dir = Path(self.build_lib) / "zynk" / "bin"
            _bundle_cli(package_dir)
        super().run()


class BdistWheel(_bdist_wheel):  # type: ignore[misc]
    def finalize_options(self) -> None:
        super().finalize_options()
        if _bundle_cli_enabled():
            self.root_is_pure = False


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
