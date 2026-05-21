"""Regenerate the Effect TypeScript client via the zynk CLI."""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import commands  # noqa: F401  (registers the @command decorators)


def main() -> None:
    output = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frontend",
        "src",
        "api.ts",
    )
    output = os.path.normpath(output)
    subprocess.run(
        [
            "zynk",
            "gen",
            "effect",
            "--target",
            "python",
            "--out",
            output,
            "--app",
            "main:app",
        ],
        cwd=os.path.dirname(__file__),
        check=True,
    )
    print(f"Generated Effect client: {output}")


if __name__ == "__main__":
    main()
