"""Regenerate the Effect TypeScript client."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import commands  # noqa: F401  (registers the @command decorators)

# Importing this module registers the "effect" generator with Zynk.
import zynk.generators.effect  # noqa: F401
from zynk.codegen import generate_client


def main() -> None:
    output = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frontend",
        "src",
        "api.ts",
    )
    output = os.path.normpath(output)
    generate_client(output, language="effect")
    print(f"Generated Effect client: {output}")


if __name__ == "__main__":
    main()
