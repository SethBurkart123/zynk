"""Run the Zynk bridge for the Effect connector demo."""

from __future__ import annotations

import commands  # noqa: F401  (registers commands)

import zynk.generators.effect  # noqa: F401  (registers the Effect backend)
from zynk import Bridge

app = Bridge(
    generate_ts=None,  # Use generate.py to (re)build the typed client.
    port=8000,
)

if __name__ == "__main__":
    app.run(dev=True)
