"""Run the Zynk bridge for the Effect connector demo."""

from __future__ import annotations

import commands  # noqa: F401  (registers commands)

from zynk import Bridge

app = Bridge(port=8000)

if __name__ == "__main__":
    app.run(dev=True)
