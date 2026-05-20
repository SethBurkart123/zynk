"""
Kitchen Sink Example - Main Entry Point

Demonstrates Zynk setup with multiple command modules.
"""

# Import modules to register their commands and handlers (side-effect imports)
import chat  # noqa: F401
import tasks  # noqa: F401
import uploads  # noqa: F401
import users  # noqa: F401
import weather  # noqa: F401

from zynk import Bridge

# Create the bridge runtime. Use `zynk gen` for TypeScript client generation.
app = Bridge(
    host="127.0.0.1",
    port=8000,
    title="Kitchen Sink API",
    debug=False,
)

if __name__ == "__main__":
    # dev=True enables Uvicorn hot-reloading.
    app.run(dev=True)
