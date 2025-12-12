"""
Kitchen Sink Example - Main Entry Point

Demonstrates Zynk setup with multiple command modules.
"""

# Import modules to register their commands (side-effect imports)
import tasks
import users
import weather

from zynk import Bridge

# Create the bridge with TypeScript generation configured
app = Bridge(
    generate_ts="../frontend/src/generated/api.ts",
    host="127.0.0.1",
    port=8000,
    title="Kitchen Sink API",
    debug=False,
)

if __name__ == "__main__":
    # dev=True enables:
    # - Uvicorn hot-reloading
    # - TypeScript regeneration on file changes
    app.run(dev=True)
