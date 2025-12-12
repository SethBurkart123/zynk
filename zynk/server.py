"""
Server Module

Provides the factory function for Uvicorn reload support.
This module is the entry point when running with --reload.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from typing import Any

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Environment variable name for passing config between processes
_CONFIG_ENV_VAR = "ZYNK_SERVER_CONFIG"


def set_config(
    generate_ts: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    cors_origins: list[str] | None = None,
    title: str = "Zynk API",
    debug: bool = False,
    main_module: str | None = None,
    import_modules: list[str] | None = None,
) -> None:
    """
    Set the server configuration via environment variable.

    Called before run() to configure the server for factory mode.
    Uses environment variables to pass config to Uvicorn's subprocess.
    """
    config = {
        "generate_ts": generate_ts,
        "host": host,
        "port": port,
        "cors_origins": cors_origins or ["*"],
        "title": title,
        "debug": debug,
        "main_module": main_module,
        "import_modules": import_modules or [],
    }
    # Store in environment variable for subprocess access
    os.environ[_CONFIG_ENV_VAR] = json.dumps(config)


def get_config() -> dict[str, Any]:
    """Get the current configuration from environment variable."""
    config_json = os.environ.get(_CONFIG_ENV_VAR)
    if config_json:
        return json.loads(config_json)
    # Default config
    return {
        "generate_ts": None,
        "host": "127.0.0.1",
        "port": 8000,
        "cors_origins": ["*"],
        "title": "Zynk API",
        "debug": False,
        "main_module": None,
        "import_modules": [],
    }


def create_app() -> FastAPI:
    """
    Factory function for creating the FastAPI app.

    This is called by Uvicorn on each reload.
    It re-imports command modules to re-register commands,
    then creates a fresh Bridge instance and generates TypeScript.
    """
    from .bridge import Bridge
    from .registry import CommandRegistry

    config = get_config()

    CommandRegistry.reset()

    import_modules = config.get("import_modules", [])
    logger.debug(f"Re-importing modules: {import_modules}")

    for module_name in import_modules:
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]

            importlib.import_module(module_name)
            logger.debug(f"Imported module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to import module '{module_name}': {e}")

    bridge = Bridge(
        generate_ts=config.get("generate_ts"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8000),
        cors_origins=config.get("cors_origins"),
        title=config.get("title", "Zynk API"),
        debug=config.get("debug", False),
    )

    bridge.generate_typescript_client()

    logger.debug("Application reloaded")

    return bridge.app
