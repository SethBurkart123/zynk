"""
Runner Module

Provides the run() function for easy server startup with hot-reload support.
Handles Uvicorn configuration and process management.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

logger = logging.getLogger(__name__)


def run(
    generate_ts: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    cors_origins: list[str] | None = None,
    title: str = "Zynk API",
    debug: bool = False,
    dev: bool = False,
    reload_dirs: list[str] | None = None,
    import_modules: list[str] | None = None,
) -> None:
    """
    Run the Zynk server.

    This is the main entry point for starting a Zynk application.
    It handles both production and development modes.

    Args:
        generate_ts: Path where TypeScript client will be generated.
        host: Host to bind the server to.
        port: Port to bind the server to.
        cors_origins: List of allowed CORS origins.
        title: API title for documentation.
        debug: Enable debug logging.
        dev: Enable development mode with hot-reloading.
        reload_dirs: Directories to watch for changes (dev mode only).
        import_modules: List of module names containing commands to import.

    Example:
        from zynk import run

        if __name__ == "__main__":
            run(
                generate_ts="../frontend/src/api.ts",
                dev=True,
                import_modules=["users", "weather"],
            )
    """
    import uvicorn

    from .bridge import Bridge
    from .generator import generate_typescript
    from .registry import get_registry
    from .server import set_config

    # Set configuration for factory mode
    set_config(
        generate_ts=generate_ts,
        host=host,
        port=port,
        cors_origins=cors_origins,
        title=title,
        debug=debug,
        import_modules=import_modules or [],
    )

    log_level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    rich_handler = RichHandler(
        level=log_level,
        console=Console(),
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    logging.basicConfig(
        level=log_level,
        handlers=[rich_handler],
        format="%(message)s",
    )

    if generate_ts:
        try:
            generate_typescript(generate_ts)
            logger.info(f"TypeScript client generated: {generate_ts}")
        except Exception as e:
            logger.error(f"Failed to generate TypeScript client: {e}")

    registry = get_registry()
    commands = registry.get_all_commands()

    content = f"""Server:     http://{host}:{port}
Mode:       {'Development' if dev else 'Production'}
Commands:   {len(commands)}"""
    if generate_ts:
        content += f"\nTypeScript: {generate_ts}"

    console = Console()
    panel = Panel.fit(content, title=f"Zynk - {title}", border_style="blue")
    console.print(panel)
    console.print("")

    for cmd in commands.values():
        channel_marker = " [channel]" if cmd.has_channel else ""
        console.print(f"  • {cmd.name}{channel_marker}")
    console.print()

    if dev:
        logger.info("Starting in development mode with hot-reload...")

        watch_dirs = reload_dirs or ["."]

        uvicorn.run(
            "zynk.server:create_app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=watch_dirs,
            factory=True,
            log_level="debug" if debug else "info",
        )
    else:
        bridge = Bridge(
            generate_ts=generate_ts,
            host=host,
            port=port,
            cors_origins=cors_origins,
            title=title,
            debug=debug,
        )

        uvicorn.run(
            bridge.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info",
        )
