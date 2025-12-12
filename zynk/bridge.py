"""
Bridge Module

The main Bridge class that orchestrates the Zynk server.
Handles FastAPI setup, routing, hot-reloading, and TypeScript generation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .channel import Channel, channel_manager
from .errors import (
    BridgeError,
    CommandExecutionError,
    CommandNotFoundError,
    InternalError,
    ValidationError,
)
from .generator import generate_typescript
from .registry import CommandInfo, get_registry

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, debug: bool = False) -> None:
    """Configure logging for Zynk."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    rich_handler = RichHandler(
        level=level,
        console=Console(),
        show_time=False,
        show_level=True,
        show_path=debug,
        markup=True,
        rich_tracebacks=True,
    )

    logging.basicConfig(
        level=level,
        handlers=[rich_handler],
        format="%(message)s",
    )


class Bridge:
    """
    The main Zynk server class.

    Wraps FastAPI and Uvicorn to provide:
    - Automatic command routing
    - TypeScript client generation
    - Hot-reloading in development
    - Channel/streaming support

    Usage:
        from zynk import Bridge
        import users  # Side-effect import registers commands

        app = Bridge(
            generate_ts="../frontend/src/api.ts",
            host="127.0.0.1",
            port=8000
        )

        if __name__ == "__main__":
            app.run(dev=True)
    """

    def __init__(
        self,
        generate_ts: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        cors_origins: list[str] | None = None,
        title: str = "Zynk API",
        debug: bool = False,
    ):
        """
        Initialize the Bridge.

        Args:
            generate_ts: Path where TypeScript client will be generated.
                         If None, no TypeScript generation occurs.
            host: Host to bind the server to.
            port: Port to bind the server to.
            cors_origins: List of allowed CORS origins. Defaults to ["*"].
            title: API title for documentation.
            debug: Enable debug logging.
        """
        self.generate_ts = generate_ts
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self.title = title
        self.debug = debug
        self._ts_generated = False

        setup_logging(logging.DEBUG if debug else logging.INFO, debug=debug)

        self.app = FastAPI(title=title)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()
        self._setup_error_handlers()

    def _setup_routes(self) -> None:
        """Setup the API routes."""

        @self.app.get("/")
        async def root():
            """Health check endpoint."""
            registry = get_registry()
            commands = registry.get_all_commands()
            return {
                "status": "ok",
                "bridge": self.title,
                "commands": list(commands.keys()),
            }

        @self.app.get("/commands")
        async def list_commands():
            """List all registered commands."""
            registry = get_registry()
            commands = registry.get_all_commands()
            return {
                "commands": [
                    {
                        "name": cmd.name,
                        "module": cmd.module,
                        "has_channel": cmd.has_channel,
                        "params": list(cmd.params.keys()),
                    }
                    for cmd in commands.values()
                ]
            }

        @self.app.post("/command/{command_name}")
        async def execute_command(command_name: str, request: Request):
            """Execute a command."""
            registry = get_registry()
            cmd = registry.get_command(command_name)

            if not cmd:
                raise CommandNotFoundError(command_name)

            # Parse request body
            try:
                body = await request.json() if await request.body() else {}
            except Exception as e:
                raise ValidationError(f"Invalid JSON body: {e}")

            # Execute command
            result = await self._execute_command(cmd, body)

            return {"result": result}

        @self.app.post("/channel/{command_name}")
        async def init_channel(command_name: str, request: Request):
            """Initialize a channel for streaming."""
            registry = get_registry()
            cmd = registry.get_command(command_name)

            if not cmd:
                raise CommandNotFoundError(command_name)

            if not cmd.has_channel:
                raise ValidationError(
                    f"Command '{command_name}' does not support channels"
                )

            try:
                body = await request.json() if await request.body() else {}
            except Exception as e:
                raise ValidationError(f"Invalid JSON body: {e}")

            channel = await channel_manager.create()

            asyncio.create_task(
                self._execute_channel_command(cmd, body, channel)
            )

            return {"channelId": channel.id}

        @self.app.get("/channel/stream/{channel_id}")
        async def stream_channel(channel_id: str):
            """Stream channel messages via SSE."""
            channel = await channel_manager.get(channel_id)

            if not channel:
                raise HTTPException(
                    status_code=404,
                    detail=f"Channel '{channel_id}' not found"
                )

            async def event_generator():
                try:
                    async for message in channel:
                        yield message.to_sse()
                        if message.event in ("close", "error"):
                            break
                finally:
                    await channel_manager.remove(channel_id)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    def _setup_error_handlers(self) -> None:
        """Setup exception handlers."""

        @self.app.exception_handler(BridgeError)
        async def bridge_error_handler(request: Request, exc: BridgeError):
            status_code = 400
            if isinstance(exc, CommandNotFoundError):
                status_code = 404
            elif isinstance(exc, InternalError):
                status_code = 500

            return JSONResponse(
                status_code=status_code,
                content=exc.to_dict(),
            )

        @self.app.exception_handler(PydanticValidationError)
        async def validation_error_handler(request: Request, exc: PydanticValidationError):
            return JSONResponse(
                status_code=400,
                content={
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                },
            )

        @self.app.exception_handler(Exception)
        async def general_error_handler(request: Request, exc: Exception):
            logger.exception("Unhandled exception")
            return JSONResponse(
                status_code=500,
                content={
                    "code": "INTERNAL_ERROR",
                    "message": str(exc) if self.debug else "An internal error occurred",
                },
            )

    async def _execute_command(
        self,
        cmd: CommandInfo,
        args: dict[str, Any],
    ) -> Any:
        """
        Execute a command with the given arguments.

        Args:
            cmd: The command to execute.
            args: The arguments dictionary.

        Returns:
            The command result.

        Raises:
            ValidationError: If argument validation fails.
            CommandExecutionError: If command execution fails.
        """
        try:
            kwargs = {}
            for param_name in cmd.params:
                if param_name in args:
                    kwargs[param_name] = args[param_name]

            result = await cmd.func(**kwargs)

            if isinstance(result, BaseModel):
                return result.model_dump()
            elif isinstance(result, list):
                return [
                    item.model_dump() if isinstance(item, BaseModel) else item
                    for item in result
                ]

            return result

        except TypeError as e:
            raise ValidationError(f"Invalid arguments: {e}")
        except Exception as e:
            logger.exception(f"Command '{cmd.name}' failed")
            raise CommandExecutionError(str(e))

    async def _execute_channel_command(
        self,
        cmd: CommandInfo,
        args: dict[str, Any],
        channel: Channel,
    ) -> None:
        """
        Execute a channel command in the background.

        Args:
            cmd: The command to execute.
            args: The arguments dictionary.
            channel: The channel for streaming responses.
        """
        try:
            kwargs = {"channel": channel}
            for param_name in cmd.params:
                if param_name in args:
                    kwargs[param_name] = args[param_name]

            await cmd.func(**kwargs)
            await channel.close()

        except Exception as e:
            logger.exception(f"Channel command '{cmd.name}' failed")
            await channel.send_error(str(e))

    def generate_typescript_client(self) -> None:
        """Generate the TypeScript client if configured."""
        if self.generate_ts:
            try:
                generate_typescript(self.generate_ts)
                self._ts_generated = True
                logger.info(f"✓ TypeScript client generated: {self.generate_ts}")
            except Exception:
                logger.exception("✗ Failed to generate TypeScript client")

    def run(self, dev: bool = False) -> None:
        """
        Run the server.

        Args:
            dev: Enable development mode with hot-reloading.
        """
        import uvicorn

        self.generate_typescript_client()

        registry = get_registry()
        commands = registry.get_all_commands()

        mode = "Development" if dev else "Production"

        content = f"""Server:     http://{self.host}:{self.port}
Mode:       {mode}
Commands:   {len(commands)}"""
        if self.generate_ts:
            content += f"\nTypeScript: {self.generate_ts}"

        console = Console()
        panel = Panel.fit(content, title=self.title, border_style="blue")
        console.print("")
        console.print(panel)
        console.print("")

        if self.debug:
            logger.debug("Registered commands:")
            for cmd in commands.values():
                channel_marker = " [channel]" if cmd.has_channel else ""
                logger.debug(f"  - {cmd.name}{channel_marker} ({cmd.module})")

            command_modules = set()
            for cmd in commands.values():
                module_name = cmd.module
                if not module_name.startswith("zynk"):
                    command_modules.add(module_name)

            from . import server
            server.set_config(
                generate_ts=self.generate_ts,
                host=self.host,
                port=self.port,
                cors_origins=self.cors_origins,
                title=self.title,
                debug=self.debug,
                import_modules=list(command_modules),
            )

            uvicorn.run(
                "zynk.server:create_app",
                host=self.host,
                port=self.port,
                reload=True,
                reload_dirs=["."],
                factory=True,
                log_level="info" if not self.debug else "debug",
            )
        else:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info" if not self.debug else "debug",
            )


# Global bridge instance for factory pattern
_bridge_config: dict[str, Any] = {}


def configure_bridge(**kwargs) -> None:
    """
    Configure the bridge for factory-based startup.

    This is called in main.py before run().
    """
    global _bridge_config
    _bridge_config = kwargs


def create_app() -> FastAPI:
    """
    Factory function for creating the FastAPI app.

    Used by Uvicorn in reload mode.
    """
    global _bridge_config

    # Re-import the main module to re-register commands
    # This is handled by Uvicorn's reload mechanism

    bridge = Bridge(**_bridge_config)
    bridge.generate_typescript_client()
    return bridge.app
