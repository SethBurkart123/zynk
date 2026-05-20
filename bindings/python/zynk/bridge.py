"""
Bridge Module

The main Bridge class that orchestrates the Zynk server.
Handles FastAPI setup, routing, hot-reloading, and TypeScript generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import types
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from fastapi import FastAPI, Request
from fastapi import WebSocket as FastAPIWebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from starlette.datastructures import UploadFile as StarletteUploadFile

from .channel import Channel
from .core.schema_json import dump_api_graph_json
from .errors import (
    BridgeError,
    CommandNotFoundError,
    InternalError,
    StaticHandlerNotFoundError,
    UploadHandlerNotFoundError,
    UploadValidationError,
    ValidationError,
)
from .generator import generate_typescript
from .registry import CommandInfo, get_registry
from .runtime.dispatch import (
    execute_channel_command as dispatch_channel_command,
)
from .runtime.dispatch import (
    execute_command as dispatch_command,
)
from .runtime.dispatch import (
    execute_static as dispatch_static,
)
from .runtime.dispatch import (
    execute_upload as dispatch_upload,
)
from .static import StaticFile, StaticInfo
from .upload import UploadFile, UploadInfo
from .websocket import WebSocket

logger = logging.getLogger(__name__)


def instantiate_model(data: Any, type_hint: Any) -> Any:
    """
    Recursively instantiate Pydantic models from dicts.

    Args:
        data: The data to convert (dict, list, or primitive).
        type_hint: The expected type hint.

    Returns:
        Instantiated Pydantic model or original data.
    """
    if data is None:
        return None

    # Get the origin and args for generic types
    origin = get_origin(type_hint) if type_hint else None
    args = get_args(type_hint) if type_hint else ()

    # Handle Union types (Optional, T | None, etc.)
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return instantiate_model(data, non_none_args[0])

    # Handle lists
    if isinstance(data, list) and origin is list and args:
        item_type = args[0]
        return [instantiate_model(item, item_type) for item in data]

    # Handle dicts that should become Pydantic models
    if isinstance(data, dict):
        if (
            type_hint
            and isinstance(type_hint, type)
            and issubclass(type_hint, BaseModel)
        ):
            # Recursively instantiate nested models first
            model_fields = type_hint.model_fields
            converted = {}
            for key, value in data.items():
                field_info = model_fields.get(key)
                if field_info:
                    converted[key] = instantiate_model(value, field_info.annotation)
                else:
                    converted[key] = value
            return type_hint(**converted)

    return data


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
        app_init: str | None = None,
        reload_dirs: list[str] | None = None,
        reload_includes: list[str] | None = None,
        reload_excludes: list[str] | None = None,
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
            reload_dirs: Directories to watch for changes (dev mode only).
                         Defaults to ["."] (current working directory).
            reload_includes: Glob patterns to include in file watching (dev mode only).
            reload_excludes: Glob patterns to exclude from file watching (dev mode only).
        """
        self.generate_ts = generate_ts
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self.title = title
        self.debug = debug
        self.app_init = app_init
        self.reload_dirs = reload_dirs
        self.reload_includes = reload_includes
        self.reload_excludes = reload_excludes
        self._ts_generated = False
        self._shutdown_callbacks: list[Callable[[], Any]] = []

        setup_logging(logging.DEBUG if debug else logging.INFO, debug=debug)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            for callback in self._shutdown_callbacks:
                try:
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"Shutdown callback error: {e}")

        self.app = FastAPI(title=title, lifespan=lifespan)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()
        self._setup_error_handlers()

    def use(self, middleware_class: type, *args: Any, **kwargs: Any) -> None:
        """Add a FastAPI/Starlette middleware to the underlying app."""
        self.app.add_middleware(middleware_class, *args, **kwargs)

    def on_shutdown(self, callback: Callable[[], Any]) -> None:
        """Register a callback to run on server shutdown."""
        self._shutdown_callbacks.append(callback)

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
        async def channel_command(command_name: str, request: Request):
            """Execute a streaming channel command, returning SSE directly."""
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

            channel = Channel()
            asyncio.create_task(self._execute_channel_command(cmd, body, channel))

            async def event_generator():
                async for message in channel:
                    yield message.to_sse()
                    if message.event in ("close", "error"):
                        break

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @self.app.websocket("/ws/{handler_name}")
        async def websocket_endpoint(websocket: FastAPIWebSocket, handler_name: str):
            """WebSocket endpoint for message handlers."""
            registry = get_registry()
            handler = registry.get_message_handler(handler_name)

            if not handler:
                await websocket.close(
                    code=4004, reason=f"Handler '{handler_name}' not found"
                )
                return

            ws = WebSocket(
                websocket=websocket,
                server_events=handler.server_events,
                client_events=handler.client_events,
            )

            try:
                await ws.accept()
                await handler.func(ws)
            except Exception as e:
                logger.exception(f"WebSocket handler '{handler_name}' failed")
                await ws.close(code=1011, reason=str(e))

        @self.app.post("/upload/{handler_name}")
        async def upload_endpoint(handler_name: str, request: Request):
            """
            Handle file uploads with XMLHttpRequest progress support.

            Files are sent as multipart/form-data with additional args in _args field.
            """
            registry = get_registry()
            upload_handler = registry.get_upload(handler_name)

            if not upload_handler:
                raise UploadHandlerNotFoundError(handler_name)

            form = await request.form()
            try:
                args = json.loads(form.get("_args", "{}"))
            except Exception as e:
                raise ValidationError(f"Invalid _args JSON: {e}")

            # Convert multipart UploadFiles to our UploadFile wrapper
            upload_files: list[UploadFile] = []
            files = [
                file
                for file in form.getlist("files")
                if isinstance(file, StarletteUploadFile)
            ]
            for f in files:
                # Read content to get size
                content = await f.read()
                await f.seek(0)  # Reset for later reading

                wrapped = UploadFile(f, size=len(content))

                # Validate file
                try:
                    upload_handler.validate_file(wrapped)
                except ValueError as e:
                    raise UploadValidationError(str(e), f.filename)

                upload_files.append(wrapped)

            # Execute the upload handler
            result = await self._execute_upload(upload_handler, upload_files, args)

            return {"result": result}

        @self.app.api_route("/static/{handler_name}", methods=["GET", "HEAD"])
        async def static_endpoint(handler_name: str, request: Request):
            """Serve static files; query params are passed to the handler."""
            registry = get_registry()
            static_handler = registry.get_static(handler_name)

            if not static_handler:
                raise StaticHandlerNotFoundError(handler_name)

            args = dict(request.query_params)
            result = await self._execute_static(static_handler, args)
            return FileResponse(
                path=result.path,
                media_type=result.guess_content_type(),
                headers={"X-Content-Type-Options": "nosniff", **result.headers},
            )

    async def _execute_static(
        self,
        handler: StaticInfo,
        args: dict[str, Any],
    ) -> StaticFile:
        return await dispatch_static(handler, args)

    async def _execute_upload(
        self,
        handler: UploadInfo,
        files: list[UploadFile],
        args: dict[str, Any],
    ) -> Any:
        return await dispatch_upload(handler, files, args)

    def _setup_error_handlers(self) -> None:
        """Setup exception handlers."""

        @self.app.exception_handler(BridgeError)
        async def bridge_error_handler(request: Request, exc: BridgeError):
            status_code = 400
            if isinstance(
                exc,
                (
                    CommandNotFoundError,
                    UploadHandlerNotFoundError,
                    StaticHandlerNotFoundError,
                ),
            ):
                status_code = 404
            elif isinstance(exc, InternalError):
                status_code = 500

            return JSONResponse(
                status_code=status_code,
                content=exc.to_dict(),
            )

        @self.app.exception_handler(PydanticValidationError)
        async def validation_error_handler(
            request: Request, exc: PydanticValidationError
        ):
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
        return await dispatch_command(cmd, args)

    async def _execute_channel_command(
        self,
        cmd: CommandInfo,
        args: dict[str, Any],
        channel: Channel,
    ) -> None:
        await dispatch_channel_command(cmd, args, channel)

    def dump_schema_json(self) -> str:
        """Return the registered API graph as canonical ``zynk-schema`` JSON."""
        return dump_api_graph_json(get_registry().get_api_graph())

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
        message_handlers = registry.get_all_message_handlers()
        uploads = registry.get_all_uploads()
        statics = registry.get_all_statics()

        self._print_startup_banner(dev, commands, message_handlers, uploads, statics)
        if self.debug:
            self._log_registered_handlers(commands, message_handlers, uploads, statics)

        if dev:
            self._run_dev(uvicorn, commands, message_handlers, uploads)
        else:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="debug" if self.debug else "info",
            )

    def _print_startup_banner(self, dev, commands, message_handlers, uploads, statics) -> None:
        mode = "Development" if dev else "Production"
        lines = [
            f"Server:     http://{self.host}:{self.port}",
            f"Mode:       {mode}",
            f"Commands:   {len(commands)}",
        ]
        if uploads:
            lines.append(f"Uploads:    {len(uploads)}")
        if statics:
            lines.append(f"Static:     {len(statics)}")
        if message_handlers:
            lines.append(f"WebSockets: {len(message_handlers)}")
        if self.generate_ts:
            lines.append(f"TypeScript: {self.generate_ts}")

        console = Console()
        panel = Panel.fit("\n".join(lines), title=self.title, border_style="blue")
        console.print("")
        console.print(panel)
        console.print("")

    def _log_registered_handlers(self, commands, message_handlers, uploads, statics) -> None:
        logger.debug("Registered commands:")
        for cmd in commands.values():
            channel_marker = " [channel]" if cmd.has_channel else ""
            logger.debug(f"  - {cmd.name}{channel_marker} ({cmd.module})")
        if uploads:
            logger.debug("Registered upload handlers:")
            for upload in uploads.values():
                multi_marker = " [multi]" if upload.is_multi_file else ""
                logger.debug(f"  - /upload/{upload.name}{multi_marker} ({upload.module})")
        if statics:
            logger.debug("Registered static handlers:")
            for static in statics.values():
                logger.debug(f"  - /static/{static.name} ({static.module})")
        if message_handlers:
            logger.debug("Registered message handlers:")
            for handler in message_handlers.values():
                logger.debug(f"  - /ws/{handler.name} ({handler.module})")

    def _run_dev(self, uvicorn, commands, message_handlers, uploads) -> None:
        from . import server

        user_modules = self._collect_user_modules(commands, message_handlers, uploads)
        server.set_config(
            generate_ts=self.generate_ts,
            host=self.host,
            port=self.port,
            cors_origins=self.cors_origins,
            title=self.title,
            debug=self.debug,
            app_init=self.app_init,
            import_modules=user_modules,
        )

        uvicorn.run(
            app="zynk.server:create_app",
            host=self.host,
            port=self.port,
            reload=True,
            reload_dirs=self.reload_dirs or ["."],
            reload_includes=self._normalize_patterns(self.reload_includes),
            reload_excludes=self._build_reload_excludes(),
            factory=True,
            log_level="debug" if self.debug else "info",
        )

    @staticmethod
    def _collect_user_modules(commands, message_handlers, uploads) -> list[str]:
        modules: set[str] = set()
        for source in (commands.values(), message_handlers.values(), uploads.values()):
            for item in source:
                if not item.module.startswith("zynk"):
                    modules.add(item.module)
        return list(modules)

    def _build_reload_excludes(self) -> list[str]:
        defaults = [".git", "__pycache__", "node_modules", ".venv"]
        if self.reload_excludes is None:
            return defaults
        user = self._normalize_patterns(self.reload_excludes) or []
        return user + [d for d in defaults if d not in user]

    @staticmethod
    def _normalize_patterns(patterns: list[str] | None) -> list[str] | None:
        if patterns is None:
            return None
        cwd = Path.cwd()
        normalized = []
        for pattern in patterns:
            p = Path(pattern)
            if not p.is_absolute():
                normalized.append(pattern)
                continue
            try:
                normalized.append(str(p.relative_to(cwd)))
            except ValueError:
                normalized.append(pattern)
        return normalized


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
