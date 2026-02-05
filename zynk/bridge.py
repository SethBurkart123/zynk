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
from typing import Any, Union, get_args, get_origin

from fastapi import FastAPI, File, Request, UploadFile as FastAPIUploadFile
from fastapi import WebSocket as FastAPIWebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .channel import Channel
from .errors import (
    BridgeError,
    CommandExecutionError,
    CommandNotFoundError,
    InternalError,
    StaticHandlerNotFoundError,
    UploadHandlerNotFoundError,
    UploadValidationError,
    ValidationError,
)
from .generator import generate_typescript
from .registry import CommandInfo, get_registry
from .upload import UploadFile, UploadInfo
from .static import StaticFile, StaticInfo
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
            reload_includes: Glob patterns to include in file watching (dev mode only).
            reload_excludes: Glob patterns to exclude from file watching (dev mode only).
        """
        self.generate_ts = generate_ts
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self.title = title
        self.debug = debug
        self.reload_includes = reload_includes
        self.reload_excludes = reload_excludes
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
        async def upload_endpoint(
            handler_name: str,
            request: Request,
            files: list[FastAPIUploadFile] = File(default=[]),
        ):
            """
            Handle file uploads with XMLHttpRequest progress support.

            Files are sent as multipart/form-data with additional args in _args field.
            """
            registry = get_registry()
            upload_handler = registry.get_upload(handler_name)

            if not upload_handler:
                raise UploadHandlerNotFoundError(handler_name)

            # Parse additional args from _args form field
            form = await request.form()
            try:
                args = json.loads(form.get("_args", "{}"))
            except Exception as e:
                raise ValidationError(f"Invalid _args JSON: {e}")

            # Convert FastAPI UploadFiles to our UploadFile wrapper
            upload_files: list[UploadFile] = []
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

        @self.app.get("/static/{handler_name}")
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
        """
        Execute a static file handler.

        Args:
            handler: The static handler info.
            args: Query parameter arguments.

        Returns:
            The StaticFile result.
        """
        try:
            kwargs = {}
            for param_name, param_type in handler.params.items():
                if param_name in args:
                    value = args[param_name]
                    if param_type is int:
                        value = int(value)
                    elif param_type is float:
                        value = float(value)
                    elif param_type is bool:
                        value = value.lower() in ("true", "1", "yes")
                    kwargs[param_name] = value
                elif param_name not in handler.optional_params:
                    raise ValidationError(f"Missing required parameter: {param_name}")

            if handler.is_async:
                result = await handler.func(**kwargs)
            else:
                result = handler.func(**kwargs)

            if not isinstance(result, StaticFile):
                raise CommandExecutionError(
                    f"Static handler '{handler.name}' must return a StaticFile"
                )

            return result

        except FileNotFoundError as e:
            raise ValidationError(str(e))
        except ValueError as e:
            raise ValidationError(str(e))
        except Exception as e:
            logger.exception(f"Static handler '{handler.name}' failed")
            raise CommandExecutionError(str(e))

    async def _execute_upload(
        self,
        handler: UploadInfo,
        files: list[UploadFile],
        args: dict[str, Any],
    ) -> Any:
        """
        Execute an upload handler.

        Args:
            handler: The upload handler info.
            files: List of uploaded files.
            args: Additional arguments from _args.

        Returns:
            The handler result.
        """
        try:
            kwargs = {}

            # Add file(s) parameter
            if handler.is_multi_file:
                kwargs[handler.file_param] = files
            else:
                # Single file - take the first one
                if not files:
                    raise ValidationError("No file provided")
                kwargs[handler.file_param] = files[0]

            # Add other parameters
            for param_name, param_type in handler.params.items():
                if param_name in args:
                    kwargs[param_name] = instantiate_model(args[param_name], param_type)

            result = await handler.func(**kwargs)

            if isinstance(result, BaseModel):
                return result.model_dump()
            elif isinstance(result, list):
                return [
                    item.model_dump() if isinstance(item, BaseModel) else item
                    for item in result
                ]

            return result

        except PydanticValidationError as e:
            raise ValidationError(f"Validation failed: {e}")
        except TypeError as e:
            raise ValidationError(f"Invalid arguments: {e}")
        except Exception as e:
            logger.exception(f"Upload handler '{handler.name}' failed")
            raise CommandExecutionError(str(e))

    def _setup_error_handlers(self) -> None:
        """Setup exception handlers."""

        @self.app.exception_handler(BridgeError)
        async def bridge_error_handler(request: Request, exc: BridgeError):
            status_code = 400
            if isinstance(exc, (CommandNotFoundError, UploadHandlerNotFoundError, StaticHandlerNotFoundError)):
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
            for param_name, param_type in cmd.params.items():
                if param_name in args:
                    # Instantiate Pydantic models from dicts
                    kwargs[param_name] = instantiate_model(args[param_name], param_type)

            result = await cmd.func(**kwargs)

            if isinstance(result, BaseModel):
                return result.model_dump()
            elif isinstance(result, list):
                return [
                    item.model_dump() if isinstance(item, BaseModel) else item
                    for item in result
                ]

            return result

        except PydanticValidationError as e:
            raise ValidationError(f"Validation failed: {e}")
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
            for param_name, param_type in cmd.params.items():
                if param_name in args:
                    # Instantiate Pydantic models from dicts
                    kwargs[param_name] = instantiate_model(args[param_name], param_type)

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
        message_handlers = registry.get_all_message_handlers()
        uploads = registry.get_all_uploads()
        statics = registry.get_all_statics()

        mode = "Development" if dev else "Production"

        content = f"""Server:     http://{self.host}:{self.port}
Mode:       {mode}
Commands:   {len(commands)}"""
        if uploads:
            content += f"\nUploads:    {len(uploads)}"
        if statics:
            content += f"\nStatic:     {len(statics)}"
        if message_handlers:
            content += f"\nWebSockets: {len(message_handlers)}"
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
            if uploads:
                logger.debug("Registered upload handlers:")
                for upload in uploads.values():
                    multi_marker = " [multi]" if upload.is_multi_file else ""
                    logger.debug(
                        f"  - /upload/{upload.name}{multi_marker} ({upload.module})"
                    )
            if statics:
                logger.debug("Registered static handlers:")
                for static in statics.values():
                    logger.debug(f"  - /static/{static.name} ({static.module})")
            if message_handlers:
                logger.debug("Registered message handlers:")
                for handler in message_handlers.values():
                    logger.debug(f"  - /ws/{handler.name} ({handler.module})")

        if dev:
            command_modules = set()
            for cmd in commands.values():
                module_name = cmd.module
                if not module_name.startswith("zynk"):
                    command_modules.add(module_name)
            for upload in uploads.values():
                module_name = upload.module
                if not module_name.startswith("zynk"):
                    command_modules.add(module_name)
            for handler in message_handlers.values():
                module_name = handler.module
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

            uvicorn_kwargs = {
                "app": "zynk.server:create_app",
                "host": self.host,
                "port": self.port,
                "reload": True,
                "reload_dirs": ["."],
                "factory": True,
                "log_level": "debug" if self.debug else "info",
            }
            if self.reload_includes is not None:
                uvicorn_kwargs["reload_includes"] = self.reload_includes
            if self.reload_excludes is not None:
                uvicorn_kwargs["reload_excludes"] = self.reload_excludes

            uvicorn.run(**uvicorn_kwargs)
        else:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="debug" if self.debug else "info",
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
