"""Shared execution paths for registered handlers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from zynk.errors import CommandExecutionError, ValidationError
from zynk.registry import CommandInfo
from zynk.static import StaticFile, StaticInfo
from zynk.upload import UploadFile, UploadInfo

from .coerce import coerce_query_value, instantiate_model, serialize_result

logger = logging.getLogger(__name__)


async def execute_command(cmd: CommandInfo, args: dict[str, Any]) -> Any:
    try:
        await run_deps(getattr(cmd, "deps", []))
        kwargs = build_kwargs(cmd.params, args)
        return serialize_result(await cmd.func(**kwargs))
    except PydanticValidationError as e:
        raise ValidationError(f"Validation failed: {e}")
    except TypeError as e:
        raise ValidationError(f"Invalid arguments: {e}")
    except Exception as e:
        logger.exception(f"Command '{cmd.name}' failed")
        raise CommandExecutionError(str(e))


async def execute_channel_command(cmd: CommandInfo, args: dict[str, Any], channel: Any) -> None:
    try:
        kwargs = {"channel": channel, **build_kwargs(cmd.params, args)}
        await cmd.func(**kwargs)
        await channel.close()
    except Exception as e:
        logger.exception(f"Channel command '{cmd.name}' failed")
        await channel.send_error(str(e))


async def execute_upload(handler: UploadInfo, files: list[UploadFile], args: dict[str, Any]) -> Any:
    try:
        kwargs = {}
        if handler.is_multi_file:
            kwargs[handler.file_param] = files
        else:
            if not files:
                raise ValidationError("No file provided")
            kwargs[handler.file_param] = files[0]
        kwargs.update(build_kwargs(handler.params, args))
        return serialize_result(await handler.func(**kwargs))
    except PydanticValidationError as e:
        raise ValidationError(f"Validation failed: {e}")
    except TypeError as e:
        raise ValidationError(f"Invalid arguments: {e}")
    except Exception as e:
        logger.exception(f"Upload handler '{handler.name}' failed")
        raise CommandExecutionError(str(e))


async def execute_static(handler: StaticInfo, args: dict[str, Any]) -> StaticFile:
    try:
        kwargs = {}
        for param_name, param_type in handler.params.items():
            if param_name in args:
                kwargs[param_name] = coerce_query_value(args[param_name], param_type)
            elif param_name not in handler.optional_params:
                raise ValidationError(f"Missing required parameter: {param_name}")

        result = await handler.func(**kwargs) if handler.is_async else handler.func(**kwargs)
        if not isinstance(result, StaticFile):
            raise CommandExecutionError(f"Static handler '{handler.name}' must return a StaticFile")
        return result
    except FileNotFoundError as e:
        raise ValidationError(str(e))
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.exception(f"Static handler '{handler.name}' failed")
        raise CommandExecutionError(str(e))


async def run_deps(deps: list[Any]) -> None:
    for dep in deps:
        result = dep()
        if asyncio.iscoroutine(result):
            await result


def build_kwargs(params: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return {
        param_name: instantiate_model(args[param_name], param_type)
        for param_name, param_type in params.items()
        if param_name in args
    }
