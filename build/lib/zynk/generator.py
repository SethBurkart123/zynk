"""
TypeScript Generator Module

Generates strictly-typed TypeScript client code from Python command definitions.
Handles Pydantic model conversion and produces tree-shakeable exports.
"""

from __future__ import annotations

import logging
import types
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from .registry import CommandInfo, get_registry
from .static import StaticInfo
from .upload import UploadInfo
from .websocket import MessageHandlerInfo

logger = logging.getLogger(__name__)


# Python to TypeScript type mapping
PYTHON_TO_TS_TYPES: dict[Any, str] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    bytes: "string",  # Base64 encoded
    type(None): "undefined",
    None: "undefined",
}


def python_name_to_camel_case(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def python_name_to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(x.title() for x in name.split("_"))


class TypeScriptGenerator:
    """
    Generates TypeScript client code from Zynk command registry.

    Features:
    - Converts Pydantic models to TypeScript interfaces
    - Generates flat function exports for tree-shaking
    - Handles Optional, List, Dict types
    - Generates internal bridge utilities
    """

    def __init__(self):
        self._generated_models: set[str] = set()
        self._model_dependencies: dict[str, set[str]] = {}

    def _type_to_ts(self, type_hint: Any, models_to_generate: set[str]) -> str:
        """
        Convert a Python type hint to TypeScript type.

        Args:
            type_hint: The Python type hint.
            models_to_generate: Set to collect Pydantic model names that need generation.

        Returns:
            The TypeScript type string.
        """
        if type_hint is None:
            return "void"

        # Check direct type mapping
        if type_hint in PYTHON_TO_TS_TYPES:
            return PYTHON_TO_TS_TYPES[type_hint]

        # Handle Any
        if type_hint is Any:
            return "unknown"

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Union types (both old Union[T, None] and new T | None syntax)
        if origin is Union or origin is types.UnionType:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and type(None) in args:
                inner = self._type_to_ts(non_none_args[0], models_to_generate)
                return f"{inner} | undefined"
            else:
                ts_types = [self._type_to_ts(a, models_to_generate) for a in args]
                return " | ".join(ts_types)

        if origin is list:
            if args:
                inner = self._type_to_ts(args[0], models_to_generate)
                return f"{inner}[]"
            return "unknown[]"

        if origin is dict:
            if len(args) >= 2:
                key_type = self._type_to_ts(args[0], models_to_generate)
                value_type = self._type_to_ts(args[1], models_to_generate)
                if key_type not in ("string", "number"):
                    key_type = "string"
                return f"Record<{key_type}, {value_type}>"
            return "Record<string, unknown>"

        if origin is tuple:
            if args:
                inner_types = [self._type_to_ts(a, models_to_generate) for a in args]
                return f"[{', '.join(inner_types)}]"
            return "unknown[]"

        if origin is set:
            if args:
                inner = self._type_to_ts(args[0], models_to_generate)
                return f"{inner}[]"
            return "unknown[]"

        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            models_to_generate.add(type_hint.__name__)
            return type_hint.__name__

        # Handle Enum types - convert to string literal union for str enums
        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            # For string enums, generate a union of string literals
            if issubclass(type_hint, str):
                literals = [f'"{member.value}"' for member in type_hint]
                return " | ".join(literals)
            # For other enums, use the enum values' types
            return "string"

        if isinstance(type_hint, type):
            type_name = type_hint.__name__
            if type_name in PYTHON_TO_TS_TYPES:
                return PYTHON_TO_TS_TYPES[type_hint]
            return "unknown"

        return "unknown"

    def _generate_model_interface(
        self,
        model: type[BaseModel],
        models_to_generate: set[str],
    ) -> str:
        """
        Generate TypeScript interface for a Pydantic model.

        Args:
            model: The Pydantic model class.
            models_to_generate: Set to collect nested model names.

        Returns:
            TypeScript interface definition string.
        """
        lines = []

        if model.__doc__:
            lines.append("/**")
            for line in model.__doc__.strip().split("\n"):
                lines.append(f" * {line.strip()}")
            lines.append(" */")

        lines.append(f"export interface {model.__name__} {{")

        for field_name, field_info in model.model_fields.items():
            ts_name = python_name_to_camel_case(field_name)
            ts_type = self._type_to_ts(field_info.annotation, models_to_generate)

            is_optional = (
                not field_info.is_required()
                or get_origin(field_info.annotation) is Union
                and type(None) in get_args(field_info.annotation)
            )

            optional_mark = "?" if is_optional else ""

            description = field_info.description
            if description:
                lines.append(f"    /** {description} */")

            lines.append(f"    {ts_name}{optional_mark}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

    def _generate_inline_field_mapping(
        self,
        model: type[BaseModel],
        obj_ref: str,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate inline object mapping for converting camelCase to snake_case.

        Args:
            model: The Pydantic model class
            obj_ref: The object reference (e.g., "args.body")
            models_to_generate: Set to collect nested model names

        Returns:
            TypeScript expression for the mapped object
        """
        needs_object_literal = False
        field_mappings = []

        for field_name, field_info in model.model_fields.items():
            ts_name = python_name_to_camel_case(field_name)
            annotation = field_info.annotation

            origin = get_origin(annotation)
            args = get_args(annotation)
            actual_type = annotation
            is_optional = False

            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]
                    is_optional = type(None) in args

            actual_origin = get_origin(actual_type)
            actual_args = get_args(actual_type)

            name_needs_conversion = ts_name != field_name

            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                nested_mapping = self._generate_inline_field_mapping(
                    actual_type, f"{obj_ref}.{ts_name}", models_to_generate
                )
                if nested_mapping != f"{obj_ref}.{ts_name}" or name_needs_conversion:
                    needs_object_literal = True
                    if is_optional:
                        field_mappings.append(
                            f"{field_name}: {obj_ref}.{ts_name} != null ? {nested_mapping} : undefined"
                        )
                    else:
                        field_mappings.append(f"{field_name}: {nested_mapping}")
                else:
                    field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")

            elif actual_origin is list and actual_args:
                item_type = actual_args[0]
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    item_mapping = self._generate_inline_field_mapping(
                        item_type, "item", models_to_generate
                    )
                    if item_mapping != "item" or name_needs_conversion:
                        needs_object_literal = True
                        if is_optional:
                            field_mappings.append(
                                f"{field_name}: {obj_ref}.{ts_name} != null ? "
                                f"{obj_ref}.{ts_name}.map(item => ({item_mapping})) : undefined"
                            )
                        else:
                            field_mappings.append(
                                f"{field_name}: {obj_ref}.{ts_name}.map(item => ({item_mapping}))"
                            )
                    else:
                        field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")
                elif name_needs_conversion:
                    needs_object_literal = True
                    field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")
                else:
                    field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")

            elif name_needs_conversion:
                needs_object_literal = True
                field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")
            else:
                field_mappings.append(f"{field_name}: {obj_ref}.{ts_name}")

        if not needs_object_literal:
            return obj_ref

        return "{ " + ", ".join(field_mappings) + " }"

    def _generate_response_field_mapping(
        self,
        model: type[BaseModel],
        obj_ref: str,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate inline object mapping for converting snake_case response to camelCase.

        This is the reverse of _generate_inline_field_mapping - used for response data
        coming FROM the server (snake_case) TO TypeScript (camelCase).

        Args:
            model: The Pydantic model class
            obj_ref: The object reference (e.g., "_r" for response)
            models_to_generate: Set to collect nested model names

        Returns:
            TypeScript expression for the mapped object
        """
        needs_object_literal = False
        field_mappings = []

        for field_name, field_info in model.model_fields.items():
            ts_name = python_name_to_camel_case(field_name)
            annotation = field_info.annotation

            origin = get_origin(annotation)
            args = get_args(annotation)
            actual_type = annotation
            is_optional = False

            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]
                    is_optional = type(None) in args

            actual_origin = get_origin(actual_type)
            actual_args = get_args(actual_type)

            name_needs_conversion = ts_name != field_name

            # For responses: { camelCase: response.snake_case }
            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                nested_mapping = self._generate_response_field_mapping(
                    actual_type, f"{obj_ref}.{field_name}", models_to_generate
                )
                if nested_mapping != f"{obj_ref}.{field_name}" or name_needs_conversion:
                    needs_object_literal = True
                    if is_optional:
                        field_mappings.append(
                            f"{ts_name}: {obj_ref}.{field_name} != null ? {nested_mapping} : undefined"
                        )
                    else:
                        field_mappings.append(f"{ts_name}: {nested_mapping}")
                else:
                    field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")

            elif actual_origin is list and actual_args:
                item_type = actual_args[0]
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    item_mapping = self._generate_response_field_mapping(
                        item_type, "item", models_to_generate
                    )
                    if item_mapping != "item" or name_needs_conversion:
                        needs_object_literal = True
                        if is_optional:
                            field_mappings.append(
                                f"{ts_name}: {obj_ref}.{field_name} != null ? "
                                f"{obj_ref}.{field_name}.map((item: any) => ({item_mapping})) : undefined"
                            )
                        else:
                            field_mappings.append(
                                f"{ts_name}: {obj_ref}.{field_name}.map((item: any) => ({item_mapping}))"
                            )
                    else:
                        field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")
                elif name_needs_conversion:
                    needs_object_literal = True
                    field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")
                else:
                    field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")

            elif name_needs_conversion:
                needs_object_literal = True
                field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")
            else:
                field_mappings.append(f"{ts_name}: {obj_ref}.{field_name}")

        if not needs_object_literal:
            return obj_ref

        return "{ " + ", ".join(field_mappings) + " }"

    def _generate_command_function(
        self,
        cmd: CommandInfo,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate TypeScript function for a command.

        Args:
            cmd: The command info.
            models_to_generate: Set to collect model names.

        Returns:
            TypeScript function definition string.
        """
        lines = []

        # Function name (convert to camelCase)
        fn_name = python_name_to_camel_case(cmd.name)

        # Generate parameter interface with camelCase names
        params_type = "void"
        param_mapping = []  # List of (camelCase, snake_case, is_optional, model_name_or_none) tuples
        if cmd.params:
            param_fields = []
            required_fields = []
            optional_fields = []
            for param_name, param_type in cmd.params.items():
                # Convert to camelCase for TypeScript
                ts_param_name = python_name_to_camel_case(param_name)
                ts_type = self._type_to_ts(param_type, models_to_generate)
                is_optional = param_name in cmd.optional_params

                # Check if this param is a Pydantic model (needs converter)
                model_name = None
                actual_type = param_type
                origin = get_origin(param_type)
                args = get_args(param_type)

                # Handle Optional/Union types
                if origin is Union or (
                    hasattr(types, "UnionType") and origin is types.UnionType
                ):
                    non_none_args = [a for a in args if a is not type(None)]
                    if non_none_args:
                        actual_type = non_none_args[0]

                if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                    model_name = actual_type.__name__

                # Use ?: for optional params, strip " | undefined" suffix if present
                if is_optional:
                    clean_type = ts_type.removesuffix(" | undefined")
                    optional_fields.append(f"{ts_param_name}?: {clean_type}")
                else:
                    required_fields.append(f"{ts_param_name}: {ts_type}")
                param_mapping.append(
                    (ts_param_name, param_name, is_optional, model_name)
                )

            # Put required fields first, then optional fields
            param_fields = required_fields + optional_fields
            params_type = "{ " + "; ".join(param_fields) + " }"

        if cmd.has_channel:
            channel_type = "unknown"
            hints = {}
            try:
                hints = get_type_hints(cmd.func)
            except Exception:
                pass

            channel_hint = hints.get("channel")
            if channel_hint:
                args = get_args(channel_hint)
                if args:
                    channel_type = self._type_to_ts(args[0], models_to_generate)
            return_type = channel_type
        else:
            return_type = self._type_to_ts(cmd.return_type, models_to_generate)
            if return_type == "void" or return_type == "undefined":
                return_type = "void"

        if cmd.docstring:
            lines.append("/**")
            for line in cmd.docstring.strip().split("\n"):
                lines.append(f" * {line.strip()}")
            lines.append(" */")

        def get_param_mapping(
            camel: str, snake: str, is_optional: bool, param_type: Any
        ) -> str:
            """Generate the mapping expression for a single parameter."""
            actual_type = param_type
            origin = get_origin(param_type)
            args = get_args(param_type)

            if origin is Union or (
                hasattr(types, "UnionType") and origin is types.UnionType
            ):
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]

            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                inline_mapping = self._generate_inline_field_mapping(
                    actual_type, f"args.{camel}", models_to_generate
                )
                if is_optional:
                    return (
                        f"{snake}: args.{camel} != null ? {inline_mapping} : undefined"
                    )
                return f"{snake}: {inline_mapping}"

            return f"{snake}: args.{camel}"

        if cmd.params:
            mappings = []
            has_model_params = False

            for ts_param_name, param_name, is_optional, _ in param_mapping:
                param_type = cmd.params[param_name]
                mapping = get_param_mapping(
                    ts_param_name, param_name, is_optional, param_type
                )
                mappings.append(mapping)

                actual_type = param_type
                origin = get_origin(param_type)
                args = get_args(param_type)
                if origin is Union or (
                    hasattr(types, "UnionType") and origin is types.UnionType
                ):
                    non_none_args = [a for a in args if a is not type(None)]
                    if non_none_args:
                        actual_type = non_none_args[0]
                if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                    has_model_params = True

            all_names_match = all(
                ts_name == py_name for ts_name, py_name, _, _ in param_mapping
            )

            if all_names_match and not has_model_params:
                args_obj = "args"
            else:
                args_obj = "{ " + ", ".join(mappings) + " }"
        else:
            args_obj = "{}"

        # Check if return type needs response mapping
        actual_return_type = cmd.return_type
        return_origin = get_origin(actual_return_type) if actual_return_type else None
        return_args = get_args(actual_return_type) if actual_return_type else ()

        # Handle Optional return types
        if return_origin is Union or (
            hasattr(types, "UnionType") and return_origin is types.UnionType
        ):
            non_none_args = [a for a in return_args if a is not type(None)]
            if non_none_args:
                actual_return_type = non_none_args[0]
                return_origin = get_origin(actual_return_type)
                return_args = get_args(actual_return_type)

        # Determine if we need response mapping
        needs_response_mapping = False
        is_list_return = False
        return_model = None

        if return_origin is list and return_args:
            item_type = return_args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                needs_response_mapping = True
                is_list_return = True
                return_model = item_type
        elif isinstance(actual_return_type, type) and issubclass(
            actual_return_type, BaseModel
        ):
            needs_response_mapping = True
            return_model = actual_return_type

        if cmd.has_channel:
            if cmd.params:
                lines.append(
                    f"export function {fn_name}(args: {params_type}): "
                    f"BridgeChannel<{return_type}> {{"
                )
                lines.append(f'    return createChannel("{cmd.name}", {args_obj});')
            else:
                lines.append(
                    f"export function {fn_name}(): BridgeChannel<{return_type}> {{"
                )
                lines.append(f'    return createChannel("{cmd.name}", {{}});')
            lines.append("}")
        else:
            if cmd.params:
                lines.append(
                    f"export async function {fn_name}(args: {params_type}): "
                    f"Promise<{return_type}> {{"
                )
            else:
                lines.append(
                    f"export async function {fn_name}(): Promise<{return_type}> {{"
                )

            if needs_response_mapping and return_model:
                response_mapping = self._generate_response_field_mapping(
                    return_model, "_r", models_to_generate
                )
                if response_mapping != "_r":
                    lines.append(
                        f'    const _r = await request<any>("{cmd.name}", {args_obj});'
                    )
                    if is_list_return:
                        lines.append(
                            f"    return _r.map((_r: any) => ({response_mapping}));"
                        )
                    else:
                        lines.append(f"    return {response_mapping};")
                else:
                    lines.append(f'    return request("{cmd.name}", {args_obj});')
            else:
                lines.append(f'    return request("{cmd.name}", {args_obj});')
            lines.append("}")

        return "\n".join(lines)

    def _generate_upload_function(
        self,
        upload: UploadInfo,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate TypeScript function for an upload handler.

        Args:
            upload: The upload handler info.
            models_to_generate: Set to collect model names.

        Returns:
            TypeScript function definition string.
        """
        lines = []

        # Function name (convert to camelCase)
        fn_name = python_name_to_camel_case(upload.name)

        # Build parameter types
        param_fields = []
        required_fields = []
        optional_fields = []
        param_mapping = []  # (ts_name, py_name, is_optional)

        # Add file parameter - always required
        if upload.is_multi_file:
            required_fields.append("files: File[]")
        else:
            required_fields.append("file: File")

        # Add other parameters
        for param_name, param_type in upload.params.items():
            ts_param_name = python_name_to_camel_case(param_name)
            ts_type = self._type_to_ts(param_type, models_to_generate)
            is_optional = param_name in upload.optional_params

            if is_optional:
                clean_type = ts_type.removesuffix(" | undefined")
                optional_fields.append(f"{ts_param_name}?: {clean_type}")
            else:
                required_fields.append(f"{ts_param_name}: {ts_type}")
            param_mapping.append((ts_param_name, param_name, is_optional))

        param_fields = required_fields + optional_fields
        params_type = "{ " + "; ".join(param_fields) + " }"

        # Get return type
        return_type = self._type_to_ts(upload.return_type, models_to_generate)
        if return_type == "void" or return_type == "undefined":
            return_type = "void"

        # Generate JSDoc
        if upload.docstring:
            lines.append("/**")
            for line in upload.docstring.strip().split("\n"):
                lines.append(f" * {line.strip()}")
            lines.append(" */")

        # Generate function signature
        lines.append(
            f"export function {fn_name}(args: {params_type}): UploadHandle<{return_type}> {{"
        )

        # Build args mapping for non-file params
        if param_mapping:
            mappings = []
            for ts_name, py_name, _ in param_mapping:
                if ts_name != py_name:
                    mappings.append(f"{py_name}: args.{ts_name}")
                else:
                    mappings.append(f"{py_name}: args.{py_name}")

            # Construct the files array - wrap single file in array for createUpload
            if upload.is_multi_file:
                files_expr = "args.files"
            else:
                files_expr = "[args.file]"

            args_obj = "{ " + ", ".join(mappings) + " }"
            lines.append(
                f'    return createUpload("{upload.name}", {files_expr}, {args_obj});'
            )
        else:
            if upload.is_multi_file:
                files_expr = "args.files"
            else:
                files_expr = "[args.file]"
            lines.append(
                f'    return createUpload("{upload.name}", {files_expr}, {{}});'
            )

        lines.append("}")

        return "\n".join(lines)

    def _generate_static_function(
        self,
        static: StaticInfo,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate TypeScript URL function for a static file handler.

        Args:
            static: The static handler info.
            models_to_generate: Set to collect model names.

        Returns:
            TypeScript function definition string.
        """
        lines = []

        fn_name = python_name_to_camel_case(static.name) + "Url"

        # Build parameter types
        required_fields = []
        optional_fields = []
        param_mapping = []  # (ts_name, py_name, is_optional)

        for param_name, param_type in static.params.items():
            ts_param_name = python_name_to_camel_case(param_name)
            ts_type = self._type_to_ts(param_type, models_to_generate)
            is_optional = param_name in static.optional_params

            if is_optional:
                clean_type = ts_type.removesuffix(" | undefined")
                optional_fields.append(f"{ts_param_name}?: {clean_type}")
            else:
                required_fields.append(f"{ts_param_name}: {ts_type}")
            param_mapping.append((ts_param_name, param_name, is_optional))

        param_fields = required_fields + optional_fields
        params_type = "{ " + "; ".join(param_fields) + " }" if param_fields else "{}"

        # Generate JSDoc
        if static.docstring:
            lines.append("/**")
            for line in static.docstring.strip().split("\n"):
                lines.append(f" * {line.strip()}")
            lines.append(" */")

        # Generate function signature
        if param_fields:
            lines.append(f"export function {fn_name}(args: {params_type}): string {{")
        else:
            lines.append(f"export function {fn_name}(): string {{")

        # Build URL with query params
        lines.append("    const params = new URLSearchParams();")

        for ts_name, py_name, is_optional in param_mapping:
            if is_optional:
                lines.append(
                    f'    if (args.{ts_name} !== undefined) params.set("{py_name}", String(args.{ts_name}));'
                )
            else:
                lines.append(f'    params.set("{py_name}", String(args.{ts_name}));')

        lines.append(
            f"    return `${{getBaseUrl()}}/static/{static.name}?${{params}}`;"
        )
        lines.append("}")

        return "\n".join(lines)

    def _generate_websocket_class(
        self,
        handler: MessageHandlerInfo,
        models_to_generate: set[str],
    ) -> str:
        """
        Generate TypeScript WebSocket class for a message handler.

        Args:
            handler: The message handler info.
            models_to_generate: Set to collect model names.

        Returns:
            TypeScript class definition string.
        """
        lines = []

        # Class name in PascalCase + "Socket"
        class_name = python_name_to_pascal_case(handler.name) + "Socket"

        # Generate event type interfaces
        server_events_name = f"{python_name_to_pascal_case(handler.name)}ServerEvents"
        client_events_name = f"{python_name_to_pascal_case(handler.name)}ClientEvents"

        server_event_entries = []
        for event_name, event_type in handler.server_event_types.items():
            ts_type = self._type_to_ts(event_type, models_to_generate)
            server_event_entries.append(f"    {event_name}: {ts_type};")

        client_event_entries = []
        for event_name, event_type in handler.client_event_types.items():
            ts_type = self._type_to_ts(event_type, models_to_generate)
            client_event_entries.append(f"    {event_name}: {ts_type};")

        # Generate event interfaces
        lines.append(f"export interface {server_events_name} {{")
        lines.extend(server_event_entries)
        lines.append("}")
        lines.append("")

        lines.append(f"export interface {client_events_name} {{")
        lines.extend(client_event_entries)
        lines.append("}")
        lines.append("")

        # Generate class docstring
        if handler.docstring:
            lines.append("/**")
            for line in handler.docstring.strip().split("\n"):
                lines.append(f" * {line.strip()}")
            lines.append(" */")

        # Generate the WebSocket class
        lines.append(f"export class {class_name} {{")
        lines.append("    private ws: WebSocket | null = null;")
        lines.append(
            "    private listeners: Map<string, Set<(data: unknown) => void>> = new Map();"
        )
        lines.append("    private connectionListeners: Set<() => void> = new Set();")
        lines.append(
            "    private disconnectionListeners: Set<(event: CloseEvent) => void> = new Set();"
        )
        lines.append(
            "    private errorListeners: Set<(error: Event) => void> = new Set();"
        )
        lines.append("")

        # connect() method
        lines.append("    connect(): void {")
        lines.append("        const baseUrl = getBaseUrl().replace(/^http/, 'ws');")
        lines.append(
            f"        this.ws = new WebSocket(`${{baseUrl}}/ws/{handler.name}`);"
        )
        lines.append("")
        lines.append("        this.ws.onopen = () => {")
        lines.append("            this.connectionListeners.forEach(cb => cb());")
        lines.append("        };")
        lines.append("")
        lines.append("        this.ws.onclose = (event) => {")
        lines.append(
            "            this.disconnectionListeners.forEach(cb => cb(event));"
        )
        lines.append("        };")
        lines.append("")
        lines.append("        this.ws.onerror = (error) => {")
        lines.append("            this.errorListeners.forEach(cb => cb(error));")
        lines.append("        };")
        lines.append("")
        lines.append("        this.ws.onmessage = (event) => {")
        lines.append("            try {")
        lines.append("                const message = JSON.parse(event.data);")
        lines.append("                const eventName = message.event;")
        lines.append("                const data = message.data;")
        lines.append("                const callbacks = this.listeners.get(eventName);")
        lines.append("                if (callbacks) {")
        lines.append("                    callbacks.forEach(cb => cb(data));")
        lines.append("                }")
        lines.append("            } catch (e) {")
        lines.append(
            "                console.error('[Zynk] Failed to parse WebSocket message:', e);"
        )
        lines.append("            }")
        lines.append("        };")
        lines.append("    }")
        lines.append("")

        # disconnect() method
        lines.append("    disconnect(): void {")
        lines.append("        this.ws?.close();")
        lines.append("        this.ws = null;")
        lines.append("    }")
        lines.append("")

        # isConnected getter
        lines.append("    get isConnected(): boolean {")
        lines.append("        return this.ws?.readyState === WebSocket.OPEN;")
        lines.append("    }")
        lines.append("")

        # Generic send method
        lines.append(f"    send<K extends keyof {client_events_name}>(")
        lines.append("        event: K,")
        lines.append(f"        data: {client_events_name}[K]")
        lines.append("    ): void {")
        lines.append("        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {")
        lines.append(
            "            throw new Error('[Zynk] WebSocket is not connected');"
        )
        lines.append("        }")
        lines.append("        this.ws.send(JSON.stringify({ event, data }));")
        lines.append("    }")
        lines.append("")

        # Generic on method
        lines.append(f"    on<K extends keyof {server_events_name}>(")
        lines.append("        event: K,")
        lines.append(f"        callback: (data: {server_events_name}[K]) => void")
        lines.append("    ): () => void {")
        lines.append("        const eventStr = event as string;")
        lines.append("        if (!this.listeners.has(eventStr)) {")
        lines.append("            this.listeners.set(eventStr, new Set());")
        lines.append("        }")
        lines.append("        const cb = callback as (data: unknown) => void;")
        lines.append("        this.listeners.get(eventStr)!.add(cb);")
        lines.append("        return () => {")
        lines.append("            this.listeners.get(eventStr)?.delete(cb);")
        lines.append("        };")
        lines.append("    }")
        lines.append("")

        # onConnect method
        lines.append("    onConnect(callback: () => void): () => void {")
        lines.append("        this.connectionListeners.add(callback);")
        lines.append("        return () => {")
        lines.append("            this.connectionListeners.delete(callback);")
        lines.append("        };")
        lines.append("    }")
        lines.append("")

        # onDisconnect method
        lines.append(
            "    onDisconnect(callback: (event: CloseEvent) => void): () => void {"
        )
        lines.append("        this.disconnectionListeners.add(callback);")
        lines.append("        return () => {")
        lines.append("            this.disconnectionListeners.delete(callback);")
        lines.append("        };")
        lines.append("    }")
        lines.append("")

        # onError method
        lines.append("    onError(callback: (error: Event) => void): () => void {")
        lines.append("        this.errorListeners.add(callback);")
        lines.append("        return () => {")
        lines.append("            this.errorListeners.delete(callback);")
        lines.append("        };")
        lines.append("    }")

        for event_name, event_type in handler.server_event_types.items():
            ts_type = self._type_to_ts(event_type, models_to_generate)
            method_name = f"on{python_name_to_pascal_case(event_name)}"

            lines.append("")
            lines.append(
                f"    {method_name}(callback: (data: {ts_type}) => void): () => void {{"
            )
            if isinstance(event_type, type) and issubclass(event_type, BaseModel):
                response_mapping = self._generate_response_field_mapping(
                    event_type, "_d", models_to_generate
                )
                if response_mapping != "_d":
                    lines.append(
                        f'        return this.on("{event_name}", (_d) => callback({response_mapping}));'
                    )
                else:
                    lines.append(f'        return this.on("{event_name}", callback);')
            else:
                lines.append(f'        return this.on("{event_name}", callback);')
            lines.append("    }")

        for event_name, event_type in handler.client_event_types.items():
            ts_type = self._type_to_ts(event_type, models_to_generate)
            method_name = f"send{python_name_to_pascal_case(event_name)}"

            lines.append("")
            lines.append(f"    {method_name}(data: {ts_type}): void {{")
            if isinstance(event_type, type) and issubclass(event_type, BaseModel):
                input_mapping = self._generate_inline_field_mapping(
                    event_type, "data", models_to_generate
                )
                if input_mapping != "data":
                    lines.append(
                        f'        this.send("{event_name}", {input_mapping} as {ts_type});'
                    )
                else:
                    lines.append(f'        this.send("{event_name}", data);')
            else:
                lines.append(f'        this.send("{event_name}", data);')
            lines.append("    }")

        lines.append("}")
        lines.append("")

        # Factory function
        fn_name = f"create{python_name_to_pascal_case(handler.name)}Socket"
        lines.append(f"export function {fn_name}(): {class_name} {{")
        lines.append(f"    return new {class_name}();")
        lines.append("}")

        return "\n".join(lines)

    def _generate_internal_module(self) -> str:
        """Generate the internal bridge utilities module."""
        return """// Internal bridge utilities - do not modify
let _baseUrl: string | null = null;

export interface BridgeError {
    code: string;
    message: string;
    details?: unknown;
}

export class BridgeRequestError extends Error {
    code: string;
    details?: unknown;

    constructor(error: BridgeError) {
        super(error.message);
        this.name = "BridgeRequestError";
        this.code = error.code;
        this.details = error.details;
    }
}

export interface BridgeChannel<T> {
    subscribe(callback: (data: T) => void): void;
    onError(callback: (error: BridgeError) => void): void;
    onClose(callback: () => void): void;
    close(): void;
}

export function initBridge(baseUrl: string): void {
    _baseUrl = baseUrl.replace(/\\/$/, "");
    console.log(`[Zynk] Initialized with base URL: ${_baseUrl}`);
}

export function getBaseUrl(): string {
    if (!_baseUrl) {
        throw new Error(
            "[Zynk] Bridge not initialized. Call initBridge(url) first."
        );
    }
    return _baseUrl;
}

export async function request<T>(command: string, args: unknown): Promise<T> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/command/${command}`;

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(args),
    });

    const data = await response.json();

    if (!response.ok) {
        throw new BridgeRequestError({
            code: data.code || "UNKNOWN_ERROR",
            message: data.message || "An unknown error occurred",
            details: data.details,
        });
    }

    return data.result as T;
}

export function createChannel<T>(command: string, args: unknown): BridgeChannel<T> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/channel/${command}`;

    let abortController: AbortController | null = new AbortController();
    let messageCallback: ((data: T) => void) | null = null;
    let errorCallback: ((error: BridgeError) => void) | null = null;
    let closeCallback: (() => void) | null = null;

    const startStream = async () => {
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(args),
                signal: abortController?.signal,
            });

            if (!response.ok) {
                const data = await response.json();
                errorCallback?.({
                    code: data.code || "CHANNEL_ERROR",
                    message: data.message || "Failed to start channel",
                    details: data.details,
                });
                return;
            }

            const reader = response.body?.getReader();
            if (!reader) {
                errorCallback?.({ code: "NO_STREAM", message: "Response has no body" });
                return;
            }

            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const events = buffer.split("\\n\\n");
                buffer = events.pop() || "";

                for (const eventBlock of events) {
                    if (!eventBlock.trim()) continue;

                    let eventType = "message";
                    let eventData = "";

                    for (const line of eventBlock.split("\\n")) {
                        if (line.startsWith("event: ")) {
                            eventType = line.slice(7);
                        } else if (line.startsWith("data: ")) {
                            eventData = line.slice(6);
                        }
                    }

                    if (eventType === "error") {
                        const parsed = JSON.parse(eventData);
                        errorCallback?.({
                            code: "STREAM_ERROR",
                            message: parsed.error || "Stream error",
                        });
                    } else if (eventType === "close") {
                        closeCallback?.();
                    } else if (eventData) {
                        const parsed = JSON.parse(eventData);
                        messageCallback?.({ event: eventType, ...parsed } as T);
                    }
                }
            }

            closeCallback?.();
        } catch (err: unknown) {
            if (err instanceof Error && err.name === "AbortError") return;
            errorCallback?.({
                code: "STREAM_ERROR",
                message: err instanceof Error ? err.message : "Unknown error",
            });
        }
    };

    startStream();

    return {
        subscribe(callback: (data: T) => void): void {
            messageCallback = callback;
        },
        onError(callback: (error: BridgeError) => void): void {
            errorCallback = callback;
        },
        onClose(callback: () => void): void {
            closeCallback = callback;
        },
        close(): void {
            abortController?.abort();
            abortController = null;
        },
    };
}

export interface UploadProgressEvent {
    loaded: number;
    total: number;
    percentage: number;
}

export interface UploadHandle<T> {
    promise: Promise<T>;
    abort(): void;
    onProgress(callback: (event: UploadProgressEvent) => void): UploadHandle<T>;
}

export function createUpload<T>(
    handler: string,
    files: File[],
    args: Record<string, unknown>
): UploadHandle<T> {
    const xhr = new XMLHttpRequest();
    let progressCallback: ((e: UploadProgressEvent) => void) | null = null;

    const promise = new Promise<T>((resolve, reject) => {
        const formData = new FormData();

        // Add files
        for (const file of files) {
            formData.append("files", file);
        }

        // Add other args as JSON
        formData.append("_args", JSON.stringify(args));

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && progressCallback) {
                progressCallback({
                    loaded: e.loaded,
                    total: e.total,
                    percentage: Math.round((e.loaded / e.total) * 100),
                });
            }
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    resolve(data.result as T);
                } catch {
                    reject(new Error("Failed to parse response"));
                }
            } else {
                try {
                    const data = JSON.parse(xhr.responseText);
                    reject(new BridgeRequestError({
                        code: data.code || "UPLOAD_ERROR",
                        message: data.message || "Upload failed",
                        details: data.details,
                    }));
                } catch {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            }
        };

        xhr.onerror = () => reject(new Error("Upload failed"));
        xhr.onabort = () => reject(new Error("Upload aborted"));

        const baseUrl = getBaseUrl();
        xhr.open("POST", `${baseUrl}/upload/${handler}`);
        xhr.send(formData);
    });

    const handle: UploadHandle<T> = {
        promise,
        abort: () => xhr.abort(),
        onProgress: (cb) => {
            progressCallback = cb;
            return handle;
        },
    };

    return handle;
}
"""

    def generate(self, output_path: str) -> None:
        """
        Generate the complete TypeScript client file.

        Args:
            output_path: Path where the TypeScript file will be written.
        """
        registry = get_registry()
        commands = registry.get_all_commands()
        models = registry.get_all_models()
        message_handlers = registry.get_all_message_handlers()
        uploads = registry.get_all_uploads()
        statics = registry.get_all_statics()

        if not commands and not message_handlers and not uploads and not statics:
            logger.warning(
                "No commands, uploads, statics, or message handlers registered. Generating empty client."
            )

        output_path = Path(output_path)
        output_dir = output_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        internal_path = output_dir / "_internal.ts"
        with open(internal_path, "w") as f:
            f.write(self._generate_internal_module())
        logger.debug(f"Generated internal module: {internal_path}")

        models_to_generate: set[str] = set()
        sections: list[str] = []

        # Build imports based on what's needed
        imports = ["initBridge", "BridgeRequestError"]
        type_imports = ["BridgeError"]

        if commands:
            imports.append("request")
            # Check if any commands have channels
            if any(cmd.has_channel for cmd in commands.values()):
                imports.append("createChannel")
                type_imports.append("BridgeChannel")

        if uploads:
            imports.append("createUpload")
            type_imports.extend(["UploadHandle", "UploadProgressEvent"])

        if message_handlers or statics:
            imports.append("getBaseUrl")

        sections.append(f"""/* Auto-generated by Zynk - DO NOT EDIT */
/* Generated: {datetime.now().isoformat()} */

import {{ {", ".join(imports)} }} from "./_internal";
import type {{ {", ".join(type_imports)} }} from "./_internal";

export {{ initBridge, BridgeRequestError }};
export type {{ {", ".join(type_imports)} }};
""")

        command_functions: list[str] = []
        for cmd in sorted(commands.values(), key=lambda c: c.name):
            fn_code = self._generate_command_function(cmd, models_to_generate)
            command_functions.append(fn_code)

        for model_name, model in models.items():
            models_to_generate.add(model_name)

        generated_models: set[str] = set()
        model_interfaces: list[str] = []

        while models_to_generate - generated_models:
            current_batch = models_to_generate - generated_models
            for model_name in sorted(current_batch):
                model = models.get(model_name)
                if model:
                    interface_code = self._generate_model_interface(
                        model, models_to_generate
                    )
                    model_interfaces.append(interface_code)
                generated_models.add(model_name)

        if model_interfaces:
            sections.append("// ============ Interfaces ============\n")
            sections.append("\n\n".join(model_interfaces))
            sections.append("")

        if command_functions:
            sections.append("\n// ============ Commands ============\n")
            sections.append("\n\n".join(command_functions))

        # Generate upload functions
        upload_functions: list[str] = []
        for upload in sorted(uploads.values(), key=lambda u: u.name):
            fn_code = self._generate_upload_function(upload, models_to_generate)
            upload_functions.append(fn_code)

        if upload_functions:
            sections.append("\n// ============ Uploads ============\n")
            sections.append("\n\n".join(upload_functions))

        # Generate static URL functions
        static_functions: list[str] = []
        for static in sorted(statics.values(), key=lambda s: s.name):
            fn_code = self._generate_static_function(static, models_to_generate)
            static_functions.append(fn_code)

        if static_functions:
            sections.append("\n// ============ Static Files ============\n")
            sections.append("\n\n".join(static_functions))

        # Re-generate models in case upload handlers added new ones
        while models_to_generate - generated_models:
            current_batch = models_to_generate - generated_models
            for model_name in sorted(current_batch):
                model = models.get(model_name)
                if model:
                    interface_code = self._generate_model_interface(
                        model, models_to_generate
                    )
                    model_interfaces.append(interface_code)
                generated_models.add(model_name)

        # Generate WebSocket classes for message handlers
        websocket_classes: list[str] = []
        for handler in sorted(message_handlers.values(), key=lambda h: h.name):
            ws_code = self._generate_websocket_class(handler, models_to_generate)
            websocket_classes.append(ws_code)

        # Re-generate models in case WebSocket handlers added new ones
        while models_to_generate - generated_models:
            current_batch = models_to_generate - generated_models
            for model_name in sorted(current_batch):
                model = models.get(model_name)
                if model:
                    interface_code = self._generate_model_interface(
                        model, models_to_generate
                    )
                    model_interfaces.append(interface_code)
                generated_models.add(model_name)

        if websocket_classes:
            sections.append("\n// ============ WebSockets ============\n")
            sections.append("\n\n".join(websocket_classes))

        content = "\n".join(sections)
        with open(output_path, "w") as f:
            f.write(content)

        logger.debug(
            f"Generated TypeScript client: {output_path} "
            f"({len(commands)} commands, {len(uploads)} uploads, {len(statics)} statics, {len(message_handlers)} websockets, {len(generated_models)} interfaces)"
        )


def generate_typescript(output_path: str) -> None:
    """
    Generate TypeScript client code.

    Args:
        output_path: Path where the TypeScript file will be written.
    """
    generator = TypeScriptGenerator()
    generator.generate(output_path)
