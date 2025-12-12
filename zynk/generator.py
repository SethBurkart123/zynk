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
                not field_info.is_required() or
                get_origin(field_info.annotation) is Union and
                type(None) in get_args(field_info.annotation)
            )

            optional_mark = "?" if is_optional else ""

            description = field_info.description
            if description:
                lines.append(f"    /** {description} */")

            lines.append(f"    {ts_name}{optional_mark}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

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
        param_mapping = []  # List of (camelCase, snake_case, is_optional) tuples
        if cmd.params:
            param_fields = []
            required_fields = []
            optional_fields = []
            for param_name, param_type in cmd.params.items():
                # Convert to camelCase for TypeScript
                ts_param_name = python_name_to_camel_case(param_name)
                ts_type = self._type_to_ts(param_type, models_to_generate)
                is_optional = param_name in cmd.optional_params

                # Use ?: for optional params, strip " | undefined" suffix if present
                if is_optional:
                    clean_type = ts_type.removesuffix(" | undefined")
                    optional_fields.append(f"{ts_param_name}?: {clean_type}")
                else:
                    required_fields.append(f"{ts_param_name}: {ts_type}")
                param_mapping.append((ts_param_name, param_name, is_optional))

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

        if cmd.has_channel:
            if cmd.params:
                # Build object literal mapping camelCase to snake_case
                mappings = [f"{snake}: args.{camel}" for camel, snake, _ in param_mapping]
                args_obj = "{ " + ", ".join(mappings) + " }"
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
                # Build object literal mapping camelCase to snake_case
                mappings = [f"{snake}: args.{camel}" for camel, snake, _ in param_mapping]
                args_obj = "{ " + ", ".join(mappings) + " }"
                lines.append(
                    f"export async function {fn_name}(args: {params_type}): "
                    f"Promise<{return_type}> {{"
                )
                lines.append(f'    return request("{cmd.name}", {args_obj});')
            else:
                lines.append(
                    f"export async function {fn_name}(): Promise<{return_type}> {{"
                )
                lines.append(f'    return request("{cmd.name}", {{}});')
            lines.append("}")

        return "\n".join(lines)

    def _generate_internal_module(self) -> str:
        """Generate the internal bridge utilities module."""
        return '''// Internal bridge utilities - do not modify
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

    let eventSource: EventSource | null = null;
    let messageCallback: ((data: T) => void) | null = null;
    let errorCallback: ((error: BridgeError) => void) | null = null;
    let closeCallback: (() => void) | null = null;

    // Start the SSE connection
    const startConnection = async () => {
        // First, initiate the channel via POST
        const initResponse = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(args),
        });

        if (!initResponse.ok) {
            const data = await initResponse.json();
            if (errorCallback) {
                errorCallback({
                    code: data.code || "CHANNEL_INIT_ERROR",
                    message: data.message || "Failed to initialize channel",
                    details: data.details,
                });
            }
            return;
        }

        const { channelId } = await initResponse.json();

        // Now connect to SSE endpoint
        eventSource = new EventSource(`${baseUrl}/channel/stream/${channelId}`);

        eventSource.addEventListener("message", (event) => {
            if (messageCallback) {
                const data = JSON.parse(event.data);
                messageCallback(data as T);
            }
        });

        eventSource.addEventListener("error", () => {
            if (errorCallback) {
                errorCallback({
                    code: "CHANNEL_ERROR",
                    message: "Channel connection error",
                });
            }
        });

        eventSource.addEventListener("close", () => {
            if (closeCallback) {
                closeCallback();
            }
            eventSource?.close();
        });
    };

    // Start connection immediately
    startConnection();

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
            eventSource?.close();
            if (closeCallback) {
                closeCallback();
            }
        },
    };
}
'''

    def generate(self, output_path: str) -> None:
        """
        Generate the complete TypeScript client file.

        Args:
            output_path: Path where the TypeScript file will be written.
        """
        registry = get_registry()
        commands = registry.get_all_commands()
        models = registry.get_all_models()

        if not commands:
            logger.warning("No commands registered. Generating empty client.")

        output_path = Path(output_path)
        output_dir = output_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        internal_path = output_dir / "_internal.ts"
        with open(internal_path, "w") as f:
            f.write(self._generate_internal_module())
        logger.debug(f"Generated internal module: {internal_path}")

        models_to_generate: set[str] = set()
        sections: list[str] = []

        sections.append(f"""/* Auto-generated by Zynk - DO NOT EDIT */
/* Generated: {datetime.now().isoformat()} */

import {{ initBridge, request, createChannel, BridgeRequestError }} from "./_internal";
import type {{ BridgeChannel, BridgeError }} from "./_internal";

export {{ initBridge, BridgeRequestError }};
export type {{ BridgeChannel, BridgeError }};
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
                    interface_code = self._generate_model_interface(model, models_to_generate)
                    model_interfaces.append(interface_code)
                generated_models.add(model_name)

        if model_interfaces:
            sections.append("// ============ Interfaces ============\n")
            sections.append("\n\n".join(model_interfaces))
            sections.append("")

        if command_functions:
            sections.append("\n// ============ Commands ============\n")
            sections.append("\n\n".join(command_functions))

        content = "\n".join(sections)
        with open(output_path, "w") as f:
            f.write(content)

        logger.debug(
            f"Generated TypeScript client: {output_path} "
            f"({len(commands)} commands, {len(generated_models)} interfaces)"
        )


def generate_typescript(output_path: str) -> None:
    """
    Generate TypeScript client code.

    Args:
        output_path: Path where the TypeScript file will be written.
    """
    generator = TypeScriptGenerator()
    generator.generate(output_path)
