"""Lower the Zynk API graph into a TypeScript module that targets Effect.

The lowering only knows about IR + Pydantic model definitions; the runtime
helpers (``callCommand`` etc.) live in a static module that the generated
client imports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from zynk.core.ir import ApiGraph, Endpoint, Param
from zynk.core.naming import python_name_to_camel_case

from .options import EffectGeneratorOptions
from .types import collect_model_refs, render_model_schema, type_expr


@dataclass(slots=True)
class EffectModule:
    banner: str
    imports: list[str]
    body: list[str]

    def render(self) -> str:
        parts = [self.banner.rstrip()]
        if self.imports:
            parts.append("\n".join(self.imports))
        parts.extend(section.rstrip() for section in self.body if section.strip())
        return "\n\n".join(parts).rstrip() + "\n"


def _pascal(name: str) -> str:
    return "".join(part.title() for part in name.split("_"))


def _camel(name: str) -> str:
    return python_name_to_camel_case(name)


def _params_object(params: Iterable[Param], model_names: set[str]) -> str:
    fields: list[str] = []
    required: list[str] = []
    optional: list[str] = []
    for param in params:
        expr = type_expr(param.type, model_names)
        ts = expr.ts
        if not param.required:
            ts = ts.removesuffix(" | undefined")
            optional.append(f"{param.ts_name}?: {ts}")
        else:
            required.append(f"{param.ts_name}: {ts}")
    fields = required + optional
    if not fields:
        return "void"
    return "{ " + "; ".join(fields) + " }"


def _args_payload(params: list[Param]) -> str:
    if not params:
        return "{}"
    parts = []
    for param in params:
        if param.ts_name == param.py_name:
            parts.append(f"{param.py_name}: args.{param.ts_name}")
        else:
            parts.append(f"{param.py_name}: args.{param.ts_name}")
    return "{ " + ", ".join(parts) + " }"


def _doc_block(doc: str | None) -> list[str]:
    if not doc:
        return []
    out = ["/**"]
    for line in doc.strip().splitlines():
        out.append(f" * {line.strip()}")
    out.append(" */")
    return out


# ---------------------------------------------------------------------------
# Endpoint emitters
# ---------------------------------------------------------------------------


def _emit_rpc(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    response = type_expr(endpoint.returns, model_names)
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines: list[str] = []
    lines.extend(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}, "
            f"options?: CallOptions"
            f"): Effect.Effect<{response.ts}, ZynkError, ZynkClient> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"options?: CallOptions"
            f"): Effect.Effect<{response.ts}, ZynkError, ZynkClient> =>"
        )
    lines.append(
        f'  callCommand("{endpoint.name}", {args_payload}, '
        f"{response.schema}, options)"
    )
    return "\n".join(lines)


def _emit_channel(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    item = type_expr(endpoint.channel_item, model_names) if endpoint.channel_item else type_expr(None, model_names)
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines: list[str] = []
    lines.extend(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}, "
            f"options?: CallOptions"
            f"): Stream.Stream<{item.ts}, ZynkError, ZynkClient> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"options?: CallOptions"
            f"): Stream.Stream<{item.ts}, ZynkError, ZynkClient> =>"
        )
    lines.append(
        f'  callChannel("{endpoint.name}", {args_payload}, '
        f"{item.schema}, options)"
    )
    return "\n".join(lines)


def _emit_upload(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    response = type_expr(endpoint.returns, model_names)

    extra_fields: list[str] = []
    if endpoint.multi_file:
        file_field = "files: ReadonlyArray<File>"
    else:
        file_field = "file: File"
    extra_fields.append(file_field)

    required = []
    optional = []
    for param in endpoint.params:
        expr = type_expr(param.type, model_names)
        ts = expr.ts
        if not param.required:
            ts = ts.removesuffix(" | undefined")
            optional.append(f"{param.ts_name}?: {ts}")
        else:
            required.append(f"{param.ts_name}: {ts}")
    args_fields = [file_field, *required, *optional]
    params_type = "{ " + "; ".join(args_fields) + " }"

    args_payload = _args_payload(endpoint.params)
    files_expr = "args.files" if endpoint.multi_file else "[args.file]"

    lines = list(_doc_block(endpoint.doc))
    lines.append(
        f"export const {fn_name} = ("
        f"args: {params_type}, "
        f"options?: UploadOptions"
        f"): Effect.Effect<{response.ts}, ZynkError, ZynkClient> =>"
    )
    lines.append(
        f'  callUpload("{endpoint.name}", {files_expr}, {args_payload}, '
        f"{response.schema}, options)"
    )
    return "\n".join(lines)


def _emit_static(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name) + "Url"
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines = list(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}"
            f"): Effect.Effect<string, never, ZynkClient> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"): Effect.Effect<string, never, ZynkClient> =>"
        )
    lines.append(f'  buildStaticUrl("{endpoint.name}", {args_payload})')
    return "\n".join(lines)


def _emit_websocket(endpoint: Endpoint, model_names: set[str]) -> str:
    """Emit a typed Effect-friendly WebSocket factory.

    The returned Effect yields a small ``ZynkSocket`` object that exposes
    Effect-wrapped ``send``/``on``/``close`` operations alongside the underlying
    socket. WebSocket types stay untyped on purpose because end-users can pick
    their own concurrency model on top.
    """
    name = endpoint.name
    pascal = _pascal(name)
    server_iface = f"{pascal}ServerEvents"
    client_iface = f"{pascal}ClientEvents"
    socket_type = f"{pascal}Socket"

    server_fields = [
        f"  {event}: {type_expr(ref, model_names).ts}"
        for event, ref in (endpoint.server_events or {}).items()
    ]
    client_fields = [
        f"  {event}: {type_expr(ref, model_names).ts}"
        for event, ref in (endpoint.client_events or {}).items()
    ]

    parts: list[str] = []
    parts.append(
        f"export interface {server_iface} {{\n"
        + (",\n".join(server_fields) if server_fields else "")
        + ("\n" if server_fields else "")
        + "}"
    )
    parts.append(
        f"export interface {client_iface} {{\n"
        + (",\n".join(client_fields) if client_fields else "")
        + ("\n" if client_fields else "")
        + "}"
    )
    parts.append(
        f"export interface {socket_type} {{\n"
        f"  readonly socket: WebSocket\n"
        f"  send<K extends keyof {client_iface}>(event: K, data: {client_iface}[K]): Effect.Effect<void, ZynkNetworkError>\n"
        f"  on<K extends keyof {server_iface}>(event: K): Stream.Stream<{server_iface}[K], ZynkNetworkError>\n"
        f"  close: Effect.Effect<void>\n"
        f"}}"
    )

    factory_name = f"connect{pascal}"
    parts.append(
        "\n".join(
            [
                *_doc_block(endpoint.doc),
                f"export const {factory_name} = (): Effect.Effect<"
                f"{socket_type}, ZynkNetworkError, ZynkClient> =>",
                "  Effect.gen(function* () {",
                f"    const ws = yield* openWebSocket(\"{name}\")",
                "    return {",
                "      socket: ws,",
                f"      send<K extends keyof {client_iface}>(event: K, data: {client_iface}[K]) {{",
                "        return Effect.try({",
                "          try: () => ws.send(JSON.stringify({ event, data })),",
                "          catch: (cause) =>",
                "            new ZynkNetworkError({ url: ws.url, cause }),",
                "        })",
                "      },",
                f"      on<K extends keyof {server_iface}>(event: K) {{",
                f"        return Stream.async<{server_iface}[K], ZynkNetworkError>((emit) => {{",
                "          const listener = (ev: MessageEvent) => {",
                "            try {",
                "              const parsed = JSON.parse(ev.data) as { event: string; data: unknown }",
                "              if (parsed.event === event) {",
                f"                void emit.single(parsed.data as {server_iface}[K])",
                "              }",
                "            } catch (cause) {",
                "              void emit.fail(new ZynkNetworkError({ url: ws.url, cause }))",
                "            }",
                "          }",
                "          ws.addEventListener(\"message\", listener)",
                "          return Effect.sync(() => ws.removeEventListener(\"message\", listener))",
                "        })",
                "      },",
                "      close: Effect.sync(() => ws.close()),",
                "    }",
                "  })",
            ]
        )
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Promise surface emitters
# ---------------------------------------------------------------------------


def _emit_rpc_promise(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    response = type_expr(endpoint.returns, model_names)
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines: list[str] = []
    lines.extend(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}, "
            f"options?: CallOptions"
            f"): Promise<{response.ts}> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"options?: CallOptions"
            f"): Promise<{response.ts}> =>"
        )
    lines.append(
        f'  runPromise(callCommand("{endpoint.name}", {args_payload}, '
        f"{response.schema}, options))"
    )
    return "\n".join(lines)


def _emit_channel_promise(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    item = type_expr(endpoint.channel_item, model_names) if endpoint.channel_item else type_expr(None, model_names)
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines: list[str] = []
    lines.extend(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}, "
            f"options?: CallOptions"
            f"): AsyncIterable<{item.ts}> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"options?: CallOptions"
            f"): AsyncIterable<{item.ts}> =>"
        )
    lines.append(
        f'  toAsyncIterable(callChannel("{endpoint.name}", {args_payload}, '
        f"{item.schema}, options))"
    )
    return "\n".join(lines)


def _emit_upload_promise(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name)
    response = type_expr(endpoint.returns, model_names)

    if endpoint.multi_file:
        file_field = "files: ReadonlyArray<File>"
    else:
        file_field = "file: File"

    required = []
    optional = []
    for param in endpoint.params:
        expr = type_expr(param.type, model_names)
        ts = expr.ts
        if not param.required:
            ts = ts.removesuffix(" | undefined")
            optional.append(f"{param.ts_name}?: {ts}")
        else:
            required.append(f"{param.ts_name}: {ts}")
    args_fields = [file_field, *required, *optional]
    params_type = "{ " + "; ".join(args_fields) + " }"

    args_payload = _args_payload(endpoint.params)
    files_expr = "args.files" if endpoint.multi_file else "[args.file]"

    lines = list(_doc_block(endpoint.doc))
    lines.append(
        f"export const {fn_name} = ("
        f"args: {params_type}, "
        f"options?: UploadOptions"
        f"): Promise<{response.ts}> =>"
    )
    lines.append(
        f'  runPromise(callUpload("{endpoint.name}", {files_expr}, {args_payload}, '
        f"{response.schema}, options))"
    )
    return "\n".join(lines)


def _emit_static_promise(endpoint: Endpoint, model_names: set[str]) -> str:
    fn_name = _camel(endpoint.name) + "Url"
    params_type = _params_object(endpoint.params, model_names)
    args_payload = _args_payload(endpoint.params)
    has_params = params_type != "void"

    lines = list(_doc_block(endpoint.doc))
    if has_params:
        lines.append(
            f"export const {fn_name} = ("
            f"args: {params_type}"
            f"): Promise<string> =>"
        )
    else:
        lines.append(
            f"export const {fn_name} = ("
            f"): Promise<string> =>"
        )
    lines.append(f'  runPromise(buildStaticUrl("{endpoint.name}", {args_payload}))')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module assembly
# ---------------------------------------------------------------------------


def lower_graph(
    graph: ApiGraph,
    registry: Any,
    options: EffectGeneratorOptions | None = None,
) -> EffectModule:
    opts = options or EffectGeneratorOptions()

    rpcs = [ep for ep in graph.endpoints.values() if ep.kind == "rpc"]
    channels = [ep for ep in graph.endpoints.values() if ep.kind == "channel"]
    uploads = [ep for ep in graph.endpoints.values() if ep.kind == "upload"]
    statics = [ep for ep in graph.endpoints.values() if ep.kind == "static"]
    sockets = [ep for ep in graph.endpoints.values() if ep.kind == "ws"]

    model_names: set[str] = set()

    emit_rpc = _emit_rpc_promise if opts.resolve("commands") == "promise" else _emit_rpc
    emit_channel = _emit_channel_promise if opts.resolve("channels") == "promise" else _emit_channel
    emit_upload = _emit_upload_promise if opts.resolve("uploads") == "promise" else _emit_upload
    emit_static = _emit_static_promise if opts.resolve("statics") == "promise" else _emit_static

    rpcs_src = [emit_rpc(ep, model_names) for ep in sorted(rpcs, key=lambda e: e.name)]
    channels_src = [emit_channel(ep, model_names) for ep in sorted(channels, key=lambda e: e.name)]
    uploads_src = [emit_upload(ep, model_names) for ep in sorted(uploads, key=lambda e: e.name)]
    statics_src = [emit_static(ep, model_names) for ep in sorted(statics, key=lambda e: e.name)]
    sockets_src = [_emit_websocket(ep, model_names) for ep in sorted(sockets, key=lambda e: e.name)]

    for name in graph.models:
        model_names.add(name)

    resolved: dict[str, tuple[str, str]] = {}
    pending = set(model_names)
    while pending:
        current = pending.pop()
        model_def = graph.models.get(current)
        if not model_def:
            continue
        py_type = model_def.py_type
        schema, alias = render_model_schema(py_type, model_names)
        resolved[current] = (schema, alias)
        pending |= model_names - set(resolved.keys())

    model_sections: list[str] = []
    for name in sorted(resolved):
        schema, alias = resolved[name]
        model_sections.append(schema)
        model_sections.append(alias)

    needs_stream = bool(channels_src or sockets_src)
    needs_promise = any(
        opts.resolve(k) == "promise"
        for k in ("commands", "uploads", "statics", "channels")
    )

    internal_imports = ['  callCommand,']
    if channels_src:
        internal_imports.append('  callChannel,')
    if uploads_src:
        internal_imports.append('  callUpload,')
    if statics_src:
        internal_imports.append('  buildStaticUrl,')
    if sockets_src:
        internal_imports.append('  openWebSocket,')
    if needs_promise:
        internal_imports.append('  runPromise,')
    if opts.resolve("channels") == "promise" and channels_src:
        internal_imports.append('  toAsyncIterable,')
    internal_imports.append('  type CallOptions,')
    if uploads_src:
        internal_imports.append('  type UploadOptions,')
    internal_imports.append('  type ZynkClient,')
    internal_imports.append('  type ZynkError,')
    if sockets_src:
        internal_imports.append('  ZynkNetworkError,')

    imports = [
        'import {',
        '  Effect,',
        '  Schema,'
        + ("\n  Stream," if needs_stream else ""),
        '} from "effect"',
        'import {\n' + '\n'.join(internal_imports) + '\n} from "./_effect_internal"',
        'export {',
        '  ZynkClient,',
        '  initZynk,',
        '  layerZynkClient,',
        '  type CallOptions,'
        + ("\n  type UploadOptions," if uploads_src else ""),
        '  type RetryOptions,',
        '  type BackoffStrategy,',
        '  type ZynkClientConfig,',
        '  type ZynkError,',
        '  ZynkNetworkError,',
        '  ZynkHttpError,',
        '  ZynkTimeoutError,',
        '  ZynkDecodeError,',
        '  ZynkAbortError,',
        '  ZynkStreamError,',
        '  ZynkUploadError,',
        '} from "./_effect_internal"',
    ]

    body: list[str] = []
    if model_sections:
        body.append("// ============ Schemas ============\n\n" + "\n\n".join(model_sections))
    if rpcs_src:
        body.append("// ============ Commands ============\n\n" + "\n\n".join(rpcs_src))
    if channels_src:
        body.append("// ============ Channels ============\n\n" + "\n\n".join(channels_src))
    if uploads_src:
        body.append("// ============ Uploads ============\n\n" + "\n\n".join(uploads_src))
    if statics_src:
        body.append("// ============ Static Files ============\n\n" + "\n\n".join(statics_src))
    if sockets_src:
        body.append("// ============ WebSockets ============\n\n" + "\n\n".join(sockets_src))

    return EffectModule(
        banner=(
            f"/* Auto-generated by Zynk Effect connector - DO NOT EDIT */\n"
            f"/* Generated: {datetime.now().isoformat()} */"
        ),
        imports=imports,
        body=body,
    )


# Re-export for tests/utilities.
__all__ = ["EffectModule", "lower_graph", "collect_model_refs"]
