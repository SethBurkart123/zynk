# Zynk

A general-purpose command abstraction — bring your own host and client, in any language you prefer. You implement the parts (host binding + client generator) and then magic happens :)

## What is Zynk?

Zynk is a small, stable wire protocol for the things real apps actually do between a backend and a frontend: commands (request/response), channels (server-streamed values), uploads, static files, and websockets. It runs over plain HTTP + SSE + WS, with snake- or camel-case JSON, structured error envelopes, and an explicit schema (the IR) describing every endpoint and type.

That IR is the contract. Any **host binding** that serves the protocol can talk to any **client generator** that consumes the IR — regardless of the languages on either end. A TypeScript client generated from a Python host works unchanged against a Rust Axum host, byte-for-byte, because they both speak Zynk on the wire. Implement a host binding for your backend stack, or a generator for your frontend stack, and you instantly interop with everything else in the ecosystem.

## How it works

1. You define commands, channels, uploads, static endpoints, and websockets once — this populates the IR (`crates/zynk-schema`).
2. A **host binding** in your backend language serves them over the Zynk wire protocol (HTTP + SSE + WS, with the documented JSON shapes and error envelopes).
3. A **client generator** for your frontend language reads the IR and emits a type-safe client over the same wire shapes.
4. Any conforming client talks to any conforming host. That's it.

## Bring your own host

A host binding is responsible for serving the Zynk wire protocol from your backend framework of choice. It must:

- Accept the documented routes for each `EndpointKind` (`rpc`, `channel`, `upload`, `static`, `ws`).
- Decode params using either snake_case or camelCase JSON, and serialize responses, channel items, and WS events the same way.
- Emit Zynk's structured error envelopes (validation, not-found, internal, etc.) so generated clients can surface them in a typed way.
- Expose the IR for the running app so a client generator can pull it (typically over a `--app` flag or a local introspection endpoint).

The Rust + Axum host at [`crates/zynk-axum/src/`](./crates/zynk-axum/src/) is the canonical reference. It's a single-file binding (`lib.rs`) that turns a set of registered handlers into a fully conforming Axum router — useful both as documentation of the protocol and as a worked example.

## Bring your own client

A client generator consumes the IR — `ApiGraph` from [`crates/zynk-schema`](./crates/zynk-schema/src/), which is just `Endpoint`s plus model and enum definitions — and emits an idiomatic, type-safe client in your target language. Concretely it needs to:

- Produce typed function signatures for each `Endpoint`, keyed off `EndpointKind`.
- Translate the IR's `TypeRef`s (primitives, models, enums, options, arrays, records, literals, generics) into the target language's type system.
- Speak the same wire shapes as any host: command JSON, SSE channel framing, upload multipart, static URLs, and WS event envelopes.
- Pass through Zynk's error envelope so callers get structured errors, not stringly-typed failures.

The plain TypeScript generator at [`crates/zynk-gen-ts/src/`](./crates/zynk-gen-ts/src/) is the canonical reference — zero-runtime-dependency output, with a parallel Effect-TS target at [`crates/zynk-gen-effect/src/`](./crates/zynk-gen-effect/src/) showing the same IR retargeted at a different ecosystem.

## Bundled reference implementations

The repo ships two reference hosts and two reference clients, all speaking the same protocol:

| Role | Language / Stack | Location |
| --- | --- | --- |
| Host | Python (FastAPI) | [`bindings/python`](./bindings/python) |
| Host | Rust (Axum) | [`crates/zynk-axum`](./crates/zynk-axum) |
| Client | TypeScript (zero deps) | [`crates/zynk-gen-ts`](./crates/zynk-gen-ts) |
| Client | Effect-TS (`effect/Schema`) | [`crates/zynk-gen-effect`](./crates/zynk-gen-effect) |
| Shared IR | — | [`crates/zynk-schema`](./crates/zynk-schema) |
| Codegen core | — | [`crates/zynk-codegen`](./crates/zynk-codegen) |
| CLI (`zynk`) | — | [`crates/zynk-cli`](./crates/zynk-cli) |

The `zynk` CLI ships via both `pip install zynk` (bundled inside the Python wheel) and `cargo install zynk-cli`. Either route puts the same binary on your `PATH`, and it drives schema dump, codegen, and watch mode against any conforming host.

## Quick start with the reference stack

Define a command on a Python host:

```python
# main.py
from pydantic import BaseModel
from zynk import Bridge, command

class User(BaseModel):
    id: int
    name: str
    email: str

@command
async def get_user(user_id: int) -> User:
    return User(id=user_id, name="Alice", email="alice@example.com")

bridge = Bridge(title="Users API", port=8000)

if __name__ == "__main__":
    bridge.run(dev=True)
```

Generate a TypeScript client from the IR:

```bash
zynk gen typescript --target python --app main:bridge --out ./frontend/src/generated
```

Call it from the frontend:

```typescript
import { initBridge, getUser } from "./generated/api";

initBridge("http://127.0.0.1:8000");

const user = await getUser({ userId: 123 });
console.log(user.name);
```

And yes — the same generated TS client works unchanged against the Rust Axum host on a different port. Same wire protocol, same IR, different language. That's the magic.

## Repo layout

- `crates/` — Rust workspace: shared IR (`zynk-schema`), codegen core (`zynk-codegen`), language targets (`zynk-gen-ts`, `zynk-gen-effect`), the Rust host binding (`zynk-runtime`, `zynk-macros`, `zynk-axum`), and the `zynk` CLI (`zynk-cli`).
- `bindings/python/` — the Python host binding (`Bridge`, `@command`, `Channel`, FastAPI runtime). The published wheel bundles the `zynk` CLI binary.
- `examples/` — runnable demonstrations of cross-language interop.
- `scripts/`, `dist/` — packaging and release plumbing.

## Examples

- [`examples/kitchen-sink`](./examples/kitchen-sink) — Python FastAPI host on port 8100, a Bun/Vite frontend driven by the generated TS client, and a wire-parity harness.
- [`examples/rust-axum-kitchen-sink`](./examples/rust-axum-kitchen-sink) — A Rust Axum host of the same kitchen sink on port 8101.

The same generated TypeScript client is pointed at both hosts and the harness checks byte-for-byte parity — a working proof that "one IR, many hosts, many clients" isn't just aspirational.

## Development

```bash
# Rust workspace (IR, codegen, runtime, macros, axum, CLI)
cargo test --workspace

# Python host binding
cd bindings/python && uv run pytest

# Cross-host wire-parity harness
cd examples/kitchen-sink/harness && bun test
```

## License

[MIT](./LICENSE)
