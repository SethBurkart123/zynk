# Zynk

A type-safe Python/Rust → TypeScript framework. Write your backend once in either language and ship a fully typed TypeScript client that talks to it.

## What is Zynk?

Zynk v1 is a multi-language framework built around a shared API graph. A Rust codegen core (`zynk-codegen`, `zynk-gen-ts`, `zynk-gen-effect`) consumes a canonical schema and emits zero-dependency TypeScript clients. Backends speak that same wire contract from either a Python runtime (FastAPI under the hood) or a Rust runtime (Axum), so one generated client works against either server.

## Why

- **Cross-language backends, one wire contract.** Python and Rust servers serialize the same payloads byte-for-byte. The same generated TypeScript client switches between them by changing a base URL.
- **Type safety end to end.** Pydantic models, Rust structs, enums, generics, channels, websockets, file uploads, and error envelopes all survive the trip into TypeScript.
- **Zero-dependency client output.** Generated TypeScript ships with no runtime deps. An optional Effect target is available for `effect/Schema` users.
- **Standalone CLI.** Codegen lives in a Rust binary (`zynk`) that is shipped inside the Python wheel and also installable on its own.
- **Hot reload friendly.** `zynk dev` watches your Python source and regenerates the client whenever the schema changes.

## Install

For Python projects (gets both the server runtime and the `zynk` CLI binary):

```bash
pip install zynk
```

For Rust-only consumers, or to install just the CLI:

```bash
cargo install zynk-cli
```

Both routes install the same `zynk` binary on your `PATH`.

## Quick start (Python)

Define commands and a `Bridge`:

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

Generate the TypeScript client:

```bash
zynk gen typescript --target python --app main:bridge --out ./frontend/src/generated
```

Use it from the frontend:

```typescript
import { initBridge, getUser } from "./generated/api";

initBridge("http://127.0.0.1:8000");

const user = await getUser({ userId: 123 });
console.log(user.name); // string
```

Streaming uses `Channel`:

```python
from zynk import Channel, command
import asyncio

@command
async def stream_updates(channel: Channel[dict]) -> None:
    for i in range(10):
        await channel.send({"count": i})
        await asyncio.sleep(1)
```

```typescript
const channel = streamUpdates({});
channel.subscribe((data) => console.log(data));
channel.onClose(() => console.log("done"));
```

## Rust backend

The same `Bridge` concept exists in Rust through the `zynk-macros`, `zynk-runtime`, and `zynk-axum` crates. Handlers are tagged with attribute macros and registered on a `ZynkBridge` that configures an Axum router:

```rust
use axum::Router;
use serde::{Deserialize, Serialize};
use zynk_axum::ZynkBridge;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub email: String,
}

#[zynk::command]
pub async fn get_user(user_id: i64) -> User {
    User { id: user_id, name: "Alice".into(), email: "alice@example.com".into() }
}

pub fn router() -> Router {
    ZynkBridge::new()
        .title("Users API")
        .configure(Router::new())
}
```

Generate the TypeScript client against the Rust binary:

```bash
zynk gen typescript --target rust --app rust-axum-kitchen-sink --out ./frontend/src/generated
```

The output is the same shape as the Python target, so a single generated client can be pointed at either server.

## CLI reference

| Command | Purpose |
| --- | --- |
| `zynk gen typescript --target <python\|rust> --app <app> --out <dir>` | One-shot TypeScript client generation. |
| `zynk gen effect --target <python\|rust> --app <app> --out <dir>` | Generate an `effect/Schema` client. |
| `zynk dev --app <app> --out <dir>` | Watch the source app and regenerate the client on every change. |
| `zynk schema --app <app>` | Dump the canonical API graph JSON to stdout for inspection. |

`<app>` is `module:bridge` for Python (e.g. `main:bridge`) and the Cargo bin name or path to a Rust binary for `--target rust`.

## Workspace layout

- `crates/` — Rust workspace:
  - `zynk-schema` — canonical API graph types.
  - `zynk-codegen`, `zynk-gen-ts`, `zynk-gen-effect` — generator core and language targets.
  - `zynk-runtime`, `zynk-macros`, `zynk-axum` — Rust server runtime, attribute macros, and Axum integration.
  - `zynk-cli` — the `zynk` binary.
- `bindings/python/` — the `zynk` Python package: `Bridge`, `@command`, `Channel`, registry, FastAPI runtime. The published wheel bundles the Rust `zynk` CLI binary so `pip install zynk` is enough.
- `examples/` — runnable examples (see below).

## Examples

- `examples/kitchen-sink/` — Python FastAPI backend on port 8100 with a Bun/Vite frontend and a Bun-based wire-parity harness. Covers users, tasks, weather streaming, uploads, chat, and websockets.
- `examples/rust-axum-kitchen-sink/` — A Rust Axum port of the same kitchen sink on port 8101, sharing the wire contract so the same generated TypeScript client works against either backend.

## Development

Run the full test matrix from the repo root:

```bash
# Rust workspace (codegen, runtime, macros, axum, CLI)
cargo test --workspace

# Python bindings
cd bindings/python && uv run pytest

# Cross-server wire-parity harness
cd examples/kitchen-sink/harness && bun test
```

## License

[MIT](./LICENSE)
