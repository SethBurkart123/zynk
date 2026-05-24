# Zynk

Python server runtime for [Zynk](https://github.com/SethBurkart123/zynk) -- a general-purpose command protocol for building type-safe bridges between backends and frontends.

## Install

```bash
pip install zynk
```

The `zynk` CLI is bundled in platform wheels. If no prebuilt binary is available for your platform, install the standalone CLI with:

```bash
cargo install zynk-cli
```

## Quick start

```python
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

Generate a TypeScript client:

```bash
zynk gen typescript --target python --app main:bridge --out ./frontend/src/generated
```

Use it from the frontend:

```typescript
import { initBridge, getUser } from "./generated/api";

initBridge("http://127.0.0.1:8000");

const user = await getUser({ userId: 123 });
console.log(user.name);
```

## Features

### Commands (request/response)

```python
from zynk import command

@command
async def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### Channels (server-streamed values)

```python
import asyncio
from zynk import command, Channel

@command
async def stream_data(channel: Channel[dict]) -> None:
    for i in range(10):
        await channel.send({"value": i})
        await asyncio.sleep(0.1)
```

### WebSockets (bidirectional)

```python
from pydantic import BaseModel
from zynk import message, WebSocket

class ChatMessage(BaseModel):
    text: str

class ServerEvents:
    chat_message: ChatMessage

class ClientEvents:
    chat_message: ChatMessage

@message
async def chat(ws: WebSocket[ServerEvents, ClientEvents]) -> None:
    @ws.on("chat_message")
    async def on_message(data: ChatMessage):
        await ws.send("chat_message", data)

    await ws.listen()
```

### File uploads

```python
from pydantic import BaseModel
from zynk import upload, UploadFile

class UploadResult(BaseModel):
    filename: str
    size: int

@upload(max_size="10MB", allowed_types=["image/*", "application/pdf"])
async def upload_file(file: UploadFile) -> UploadResult:
    content = await file.read()
    return UploadResult(filename=file.filename, size=file.size)
```

### Static files

```python
from zynk import static, StaticFile

@static
async def user_avatar(user_id: str) -> StaticFile:
    return StaticFile.from_directory(base_dir="./avatars", filename=f"{user_id}.png")
```

## How it works

Zynk defines a stable wire protocol (HTTP + SSE + WebSocket with JSON envelopes). The Python package provides the server runtime (`Bridge`, decorators, type extraction), and the `zynk` CLI reads the schema IR to generate type-safe clients in TypeScript or Effect-TS.

Any generated client works with any conforming host -- a TypeScript client generated from a Python host works unchanged against a Rust Axum host.

## Links

- [Repository](https://github.com/SethBurkart123/zynk)
- [Rust crates](https://crates.io/crates/zynk-schema)
- [CLI (`cargo install zynk-cli`)](https://crates.io/crates/zynk-cli)

## License

[MIT](https://github.com/SethBurkart123/zynk/blob/main/LICENSE)
