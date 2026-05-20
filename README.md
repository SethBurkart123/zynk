# Zynk

A multi-language bridge for typed backends and TypeScript frontends.

Define endpoints in Python today, dump a schema through the Rust CLI, and generate fully typed TypeScript clients for your frontend.

## Installation

```bash
pip install zynk
```

## Usage

Define commands with the `@command` decorator:

```python
# commands.py
from pydantic import BaseModel
from zynk import command

class User(BaseModel):
    id: int
    name: str
    email: str

@command
async def get_user(user_id: int) -> User:
    return User(id=user_id, name="Alice", email="alice@example.com")
```

Create a bridge and run it:

```python
# main.py
from zynk import Bridge
import commands  # registers commands on import

app = Bridge(title="Users API", port=8000)

if __name__ == "__main__":
    app.run(dev=True)
```

Generate the TypeScript client with the CLI:

```bash
zynk gen typescript --target python --out ./frontend/src/generated --app main:app
```

During development, run the watcher so the client is regenerated when your Python schema changes:

```bash
zynk dev --target python --out ./frontend/src/generated --app main:app
```

Use the generated TypeScript client:

```typescript
import { initBridge, getUser } from './generated/api';

initBridge('http://127.0.0.1:8000');

const user = await getUser({ userId: 123 });
console.log(user.name); // fully typed
```

## Multi-language architecture

Zynk v1 separates runtime and code generation responsibilities:

- The Python `zynk` library runs the FastAPI bridge and exposes `Bridge.dump_schema_json()` for schema introspection.
- The Rust `zynk` CLI owns client generation (`zynk gen`) and development-time regeneration (`zynk dev`).
- The schema format is shared so future server bindings, including Rust Axum, can generate the same TypeScript clients and speak the same wire protocol.

## Streaming

Use `Channel` to stream data to clients:

```python
from zynk import command, Channel
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

## License

MIT
