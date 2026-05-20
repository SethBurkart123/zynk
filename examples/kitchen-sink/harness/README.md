# Kitchen Sink Bun Harness

This harness validates the generated TypeScript client against the live Python kitchen-sink backend.

## Run

```bash
cd examples/kitchen-sink/harness
bun install
bun run test
```

The test command spawns the Python backend on port `8100`, waits for the health endpoint, exercises commands, channels, uploads, statics, and WebSockets through `../frontend/src/generated/api.ts`, then tears the server down. Each run writes evidence to `logs/<timestamp>/`, including harness events, raw request/response payloads, and server stdout/stderr.

For TypeScript-only validation:

```bash
bun run typecheck
```
