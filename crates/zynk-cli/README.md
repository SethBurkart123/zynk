# zynk-cli

`zynk-cli` is the standalone Rust command-line interface for Zynk. It installs a `zynk` binary that inspects schema-capable Python or Rust apps and generates TypeScript clients.

## Install

After publication, install the CLI from crates.io:

```bash
cargo install zynk-cli
zynk --version
```

## Common commands

Generate a plain TypeScript client from a Python bridge:

```bash
zynk gen typescript --target python --out ./frontend/src/api.ts --app main:app
```

Generate an Effect client:

```bash
zynk gen effect --target python --out ./frontend/src/api.ts --app main:app
```

Print the canonical API graph JSON for debugging:

```bash
zynk schema dump --app main:app
```

Regenerate on source changes during development:

```bash
zynk dev --out ./frontend/src/api.ts --app main:app
```
