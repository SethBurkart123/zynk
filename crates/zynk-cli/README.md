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

## Publish workflow

Publishing to crates.io is a manual maintainer step. `zynk-cli` depends on the Zynk schema and generator crates, so publish those dependency crates first, in dependency order, before publishing the CLI:

1. `zynk-schema`
2. `zynk-codegen`
3. `zynk-gen-ts`
4. `zynk-gen-effect`
5. `zynk-cli`

Before publishing, verify the crate metadata and package contents from the repository root with `cargo package` for every crate in the release set:

```bash
cargo package -p zynk-schema
cargo package -p zynk-codegen
cargo package -p zynk-gen-ts
cargo package -p zynk-gen-effect
cargo package -p zynk-cli
```

Publish each crate explicitly in dependency order, waiting for crates.io to index each dependency before publishing the next crate:

```bash
cargo publish -p zynk-schema
cargo publish -p zynk-codegen
cargo publish -p zynk-gen-ts
cargo publish -p zynk-gen-effect
cargo publish -p zynk-cli
```

Do not use `--allow-dirty` for a release. Confirm that each `Cargo.toml` has the intended `version`, `license`, `repository`, `description`, `keywords`, `categories`, and `readme` metadata before running the final publish command. For `zynk-cli`, also confirm `package.name = "zynk-cli"` and `[[bin]] name = "zynk"`.
