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

Before publishing the CLI, verify the crate metadata and package contents from the repository root:

```bash
cargo fmt -p zynk-cli -- --check
cargo clippy -p zynk-cli --all-targets -- -D warnings
cargo test -p zynk-cli --all-targets
cargo publish --dry-run -p zynk-cli
```

If the dry run succeeds, publish the crate explicitly:

```bash
cargo publish -p zynk-cli
```

Do not use `--allow-dirty` for a release. Confirm that `Cargo.toml` has the intended `version`, `license`, `repository`, `description`, `keywords`, `categories`, `readme`, and `[[bin]] name = "zynk"` metadata before running the final publish command.
