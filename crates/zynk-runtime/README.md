# zynk-runtime

Pure wire-contract runtime types for Zynk server bindings.

This crate contains the framework-agnostic data structures, error envelopes, handler traits, and endpoint metadata that server bindings (such as [`zynk-axum`](https://crates.io/crates/zynk-axum)) use to register endpoints, dispatch type-erased handlers, and preserve Zynk's JSON/SSE/WebSocket wire shapes.

It deliberately contains no HTTP framework integration — all IO is left to the concrete server binding.

## Usage

```toml
[dependencies]
zynk-runtime = "0.1"
```

Most users will not depend on this crate directly. Instead, use a server binding like `zynk-axum` together with `zynk-macros` to register endpoints via attribute macros.

## License

[MIT](https://github.com/SethBurkart123/zynk/blob/main/LICENSE)
