# zynk-macros

Proc macros for registering Rust Zynk endpoints.

This crate provides attribute macros (`#[command]`, `#[message]`, `#[upload]`, `#[static_file]`) that preserve your handler function unchanged and add link-time `EndpointMeta` registrations consumed by server bindings like [`zynk-axum`](https://crates.io/crates/zynk-axum).

## Usage

```toml
[dependencies]
zynk-macros = "0.1"
zynk-runtime = "0.1"
```

```rust,ignore
use zynk_macros::command;

#[command]
async fn greet(name: String) -> String {
    format!("Hello, {name}!")
}
```

## License

[MIT](https://github.com/SethBurkart123/zynk/blob/main/LICENSE)
