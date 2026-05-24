# zynk-axum

Axum server binding for Zynk Rust endpoints.

`ZynkBridge` consumes endpoint metadata collected through `inventory` and registers Zynk wire-contract routes on an existing Axum router. It supports all Zynk endpoint kinds: commands (request/response), channels (SSE streams), uploads (multipart), static files, and websockets.

## Usage

```toml
[dependencies]
zynk-axum = "0.1"
zynk-macros = "0.1"
zynk-runtime = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust,ignore
use zynk_axum::ZynkBridge;

#[tokio::main]
async fn main() {
    let bridge = ZynkBridge::new("My API");
    let app = bridge.into_router();
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## License

[MIT](https://github.com/SethBurkart123/zynk/blob/main/LICENSE)
