use axum::Router;
use serde_json::{json, Value};
use zynk_runtime::{HandlerKey, TypeRefStatic};

static PING_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "name",
    wire_name: "name",
    ty: TypeRefStatic::primitive("string"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "ping",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("command_smoke"),
        doc: None,
        params: PING_PARAMS,
        returns: TypeRefStatic::primitive("string"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("command_smoke::ping")),
    }
}

#[tokio::main]
async fn main() {
    let bridge = zynk_axum::ZynkBridge::new().register_handler(
        HandlerKey("command_smoke::ping"),
        |payload: Value| {
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("world");
            Ok(json!(format!("pong:{name}")))
        },
    );
    let app = bridge.configure(Router::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8101")
        .await
        .expect("bind smoke server");
    axum::serve(listener, app).await.expect("serve smoke app");
}
