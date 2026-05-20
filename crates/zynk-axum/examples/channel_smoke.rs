use axum::Router;
use serde_json::{json, Value};
use zynk_runtime::{HandlerKey, TypeRefStatic};

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "smoke_channel",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("channel_smoke"),
        doc: None,
        params: &[],
        returns: TypeRefStatic::void(),
        channel_item: Some(TypeRefStatic::primitive("string")),
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("channel_smoke::smoke_channel")),
    }
}

#[tokio::main]
async fn main() {
    let bridge = zynk_axum::ZynkBridge::new().register_channel(
        HandlerKey("channel_smoke::smoke_channel"),
        |_payload: Value, channel: zynk_axum::Channel| async move {
            channel.send(json!({ "count": 1 }))?;
            channel.send(json!("done"))?;
            Ok(())
        },
    );
    let app = bridge.configure(Router::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8101")
        .await
        .expect("bind smoke server");
    axum::serve(listener, app).await.expect("serve smoke app");
}
