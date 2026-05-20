use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use zynk_runtime::{HandlerKey, TypeRefStatic};

static PING_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "user_id",
    wire_name: "userId",
    ty: TypeRefStatic::primitive("number"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "introspection_ping",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("introspection_routes"),
        doc: Some("Introspection test command."),
        params: PING_PARAMS,
        returns: TypeRefStatic::primitive("string"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("introspection_routes::introspection_ping")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "introspection_stream",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("introspection_routes"),
        doc: Some("Introspection test channel."),
        params: &[],
        returns: TypeRefStatic::void(),
        channel_item: Some(TypeRefStatic::primitive("string")),
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("introspection_routes::introspection_stream")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "introspection_socket",
        kind: zynk_runtime::EndpointKind::Ws,
        module: Some("introspection_routes"),
        doc: None,
        params: &[],
        returns: TypeRefStatic::void(),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("introspection_routes::introspection_socket")),
    }
}

async fn get_json(router: Router, path: &str) -> (StatusCode, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(path)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body collected")
        .to_bytes();
    let json = serde_json::from_slice(&body).expect("response is json");
    (status, json)
}

#[tokio::test]
async fn root_and_commands_routes_match_python_introspection_shape() {
    let router = zynk_axum::ZynkBridge::new().configure(Router::new());

    let (status, body) = get_json(router.clone(), "/").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["bridge"], "Zynk API");
    assert_eq!(
        body["commands"],
        json!(["introspection_ping", "introspection_stream"])
    );

    let (status, body) = get_json(router, "/commands").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        json!({
            "commands": [
                {
                    "name": "introspection_ping",
                    "module": "introspection_routes",
                    "has_channel": false,
                    "params": ["user_id"]
                },
                {
                    "name": "introspection_stream",
                    "module": "introspection_routes",
                    "has_channel": true,
                    "params": []
                }
            ]
        })
    );
}
