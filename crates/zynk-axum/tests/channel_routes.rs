use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::time::Duration;
use tower::ServiceExt;
use zynk_runtime::{HandlerKey, TypeRefStatic, ZynkError, EXECUTION_ERROR};

static LIMIT_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "limit",
    wire_name: "limit",
    ty: TypeRefStatic::primitive("number"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "channel_numbers",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("channel_routes"),
        doc: Some("Stream numeric updates."),
        params: LIMIT_PARAMS,
        returns: TypeRefStatic::void(),
        channel_item: Some(TypeRefStatic::primitive("number")),
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("channel_routes::channel_numbers")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "idle_channel",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("channel_routes"),
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
        handler_key: Some(HandlerKey("channel_routes::idle_channel")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "error_channel",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("channel_routes"),
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
        handler_key: Some(HandlerKey("channel_routes::error_channel")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "rpc_only",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("channel_routes"),
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
        handler_key: Some(HandlerKey("channel_routes::rpc_only")),
    }
}

fn test_bridge() -> zynk_axum::ZynkBridge {
    zynk_axum::ZynkBridge::new()
        .keepalive_interval(Duration::from_millis(25))
        .register_channel(
            HandlerKey("channel_routes::channel_numbers"),
            |payload: Value, channel: zynk_axum::Channel| async move {
                let limit = payload
                    .get("limit")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| ZynkError::new(zynk_runtime::VALIDATION_ERROR, "bad limit"))?;
                for value in 0..limit {
                    channel.send(json!({ "value": value }))?;
                }
                Ok(())
            },
        )
        .register_channel(
            HandlerKey("channel_routes::idle_channel"),
            |_payload: Value, channel: zynk_axum::Channel| async move {
                tokio::time::sleep(Duration::from_millis(70)).await;
                channel.send(json!("done"))?;
                Ok(())
            },
        )
        .register_channel(
            HandlerKey("channel_routes::error_channel"),
            |_payload: Value, channel: zynk_axum::Channel| async move {
                channel.send(json!("before failure"))?;
                Err(ZynkError::new(EXECUTION_ERROR, "handler failed"))
            },
        )
}

async fn post(router: Router, path: &str, body: &str) -> axum::response::Response {
    router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
                .expect("request builds"),
        )
        .await
        .expect("router responds")
}

async fn body_text(response: axum::response::Response) -> String {
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body collected")
        .to_bytes();
    String::from_utf8(body.to_vec()).expect("utf-8 body")
}

#[tokio::test]
async fn channel_route_streams_python_compatible_sse_frames_and_headers() {
    let router = test_bridge().configure(Router::new());

    let response = post(router, "/channel/channel_numbers", r#"{"limit":2}"#).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");
    assert_eq!(response.headers().get("connection").unwrap(), "keep-alive");
    assert_eq!(response.headers().get("x-accel-buffering").unwrap(), "no");

    let body = body_text(response).await;
    assert!(body.starts_with(concat!(
        "event: message\n",
        "data: {\"value\": 0}\n\n",
        "event: message\n",
        "data: {\"value\": 1}\n\n",
        "event: close\n",
        "data: {\"channelId\": \"channel-",
    )));
    assert!(body.ends_with("\"}\n\n"));
}

#[tokio::test]
async fn channel_route_emits_keepalives_during_idle_periods() {
    let router = test_bridge().configure(Router::new());

    let response = post(router, "/channel/idle_channel", "{}").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;

    let keepalive_count = body.matches("event: keepalive\ndata: {}\n\n").count();
    assert!(
        keepalive_count >= 2,
        "expected at least two keepalives in body: {body:?}"
    );
    assert!(body.contains("event: message\ndata: \"done\"\n\n"));
    assert!(body.contains("event: close\ndata: {\"channelId\": "));
}

#[tokio::test]
async fn channel_route_error_event_flushes_queued_data_then_terminates_without_close_frame() {
    let router = test_bridge().configure(Router::new());

    let response = post(router, "/channel/error_channel", "{}").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;

    assert_eq!(
        body,
        concat!(
            "event: message\n",
            "data: \"before failure\"\n\n",
            "event: error\n",
            "data: {\"error\": \"handler failed\"}\n\n",
        ),
    );
    assert!(!body.contains("event: close"));
}

#[tokio::test]
async fn channel_route_unknown_or_non_channel_returns_command_not_found() {
    let router = test_bridge().configure(Router::new());

    let response = post(router.clone(), "/channel/no_such_channel", "{}").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await).expect("json error body");
    assert_eq!(body["code"], "COMMAND_NOT_FOUND");
    assert_eq!(body["details"], json!({"command": "no_such_channel"}));

    let response = post(router, "/channel/rpc_only", "{}").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await).expect("json error body");
    assert_eq!(body["code"], "COMMAND_NOT_FOUND");
}
