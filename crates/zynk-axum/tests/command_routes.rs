use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use zynk_runtime::{HandlerKey, TypeRefStatic, ZynkError, EXECUTION_ERROR};

static USER_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "user_id",
    wire_name: "userId",
    ty: TypeRefStatic::primitive("number"),
    required: true,
    default: None,
}];

static CHAT_ID_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "body",
    wire_name: "body",
    ty: TypeRefStatic::model("ChatId"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "get_user",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("command_routes"),
        doc: Some("Fetch a test user."),
        params: USER_PARAMS,
        returns: TypeRefStatic::model("User"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("command_routes::get_user")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "get_chat",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("command_routes"),
        doc: Some("Fetch a chat by flat model body."),
        params: CHAT_ID_PARAMS,
        returns: TypeRefStatic::model("ChatId"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("command_routes::get_chat")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "fail_command",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("command_routes"),
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
        handler_key: Some(HandlerKey("command_routes::fail_command")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "panic_command",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("command_routes"),
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
        handler_key: Some(HandlerKey("command_routes::panic_command")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "events",
        kind: zynk_runtime::EndpointKind::Channel,
        module: Some("command_routes"),
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
        handler_key: None,
    }
}

fn test_bridge() -> zynk_axum::ZynkBridge {
    zynk_axum::ZynkBridge::new()
        .register_handler(HandlerKey("command_routes::get_user"), |payload: Value| {
            let id = payload
                .get("user_id")
                .or_else(|| payload.get("userId"))
                .and_then(Value::as_i64)
                .ok_or_else(|| ZynkError::new(zynk_runtime::VALIDATION_ERROR, "bad id"))?;
            Ok(json!({"id": id, "name": "ada"}))
        })
        .register_handler(HandlerKey("command_routes::get_chat"), |payload: Value| {
            let id = payload
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| ZynkError::new(zynk_runtime::VALIDATION_ERROR, "bad id"))?;
            Ok(json!({"id": id}))
        })
        .register_handler(
            HandlerKey("command_routes::fail_command"),
            |_payload: Value| Err(ZynkError::new(EXECUTION_ERROR, "handler failed")),
        )
        .register_handler(
            HandlerKey("command_routes::panic_command"),
            |_payload: Value| panic!("boom"),
        )
}

async fn post_json(router: Router, path: &str, body: &str) -> (StatusCode, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
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
async fn command_route_returns_success_envelope_and_accepts_camel_or_snake_keys() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_json(router.clone(), "/command/get_user", r#"{"userId":7}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"result": {"id": 7, "name": "ada"}}));

    let (status, body) = post_json(router, "/command/get_user", r#"{"user_id":8}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"result": {"id": 8, "name": "ada"}}));
}

#[tokio::test]
async fn command_route_accepts_flat_single_model_body_without_wrapper_key() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_json(router, "/command/get_chat", r#"{"id":"chat-1"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"result": {"id": "chat-1"}}));
}

#[tokio::test]
async fn command_route_returns_validation_error_for_bad_json_and_missing_required_param() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_json(router.clone(), "/command/get_user", "{").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
    assert!(body.get("details").is_none());

    let (status, body) = post_json(router, "/command/get_user", "{}").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
    assert_eq!(body["details"], json!({"parameter": "user_id"}));
}

#[tokio::test]
async fn command_route_returns_not_found_for_unknown_or_non_rpc_endpoint() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_json(router.clone(), "/command/does_not_exist", "{}").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "COMMAND_NOT_FOUND");
    assert_eq!(body["details"], json!({"command": "does_not_exist"}));

    let (status, body) = post_json(router, "/command/events", "{}").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "COMMAND_NOT_FOUND");
}

#[tokio::test]
async fn command_route_returns_server_errors_for_handler_error_and_panic() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_json(router.clone(), "/command/fail_command", "{}").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body,
        json!({"code": "EXECUTION_ERROR", "message": "handler failed"})
    );

    let (status, body) = post_json(router, "/command/panic_command", "{}").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body,
        json!({"code": "INTERNAL_ERROR", "message": "An internal error occurred"})
    );
}

#[test]
fn dump_schema_json_returns_canonical_apigraph_with_inventory_endpoints() {
    let bridge = test_bridge();
    let dumped = bridge.dump_schema_json();
    let graph: zynk_runtime::zynk_schema::ApiGraph =
        serde_json::from_str(&dumped).expect("schema dump deserializes");

    assert!(graph.endpoints.contains_key("get_user"));
    assert!(graph.endpoints.contains_key("events"));
    let get_user = graph.endpoints.get("get_user").expect("get_user schema");
    assert_eq!(get_user.kind, zynk_runtime::EndpointKind::Rpc);
    assert_eq!(get_user.module.as_deref(), Some("command_routes"));
    assert_eq!(get_user.params[0].source_name, "user_id");
    assert_eq!(get_user.params[0].wire_name, "userId");
    assert_eq!(get_user.returns.name.as_deref(), Some("User"));

    let keys: Vec<_> = graph.endpoints.keys().cloned().collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    assert_eq!(dumped, bridge.dump_schema_json());
}

#[tokio::test]
async fn configured_router_has_command_route_registered() {
    let router = test_bridge().configure(Router::new());
    let (status, body) = post_json(router, "/command/get_user", r#"{"userId":7}"#).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["result"]["name"], "ada");
}
