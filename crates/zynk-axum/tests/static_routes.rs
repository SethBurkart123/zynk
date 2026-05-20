use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use zynk_runtime::{HandlerKey, TypeRefStatic};

static STATIC_PARAMS: &[zynk_runtime::ParamMeta] = &[
    zynk_runtime::ParamMeta {
        source_name: "id",
        wire_name: "id",
        ty: TypeRefStatic::primitive("number"),
        required: true,
        default: None,
    },
    zynk_runtime::ParamMeta {
        source_name: "include_meta",
        wire_name: "includeMeta",
        ty: TypeRefStatic::primitive("boolean"),
        required: true,
        default: None,
    },
];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "avatar_static_axum",
        kind: zynk_runtime::EndpointKind::Static,
        module: Some("static_routes"),
        doc: Some("Resolve an avatar file."),
        params: STATIC_PARAMS,
        returns: TypeRefStatic::primitive("string"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("static_routes::avatar_static_axum")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "download_static_axum",
        kind: zynk_runtime::EndpointKind::Static,
        module: Some("static_routes"),
        doc: Some("Resolve a file with an overridden nosniff header."),
        params: &[],
        returns: TypeRefStatic::primitive("string"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("static_routes::download_static_axum")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "rpc_not_static_axum",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("static_routes"),
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
        handler_key: None,
    }
}

fn fixture_path(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("zynk-axum-{name}-{}", std::process::id()));
    std::fs::write(&path, contents).expect("write fixture file");
    path
}

fn test_bridge() -> zynk_axum::ZynkBridge {
    zynk_axum::ZynkBridge::new()
        .register_static(
            HandlerKey("static_routes::avatar_static_axum"),
            |payload: Value| async move {
                assert_eq!(payload["id"], json!(42));
                assert_eq!(payload["include_meta"], json!(true));
                Ok(
                    zynk_axum::StaticFile::new(fixture_path("avatar.txt", "avatar body"))
                        .with_content_type("text/custom")
                        .with_header("x-avatar-id", "42"),
                )
            },
        )
        .register_static(
            HandlerKey("static_routes::download_static_axum"),
            |_payload: Value| async move {
                Ok(
                    zynk_axum::StaticFile::new(fixture_path("download.bin", "download body"))
                        .with_header("X-Content-Type-Options", "off"),
                )
            },
        )
}

async fn request(router: Router, method: Method, path: &str) -> axum::response::Response {
    router
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds")
}

async fn body_bytes(
    response: axum::response::Response,
) -> (StatusCode, axum::http::HeaderMap, bytes::Bytes) {
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collected")
        .to_bytes();
    (status, headers, bytes)
}

#[tokio::test]
async fn static_route_get_serves_file_with_python_compatible_headers() {
    let router = test_bridge().configure(Router::new());

    let response = request(
        router,
        Method::GET,
        "/static/avatar_static_axum?id=42&include_meta=true",
    )
    .await;
    let (status, headers, body) = body_bytes(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "avatar body");
    assert_eq!(headers.get("content-type").unwrap(), "text/custom");
    assert_eq!(headers.get("content-length").unwrap(), "11");
    assert!(headers.get("last-modified").is_some());
    assert_eq!(headers.get("x-content-type-options").unwrap(), "nosniff");
    assert_eq!(headers.get("x-avatar-id").unwrap(), "42");
}

#[tokio::test]
async fn static_route_head_returns_same_headers_without_body() {
    let router = test_bridge().configure(Router::new());

    let response = request(
        router,
        Method::HEAD,
        "/static/avatar_static_axum?id=42&include_meta=true",
    )
    .await;
    let (status, headers, body) = body_bytes(response).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty());
    assert_eq!(headers.get("content-type").unwrap(), "text/custom");
    assert_eq!(headers.get("content-length").unwrap(), "11");
    assert_eq!(headers.get("x-content-type-options").unwrap(), "nosniff");
}

#[tokio::test]
async fn static_route_coerces_query_params_and_validates_required_values() {
    let router = test_bridge().configure(Router::new());

    let response = request(
        router.clone(),
        Method::GET,
        "/static/avatar_static_axum?id=42",
    )
    .await;
    let (status, _headers, body) = body_bytes(response).await;
    let body: Value = serde_json::from_slice(&body).expect("json error body");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
    assert_eq!(body["message"], "Missing required parameter: include_meta");

    let response = request(
        router,
        Method::GET,
        "/static/avatar_static_axum?id=not-a-number&include_meta=true",
    )
    .await;
    let (status, _headers, body) = body_bytes(response).await;
    let body: Value = serde_json::from_slice(&body).expect("json error body");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
}

#[tokio::test]
async fn static_route_unknown_or_non_static_returns_static_not_found() {
    let router = test_bridge().configure(Router::new());

    let response = request(router.clone(), Method::GET, "/static/missing_axum").await;
    let (status, _headers, body) = body_bytes(response).await;
    let body: Value = serde_json::from_slice(&body).expect("json error body");
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "STATIC_HANDLER_NOT_FOUND");
    assert_eq!(body["details"], json!({"static": "missing_axum"}));

    let response = request(router, Method::GET, "/static/rpc_not_static_axum").await;
    let (status, _headers, body) = body_bytes(response).await;
    let body: Value = serde_json::from_slice(&body).expect("json error body");
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "STATIC_HANDLER_NOT_FOUND");
}

#[tokio::test]
async fn static_route_allows_user_header_to_override_nosniff_like_python() {
    let router = test_bridge().configure(Router::new());

    let response = request(router, Method::GET, "/static/download_static_axum").await;
    let (status, headers, body) = body_bytes(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "download body");
    assert_eq!(headers.get("x-content-type-options").unwrap(), "off");
}
