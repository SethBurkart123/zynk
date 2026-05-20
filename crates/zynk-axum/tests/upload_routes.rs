use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use zynk_runtime::{HandlerKey, TypeRefStatic};

static UPLOAD_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "album_id",
    wire_name: "albumId",
    ty: TypeRefStatic::primitive("number"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "upload_avatar_axum",
        kind: zynk_runtime::EndpointKind::Upload,
        module: Some("upload_routes"),
        doc: Some("Upload a single avatar."),
        params: UPLOAD_PARAMS,
        returns: TypeRefStatic::any(),
        channel_item: None,
        file_param: Some("file"),
        multi_file: false,
        max_size: Some(8),
        allowed_types: &["text/*"],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("upload_routes::upload_avatar_axum")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "upload_gallery_axum",
        kind: zynk_runtime::EndpointKind::Upload,
        module: Some("upload_routes"),
        doc: Some("Upload multiple files."),
        params: &[],
        returns: TypeRefStatic::any(),
        channel_item: None,
        file_param: Some("files"),
        multi_file: true,
        max_size: Some(32),
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("upload_routes::upload_gallery_axum")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "rpc_not_upload_axum",
        kind: zynk_runtime::EndpointKind::Rpc,
        module: Some("upload_routes"),
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

fn test_bridge() -> zynk_axum::ZynkBridge {
    zynk_axum::ZynkBridge::new()
        .register_upload(
            HandlerKey("upload_routes::upload_avatar_axum"),
            |payload: Value, files: Vec<zynk_axum::UploadFile>| async move {
                let file = files.first().expect("single file provided");
                Ok(json!({
                    "albumId": payload.get("album_id").or_else(|| payload.get("albumId")).and_then(Value::as_i64),
                    "filename": file.filename(),
                    "contentType": file.content_type(),
                    "size": file.size(),
                    "text": std::str::from_utf8(file.bytes()).expect("utf-8 upload"),
                }))
            },
        )
        .register_upload(
            HandlerKey("upload_routes::upload_gallery_axum"),
            |_payload: Value, files: Vec<zynk_axum::UploadFile>| async move {
                Ok(json!({
                    "count": files.len(),
                    "sizes": files.iter().map(zynk_axum::UploadFile::size).collect::<Vec<_>>(),
                }))
            },
        )
}

fn multipart_body(args: &str, files: &[(&str, &str, &str)]) -> (String, String) {
    let boundary = "zynk-test-boundary";
    let mut body = String::new();
    body.push_str(&format!("--{boundary}\r\n"));
    body.push_str("Content-Disposition: form-data; name=\"_args\"\r\n\r\n");
    body.push_str(args);
    body.push_str("\r\n");
    for (filename, content_type, content) in files {
        body.push_str(&format!("--{boundary}\r\n"));
        body.push_str(&format!(
            "Content-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\n"
        ));
        body.push_str(&format!("Content-Type: {content_type}\r\n\r\n"));
        body.push_str(content);
        body.push_str("\r\n");
    }
    body.push_str(&format!("--{boundary}--\r\n"));
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_multipart(
    router: Router,
    path: &str,
    args: &str,
    files: &[(&str, &str, &str)],
) -> (StatusCode, Value) {
    let (content_type, body) = multipart_body(args, files);
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", content_type)
                .body(Body::from(body))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collected")
        .to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).expect("json response"),
    )
}

#[tokio::test]
async fn upload_route_accepts_files_field_and_args_json() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_multipart(
        router,
        "/upload/upload_avatar_axum",
        r#"{"album_id": 42}"#,
        &[("avatar.txt", "text/plain", "hello")],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        json!({"result": {"albumId": 42, "filename": "avatar.txt", "contentType": "text/plain", "size": 5, "text": "hello"}})
    );
}

#[tokio::test]
async fn upload_route_supports_multiple_files() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_multipart(
        router,
        "/upload/upload_gallery_axum",
        "{}",
        &[
            ("a.bin", "application/octet-stream", "aa"),
            ("b.bin", "application/octet-stream", "bbb"),
        ],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"result": {"count": 2, "sizes": [2, 3]}}));
}

#[tokio::test]
async fn upload_route_returns_upload_not_found_for_unknown_or_non_upload() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_multipart(router.clone(), "/upload/missing_axum", "{}", &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "UPLOAD_HANDLER_NOT_FOUND");
    assert_eq!(body["details"], json!({"upload": "missing_axum"}));

    let (status, body) = post_multipart(router, "/upload/rpc_not_upload_axum", "{}", &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "UPLOAD_HANDLER_NOT_FOUND");
}

#[tokio::test]
async fn upload_route_validates_args_size_and_mime_type() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_multipart(
        router.clone(),
        "/upload/upload_avatar_axum",
        "{",
        &[("avatar.txt", "text/plain", "hello")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
    assert!(body["message"]
        .as_str()
        .unwrap()
        .starts_with("Invalid _args JSON:"));

    let (status, body) = post_multipart(
        router.clone(),
        "/upload/upload_avatar_axum",
        r#"{"album_id": 42}"#,
        &[("avatar.txt", "text/plain", "this is too large")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "UPLOAD_VALIDATION_ERROR");
    assert_eq!(body["details"], json!({"filename": "avatar.txt"}));

    let (status, body) = post_multipart(
        router,
        "/upload/upload_avatar_axum",
        r#"{"album_id": 42}"#,
        &[("avatar.json", "application/json", "{}")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "UPLOAD_VALIDATION_ERROR");
    assert_eq!(body["details"], json!({"filename": "avatar.json"}));
}

#[tokio::test]
async fn upload_route_requires_single_file_before_handler_dispatch() {
    let router = test_bridge().configure(Router::new());

    let (status, body) = post_multipart(
        router,
        "/upload/upload_avatar_axum",
        r#"{"album_id": 42}"#,
        &[],
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "VALIDATION_ERROR");
    assert_eq!(body["message"], "No file provided");
}
