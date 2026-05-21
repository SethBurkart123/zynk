use axum::Router;
use serde_json::{json, Value};
use zynk_runtime::{HandlerKey, TypeRefStatic};

static UPLOAD_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "label",
    wire_name: "label",
    ty: TypeRefStatic::primitive("string"),
    required: true,
    default: None,
}];

static STATIC_PARAMS: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "id",
    wire_name: "id",
    ty: TypeRefStatic::primitive("number"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "smoke_upload",
        kind: zynk_runtime::EndpointKind::Upload,
        module: Some("upload_static_smoke"),
        doc: None,
        params: UPLOAD_PARAMS,
        returns: TypeRefStatic::any(),
        channel_item: None,
        file_param: Some("file"),
        multi_file: false,
        max_size: Some(1_048_576),
        allowed_types: &["text/*"],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("upload_static_smoke::smoke_upload")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "smoke_static",
        kind: zynk_runtime::EndpointKind::Static,
        module: Some("upload_static_smoke"),
        doc: None,
        params: STATIC_PARAMS,
        returns: TypeRefStatic::primitive("string"),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: &[],
        client_events: &[],
        handler_key: Some(HandlerKey("upload_static_smoke::smoke_static")),
    }
}

#[tokio::main]
async fn main() {
    let static_path = std::env::temp_dir().join("zynk-axum-static-smoke.txt");
    std::fs::write(&static_path, "static smoke").expect("write smoke static file");

    let bridge = zynk_axum::ZynkBridge::new()
        .register_upload(
            HandlerKey("upload_static_smoke::smoke_upload"),
            |payload: Value, files: Vec<zynk_axum::UploadFile>| async move {
                let file = files.first().expect("file provided");
                Ok(json!({
                    "label": payload.get("label").and_then(Value::as_str),
                    "filename": file.filename(),
                    "size": file.size(),
                }))
            },
        )
        .register_static(
            HandlerKey("upload_static_smoke::smoke_static"),
            move |payload: Value| {
                let static_path = static_path.clone();
                async move {
                    Ok(zynk_axum::StaticFile::new(static_path)
                        .with_content_type("text/plain")
                        .with_header("x-static-id", payload["id"].to_string()))
                }
            },
        );
    let app = bridge.configure(Router::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8101")
        .await
        .expect("bind smoke server");
    axum::serve(listener, app).await.expect("serve smoke app");
}
