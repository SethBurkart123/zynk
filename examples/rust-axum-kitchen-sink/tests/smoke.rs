use std::net::SocketAddr;

use futures_util::{SinkExt, StreamExt};
use reqwest::multipart;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

async fn spawn_server() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");
    tokio::spawn(async move {
        axum::serve(listener, rust_axum_kitchen_sink::router())
            .await
            .expect("test server runs");
    });
    addr
}

#[tokio::test]
async fn smoke_covers_command_channel_upload_static_and_websocket() {
    let addr = spawn_server().await;
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let user: Value = client
        .post(format!("{base}/command/get_user"))
        .json(&json!({"userId": 1}))
        .send()
        .await
        .expect("command request succeeds")
        .json()
        .await
        .expect("command response is json");
    assert_eq!(user["result"]["name"], "Alice");

    let sse = client
        .post(format!("{base}/channel/stream_weather"))
        .json(&json!({"city": "Seattle", "intervalSeconds": 1.0}))
        .send()
        .await
        .expect("channel request succeeds")
        .text()
        .await
        .expect("channel body is text");
    assert!(sse.contains("event: message"), "sse body: {sse:?}");
    assert!(sse.contains("\"city\": \"Seattle\""), "sse body: {sse:?}");
    assert!(sse.contains("event: close"), "sse body: {sse:?}");

    let form = multipart::Form::new().text("_args", "{}").part(
        "files",
        multipart::Part::bytes("hello upload".as_bytes().to_vec())
            .file_name("hello.txt")
            .mime_str("text/plain")
            .expect("valid mime"),
    );
    let upload: Value = client
        .post(format!("{base}/upload/upload_file"))
        .multipart(form)
        .send()
        .await
        .expect("upload request succeeds")
        .json()
        .await
        .expect("upload response is json");
    assert_eq!(upload["result"]["filename"], "hello.txt");
    assert_eq!(upload["result"]["size"], 12);

    let static_body = client
        .get(format!("{base}/static/download_sample"))
        .send()
        .await
        .expect("static request succeeds")
        .text()
        .await
        .expect("static body is text");
    assert_eq!(static_body, "Zynk kitchen sink static sample\n");

    let (mut ws, _response) = connect_async(format!("ws://{addr}/ws/chat"))
        .await
        .expect("websocket connects");
    ws.send(Message::Text(
        json!({"event":"chat_message","data":{"user":"Ada","text":"hello","timestamp":"t"}})
            .to_string()
            .into(),
    ))
    .await
    .expect("ws send succeeds");
    let message = ws
        .next()
        .await
        .expect("ws has response")
        .expect("ws response succeeds");
    let text = message.into_text().expect("ws response is text");
    let body: Value = serde_json::from_str(&text).expect("ws response is json");
    assert_eq!(body["event"], "chat_message");
    assert_eq!(body["data"]["user"], "Ada");
}

#[test]
fn dump_schema_contains_python_endpoint_and_type_sets() {
    let graph: zynk_schema::ApiGraph =
        serde_json::from_str(&rust_axum_kitchen_sink::dump_schema_json()).expect("schema json");
    assert_eq!(graph.endpoints.len(), 26);
    for name in [
        "get_user",
        "list_users",
        "stream_weather",
        "upload_file",
        "download_sample",
        "chat",
    ] {
        assert!(
            graph.endpoints.contains_key(name),
            "missing endpoint {name}"
        );
    }
    assert!(graph.models.contains_key("Task"));
    assert!(graph.models.contains_key("WeatherUpdate"));
    assert_eq!(
        graph.enums["TaskPriority"].values,
        vec![
            json!("low"),
            json!("medium"),
            json!("high"),
            json!("urgent")
        ]
    );
}
