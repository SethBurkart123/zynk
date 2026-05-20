use std::net::SocketAddr;

use axum::Router;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use zynk_runtime::{HandlerKey, TypeRefStatic, ZynkError};

static MESSAGE_EVENT: &[zynk_runtime::ParamMeta] = &[zynk_runtime::ParamMeta {
    source_name: "chat_message",
    wire_name: "chatMessage",
    ty: TypeRefStatic::primitive("string"),
    required: true,
    default: None,
}];

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "ws_echo",
        kind: zynk_runtime::EndpointKind::Ws,
        module: Some("websocket_routes"),
        doc: Some("Echo websocket for route tests."),
        params: &[],
        returns: TypeRefStatic::void(),
        channel_item: None,
        file_param: None,
        multi_file: false,
        max_size: None,
        allowed_types: &[],
        server_events: MESSAGE_EVENT,
        client_events: MESSAGE_EVENT,
        handler_key: Some(HandlerKey("websocket_routes::ws_echo")),
    }
}

zynk_runtime::inventory::submit! {
    zynk_runtime::EndpointMeta {
        name: "ws_error",
        kind: zynk_runtime::EndpointKind::Ws,
        module: Some("websocket_routes"),
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
        handler_key: Some(HandlerKey("websocket_routes::ws_error")),
    }
}

fn test_bridge() -> zynk_axum::ZynkBridge {
    zynk_axum::ZynkBridge::new()
        .register_ws(
            HandlerKey("websocket_routes::ws_echo"),
            |payload: Value, socket: zynk_axum::WebSocket| async move {
                if payload["event"] == "chat_message" {
                    socket
                        .send("message_received", payload["data"].clone())
                        .await?;
                }
                Ok(())
            },
        )
        .register_ws(
            HandlerKey("websocket_routes::ws_error"),
            |_payload: Value, _socket: zynk_axum::WebSocket| async move {
                Err(ZynkError::new(
                    zynk_runtime::WEBSOCKET_ERROR,
                    "handler exploded",
                ))
            },
        )
}

async fn spawn_router(router: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has addr");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("server runs");
    });
    addr
}

async fn connect_ws(addr: SocketAddr, path: &str) -> TcpStream {
    let mut stream = TcpStream::connect(addr).await.expect("tcp connects");
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         Sec-WebSocket-Version: 13\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("handshake writes");

    let mut response = Vec::new();
    let mut byte = [0_u8; 1];
    while !response.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).await.expect("handshake reads");
        response.push(byte[0]);
    }
    let response = String::from_utf8(response).expect("handshake is utf-8");
    assert!(
        response.starts_with("HTTP/1.1 101"),
        "expected websocket upgrade, got {response:?}"
    );
    stream
}

async fn write_masked_text(stream: &mut TcpStream, text: &str) {
    let payload = text.as_bytes();
    assert!(
        payload.len() < 126,
        "test payloads use short websocket frames"
    );
    let mask = [1_u8, 2, 3, 4];
    let mut frame = Vec::with_capacity(6 + payload.len());
    frame.push(0x81);
    frame.push(0x80 | payload.len() as u8);
    frame.extend_from_slice(&mask);
    for (index, byte) in payload.iter().enumerate() {
        frame.push(byte ^ mask[index % 4]);
    }
    stream.write_all(&frame).await.expect("ws frame writes");
}

async fn read_frame(stream: &mut TcpStream) -> (u8, Vec<u8>) {
    let mut header = [0_u8; 2];
    stream
        .read_exact(&mut header)
        .await
        .expect("frame header reads");
    let opcode = header[0] & 0x0f;
    let masked = header[1] & 0x80 != 0;
    let mut len = u64::from(header[1] & 0x7f);
    if len == 126 {
        let mut extended = [0_u8; 2];
        stream
            .read_exact(&mut extended)
            .await
            .expect("extended frame length reads");
        len = u64::from(u16::from_be_bytes(extended));
    } else if len == 127 {
        let mut extended = [0_u8; 8];
        stream
            .read_exact(&mut extended)
            .await
            .expect("extended frame length reads");
        len = u64::from_be_bytes(extended);
    }

    let mut mask = [0_u8; 4];
    if masked {
        stream.read_exact(&mut mask).await.expect("mask reads");
    }

    let mut payload = vec![0_u8; len as usize];
    stream
        .read_exact(&mut payload)
        .await
        .expect("frame payload reads");
    if masked {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[index % 4];
        }
    }
    (opcode, payload)
}

fn close_code_and_reason(payload: &[u8]) -> (u16, String) {
    assert!(payload.len() >= 2, "close frame includes status code");
    let code = u16::from_be_bytes([payload[0], payload[1]]);
    let reason = String::from_utf8(payload[2..].to_vec()).expect("close reason utf-8");
    (code, reason)
}

#[tokio::test]
async fn websocket_route_exchanges_json_frames_with_snake_case_events() {
    let addr = spawn_router(test_bridge().configure(Router::new())).await;
    let mut socket = connect_ws(addr, "/ws/ws_echo").await;

    write_masked_text(
        &mut socket,
        r#"{"event":"unknown_event","data":{"ignored":true}}"#,
    )
    .await;
    write_masked_text(
        &mut socket,
        r#"{"event":"chat_message","data":{"body":"hello"}}"#,
    )
    .await;

    let (opcode, payload) = read_frame(&mut socket).await;
    assert_eq!(opcode, 1, "expected text frame");
    let text = String::from_utf8(payload).expect("text frame is utf-8");
    let body: Value = serde_json::from_str(&text).expect("valid ws json");
    assert_eq!(
        body,
        json!({"event": "message_received", "data": {"body": "hello"}})
    );
}

#[tokio::test]
async fn websocket_unknown_handler_closes_after_upgrade_with_4004() {
    let addr = spawn_router(test_bridge().configure(Router::new())).await;
    let mut socket = connect_ws(addr, "/ws/does_not_exist").await;

    let (opcode, payload) = read_frame(&mut socket).await;
    assert_eq!(opcode, 8, "expected close frame");
    let (code, reason) = close_code_and_reason(&payload);
    assert_eq!(code, 4004);
    assert_eq!(reason, "Handler 'does_not_exist' not found");
}

#[tokio::test]
async fn websocket_handler_error_closes_with_1011_reason() {
    let addr = spawn_router(test_bridge().configure(Router::new())).await;
    let mut socket = connect_ws(addr, "/ws/ws_error").await;

    write_masked_text(&mut socket, r#"{"event":"ignored","data":{}}"#).await;

    let (opcode, payload) = read_frame(&mut socket).await;
    assert_eq!(opcode, 8, "expected close frame");
    let (code, reason) = close_code_and_reason(&payload);
    assert_eq!(code, 1011);
    assert_eq!(reason, "handler exploded");
}
