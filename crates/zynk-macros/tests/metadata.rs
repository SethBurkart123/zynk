use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use zynk_macros::{command, message, static_file, upload};

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);

    match Pin::new(&mut future).poll(&mut context) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("test future unexpectedly pending"),
    }
}

#[command]
async fn double_count(item_count: i64, display_name: Option<String>) -> i64 {
    assert_eq!(display_name.as_deref(), Some("Ada"));
    item_count * 2
}

#[derive(Debug)]
struct ChatPayload {
    text: String,
}

#[message]
fn chat_message(room_id: String, payload: ChatPayload) {
    assert_eq!(room_id, "general");
    assert_eq!(payload.text, "hello");
}

#[derive(Clone, Debug)]
struct UploadFile {
    name: String,
}

#[upload(max_size = "10MB", allowed_types = ["image/*"])]
async fn upload_avatar(file: UploadFile, user_id: i64, display_name: Option<String>) -> String {
    assert_eq!(file.name, "avatar.png");
    assert_eq!(display_name.as_deref(), Some("Ada"));
    format!("avatar-{user_id}")
}

#[upload]
async fn upload_gallery(files: Vec<UploadFile>, album_id: i64) -> i64 {
    assert_eq!(files.len(), 2);
    album_id
}

#[derive(Debug, PartialEq)]
struct StaticFile {
    path: String,
}

#[static_file]
fn avatar_static(id: i64) -> StaticFile {
    StaticFile {
        path: format!("avatars/{id}.png"),
    }
}

#[test]
fn command_preserves_fn_behavior_and_registers_endpoint_meta() {
    assert_eq!(
        block_on_ready(double_count(21, Some(String::from("Ada")))),
        42
    );

    let meta = zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .find(|meta| meta.name == "double_count")
        .expect("command endpoint metadata is registered");

    assert_eq!(meta.kind, zynk_runtime::EndpointKind::Rpc);
    assert_eq!(
        meta.returns,
        zynk_runtime::TypeRefStatic::primitive("number")
    );
    assert_eq!(meta.params.len(), 2);
    assert_eq!(meta.params[0].source_name, "item_count");
    assert_eq!(meta.params[0].wire_name, "itemCount");
    assert!(meta.params[0].required);
    assert_eq!(
        meta.params[0].ty,
        zynk_runtime::TypeRefStatic::primitive("number")
    );
    assert_eq!(meta.params[1].source_name, "display_name");
    assert_eq!(meta.params[1].wire_name, "displayName");
    assert!(!meta.params[1].required);
    assert_eq!(
        meta.params[1].ty,
        zynk_runtime::TypeRefStatic::primitive("string")
            .optional()
            .nullable()
    );
}

#[test]
fn message_preserves_fn_behavior_and_registers_ws_meta() {
    chat_message(
        String::from("general"),
        ChatPayload {
            text: String::from("hello"),
        },
    );

    let meta = zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .find(|meta| meta.name == "chat_message")
        .expect("message endpoint metadata is registered");

    assert_eq!(meta.kind, zynk_runtime::EndpointKind::Ws);
    assert!(meta.params.is_empty());
    assert_eq!(meta.server_events.len(), 2);
    assert_eq!(meta.client_events.len(), 2);
    assert_eq!(meta.server_events[0].source_name, "room_id");
    assert_eq!(meta.server_events[0].wire_name, "roomId");
    assert_eq!(
        meta.server_events[0].ty,
        zynk_runtime::TypeRefStatic::primitive("string")
    );
    assert_eq!(meta.server_events[1].source_name, "payload");
    assert_eq!(meta.server_events[1].wire_name, "payload");
    assert_eq!(
        meta.server_events[1].ty,
        zynk_runtime::TypeRefStatic::model("ChatPayload")
    );
}

#[test]
fn upload_preserves_fn_behavior_and_registers_file_metadata() {
    assert_eq!(
        block_on_ready(upload_avatar(
            UploadFile {
                name: String::from("avatar.png"),
            },
            7,
            Some(String::from("Ada")),
        )),
        "avatar-7"
    );

    let meta = zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .find(|meta| meta.name == "upload_avatar")
        .expect("upload endpoint metadata is registered");

    assert_eq!(meta.kind, zynk_runtime::EndpointKind::Upload);
    assert_eq!(meta.file_param, Some("file"));
    assert!(!meta.multi_file);
    assert_eq!(meta.max_size, Some(10_485_760));
    assert_eq!(meta.allowed_types, ["image/*"]);
    assert_eq!(
        meta.returns,
        zynk_runtime::TypeRefStatic::primitive("string")
    );
    assert_eq!(meta.params.len(), 2);
    assert_eq!(meta.params[0].source_name, "user_id");
    assert_eq!(meta.params[0].wire_name, "userId");
    assert_eq!(
        meta.params[0].ty,
        zynk_runtime::TypeRefStatic::primitive("number")
    );
    assert_eq!(meta.params[1].source_name, "display_name");
    assert!(!meta.params[1].required);
}

#[test]
fn upload_detects_vec_upload_file_as_multi_file() {
    assert_eq!(
        block_on_ready(upload_gallery(
            vec![
                UploadFile {
                    name: String::from("one.png"),
                },
                UploadFile {
                    name: String::from("two.png"),
                },
            ],
            42,
        )),
        42
    );

    let meta = zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .find(|meta| meta.name == "upload_gallery")
        .expect("multi-file upload endpoint metadata is registered");

    assert_eq!(meta.kind, zynk_runtime::EndpointKind::Upload);
    assert_eq!(meta.file_param, Some("files"));
    assert!(meta.multi_file);
    assert_eq!(meta.max_size, None);
    assert!(meta.allowed_types.is_empty());
    assert_eq!(meta.params.len(), 1);
    assert_eq!(meta.params[0].source_name, "album_id");
}

#[test]
fn static_file_preserves_fn_behavior_and_registers_static_meta() {
    assert_eq!(
        avatar_static(5),
        StaticFile {
            path: String::from("avatars/5.png"),
        }
    );

    let meta = zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .find(|meta| meta.name == "avatar_static")
        .expect("static endpoint metadata is registered");

    assert_eq!(meta.kind, zynk_runtime::EndpointKind::Static);
    assert_eq!(meta.file_param, None);
    assert!(!meta.multi_file);
    assert_eq!(meta.max_size, None);
    assert!(meta.allowed_types.is_empty());
    assert_eq!(meta.params.len(), 1);
    assert_eq!(meta.params[0].source_name, "id");
    assert_eq!(meta.params[0].wire_name, "id");
    assert_eq!(
        meta.params[0].ty,
        zynk_runtime::TypeRefStatic::primitive("number")
    );
    assert_eq!(
        meta.returns,
        zynk_runtime::TypeRefStatic::model("StaticFile")
    );
}
