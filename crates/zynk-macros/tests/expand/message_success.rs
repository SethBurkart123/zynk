use zynk_macros::message;

#[derive(Debug)]
struct ChatPayload {
    text: String,
}

#[message]
fn chat_message(room_id: String, payload: ChatPayload) {
    let _ = (room_id, payload.text);
}

fn endpoint_registered() -> bool {
    zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .any(|meta| {
            meta.name == "chat_message"
                && meta.kind == zynk_runtime::EndpointKind::Ws
                && meta.server_events.len() == 2
                && meta.client_events.len() == 2
                && meta.server_events[0].wire_name == "roomId"
        })
}

fn main() {
    chat_message(String::from("general"), ChatPayload { text: String::from("hi") });
    assert!(endpoint_registered());
}
