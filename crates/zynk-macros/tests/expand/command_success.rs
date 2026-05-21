use zynk_macros::command;

#[command]
async fn add_one(user_id: i64, display_name: String) -> i64 {
    let _ = display_name;
    user_id + 1
}

fn endpoint_registered() -> bool {
    zynk_runtime::inventory::iter::<zynk_runtime::EndpointMeta>
        .into_iter()
        .any(|meta| {
            meta.name == "add_one"
                && meta.kind == zynk_runtime::EndpointKind::Rpc
                && meta.params.len() == 2
                && meta.params[0].source_name == "user_id"
                && meta.params[0].wire_name == "userId"
                && meta.params[0].required
        })
}

fn main() {
    let future = add_one(41, String::from("Ada"));
    std::mem::drop(future);
    assert!(endpoint_registered());
}
