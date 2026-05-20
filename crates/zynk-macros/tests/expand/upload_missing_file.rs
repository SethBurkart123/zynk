use zynk_macros::upload;

#[upload]
async fn upload_without_file(user_id: i64) -> String {
    format!("user-{user_id}")
}

fn main() {}
