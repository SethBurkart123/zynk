use zynk_macros::upload;

pub struct UploadFile;

#[upload(max_size = "10MB", allowed_types = ["image/*"])]
async fn upload_avatar(file: UploadFile, user_id: i64) -> String {
    let _ = file;
    format!("avatar-{user_id}")
}

#[upload]
async fn upload_gallery(files: Vec<UploadFile>, album_id: i64) -> i64 {
    let _ = files;
    album_id
}

fn main() {}
