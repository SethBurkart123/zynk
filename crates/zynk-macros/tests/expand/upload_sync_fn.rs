use zynk_macros::upload;

pub struct UploadFile;

#[upload]
fn upload_avatar(file: UploadFile) -> String {
    let _ = file;
    String::from("avatar")
}

fn main() {}
