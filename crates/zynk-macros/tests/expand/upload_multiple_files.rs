use zynk_macros::upload;

pub struct UploadFile;

#[upload]
async fn upload_pair(primary: UploadFile, secondary: Vec<UploadFile>) -> String {
    let _ = (primary, secondary);
    String::from("ok")
}

fn main() {}
