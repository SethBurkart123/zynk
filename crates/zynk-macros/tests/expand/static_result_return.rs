use zynk_macros::static_file;

pub struct StaticFile;

#[static_file]
fn avatar(id: i64) -> Result<StaticFile, String> {
    let _ = id;
    Err(String::from("not found"))
}

fn main() {}
