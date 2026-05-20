use zynk_macros::static_file;

pub struct StaticFile;

#[static_file]
fn avatar(id: i64) -> StaticFile {
    let _ = id;
    StaticFile
}

fn main() {}
