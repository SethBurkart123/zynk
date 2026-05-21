use zynk_macros::static_file;

#[static_file]
fn avatar(id: i64) -> String {
    format!("avatar-{id}")
}

fn main() {}
