use zynk_macros::command;

struct Greeter;

impl Greeter {
    #[command]
    async fn greet(&mut self, name: String) -> String {
        format!("hello {name}")
    }
}

fn main() {}
