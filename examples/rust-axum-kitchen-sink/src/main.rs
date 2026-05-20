use std::net::SocketAddr;

use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dump_schema_json = false;
    let mut port = 8101_u16;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dump-schema-json" => dump_schema_json = true,
            "--port" => {
                if let Some(value) = args.next() {
                    port = value.parse()?;
                }
            }
            _ => {}
        }
    }

    if dump_schema_json {
        println!("{}", rust_axum_kitchen_sink::dump_schema_json());
        return Ok(());
    }

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, rust_axum_kitchen_sink::router()).await?;
    Ok(())
}
