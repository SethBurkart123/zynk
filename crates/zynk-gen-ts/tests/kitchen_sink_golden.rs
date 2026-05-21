use std::path::Path;

use zynk_codegen::GeneratedFile;
use zynk_schema::ApiGraph;

fn fixture_graph() -> ApiGraph {
    serde_json::from_str(include_str!("fixtures/kitchen_sink_schema.json"))
        .expect("kitchen sink fixture deserializes as ApiGraph")
}

fn file<'a>(files: &'a [GeneratedFile], path: &str) -> &'a str {
    files
        .iter()
        .find(|file| file.path == Path::new(path))
        .unwrap_or_else(|| panic!("generated output contains {path}"))
        .contents
        .as_str()
}

#[test]
fn kitchen_sink_golden_matches_generated_client() {
    let graph = fixture_graph();
    let generated = zynk_gen_ts::generate(&graph);

    assert_eq!(
        file(&generated.files, "api.ts"),
        include_str!("../../../examples/kitchen-sink/frontend/src/generated/api.ts")
    );
    assert_eq!(
        file(&generated.files, "_internal.ts"),
        include_str!("../../../examples/kitchen-sink/frontend/src/generated/_internal.ts")
    );
}
