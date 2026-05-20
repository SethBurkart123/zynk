use std::fs;
use std::path::{Path, PathBuf};

use zynk_codegen::GeneratedFile;
use zynk_schema::ApiGraph;

fn fixture_graph() -> ApiGraph {
    serde_json::from_str(include_str!("../../zynk-gen-ts/tests/fixtures/kitchen_sink_schema.json"))
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

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/kitchen-sink/frontend/src/generated-effect")
}

#[test]
fn kitchen_sink_effect_golden_matches_generated_client() {
    let graph = fixture_graph();
    let generated = zynk_gen_effect::generate(&graph);
    let dir = golden_dir();

    if std::env::var_os("UPDATE_ZYNK_EFFECT_GOLDEN").is_some() {
        fs::create_dir_all(&dir).expect("create generated-effect golden dir");
        fs::write(dir.join("api.ts"), file(&generated.files, "api.ts"))
            .expect("write Effect api.ts golden");
        fs::write(
            dir.join("_effect_internal.ts"),
            file(&generated.files, "_effect_internal.ts"),
        )
        .expect("write Effect runtime golden");
        return;
    }

    let expected_api = fs::read_to_string(dir.join("api.ts"))
        .expect("Effect api.ts golden exists; set UPDATE_ZYNK_EFFECT_GOLDEN=1 to regenerate");
    let expected_runtime = fs::read_to_string(dir.join("_effect_internal.ts")).expect(
        "Effect _effect_internal.ts golden exists; set UPDATE_ZYNK_EFFECT_GOLDEN=1 to regenerate",
    );

    assert_eq!(file(&generated.files, "api.ts"), expected_api);
    assert_eq!(file(&generated.files, "_effect_internal.ts"), expected_runtime);
}
