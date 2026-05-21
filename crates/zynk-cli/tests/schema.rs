use std::path::PathBuf;
use std::process::Command;

use zynk_schema::ApiGraph;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|crates| crates.parent())
        .expect("crate lives under <repo>/crates/zynk-cli")
        .to_path_buf()
}

fn zynk_bin() -> &'static str {
    env!("CARGO_BIN_EXE_zynk")
}

fn uv_python() -> PathBuf {
    repo_root().join("bindings/python/.venv/bin/python")
}

fn assert_schema_json(stdout: &[u8]) -> ApiGraph {
    let text = std::str::from_utf8(stdout).expect("schema dump stdout is utf-8");
    serde_json::from_str::<ApiGraph>(text).expect("schema dump stdout parses as ApiGraph")
}

#[test]
fn schema_dump_app_prints_pretty_valid_apigraph_json() {
    let output = Command::new(zynk_bin())
        .args(["schema", "dump"])
        .args(["--app", "tests.fixtures.roundtrip_schema_fixture:bridge"])
        .arg("--python")
        .arg(uv_python())
        .current_dir(repo_root().join("bindings/python"))
        .output()
        .expect("run zynk schema dump --app");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr was not empty");
    let graph = assert_schema_json(&output.stdout);
    assert!(graph.endpoints.contains_key("update_profile"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\n  \"endpoints\""),
        "stdout was not pretty JSON:\n{stdout}"
    );
}

#[test]
fn schema_dump_compact_prints_single_line_valid_apigraph_json() {
    let output = Command::new(zynk_bin())
        .args(["schema", "dump", "--compact"])
        .args(["--app", "tests.fixtures.roundtrip_schema_fixture:bridge"])
        .arg("--python")
        .arg(uv_python())
        .current_dir(repo_root().join("bindings/python"))
        .output()
        .expect("run zynk schema dump --compact --app");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = assert_schema_json(&output.stdout);
    assert!(graph.endpoints.contains_key("update_profile"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.lines().count(),
        1,
        "compact output should be one line"
    );
}

#[test]
fn schema_dump_direct_command_prints_valid_apigraph_json() {
    let output = Command::new(zynk_bin())
        .args(["schema", "dump", "--"])
        .arg(uv_python())
        .arg("tests/fixtures/roundtrip_schema_fixture.py")
        .current_dir(repo_root().join("bindings/python"))
        .output()
        .expect("run zynk schema dump direct command");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = assert_schema_json(&output.stdout);
    assert!(graph.endpoints.contains_key("update_profile"));
}

#[test]
fn schema_dump_infers_top_level_python_app_command() {
    let output = Command::new(zynk_bin())
        .args(["schema", "dump", "--"])
        .arg(uv_python())
        .arg("main.py")
        .current_dir(repo_root().join("examples/kitchen-sink/backend"))
        .output()
        .expect("run zynk schema dump inferred app command");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = assert_schema_json(&output.stdout);
    assert!(graph.endpoints.contains_key("get_user"));
}

#[test]
fn schema_dump_user_app_errors_exit_nonzero_and_surface_stderr() {
    let output = Command::new(zynk_bin())
        .args(["schema", "dump"])
        .args(["--app", "nonexistent:bad"])
        .arg("--python")
        .arg(uv_python())
        .current_dir(repo_root().join("bindings/python"))
        .output()
        .expect("run zynk schema dump bad import");

    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "stdout should stay empty on failure"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ModuleNotFoundError"),
        "stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("user app schema dump subprocess failed"),
        "stderr was:\n{stderr}"
    );
}
