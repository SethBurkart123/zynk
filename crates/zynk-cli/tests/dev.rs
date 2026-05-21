use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "zynk-cli-dev-{name}-{}-{nanos}",
        std::process::id()
    ))
}

fn zynk_bin() -> &'static str {
    env!("CARGO_BIN_EXE_zynk")
}

fn schema_json(endpoint_name: &str) -> String {
    format!(
        r#"{{"endpoints":{{"{endpoint_name}":{{"name":"{endpoint_name}","kind":"rpc","params":[],"returns":{{"kind":"primitive","name":"string"}},"multiFile":false}}}},"models":{{}},"enums":{{}}}}"#
    )
}

fn write_app(dir: &Path, endpoint_name: &str, sleep_secs: Option<f32>) {
    let sleep_line = sleep_secs
        .map(|secs| format!("        time.sleep({secs})\n"))
        .unwrap_or_default();
    fs::write(
        dir.join("app.py"),
        format!(
            "import time\n\nclass Bridge:\n    def dump_schema_json(self):\n{sleep_line}        return {schema:?}\n\nbridge = Bridge()\n",
            schema = schema_json(endpoint_name),
        ),
    )
    .expect("write app fixture");
}

fn wait_for_api_containing(out: &Path, needle: &str, timeout: Duration) -> String {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = fs::read_to_string(out.join("api.ts")) {
            if contents.contains(needle) {
                return contents;
            }
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for generated api.ts to contain {needle}"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

fn wait_for_file(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if path.exists() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(50));
    }
}

fn spawn_dev(app_dir: &Path, out: &Path) -> Child {
    Command::new(zynk_bin())
        .args(["dev", "--out"])
        .arg(out)
        .args([
            "--app",
            "app:bridge",
            "--python",
            "python3",
            "--debounce-ms",
            "100",
        ])
        .current_dir(app_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn zynk dev")
}

fn stop_dev(mut child: Child) {
    if child.try_wait().expect("poll child").is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[test]
fn dev_writes_initial_files_and_regenerates_after_python_change() {
    let app_dir = temp_dir("regen-app");
    let out = temp_dir("regen-out");
    fs::create_dir_all(&app_dir).expect("create app dir");
    fs::create_dir_all(&out).expect("create out dir");
    write_app(&app_dir, "first_endpoint", None);

    let child = spawn_dev(&app_dir, &out);
    wait_for_api_containing(&out, "firstEndpoint", Duration::from_secs(5));
    wait_for_file(&out.join("_internal.ts"), Duration::from_secs(5));
    assert!(out.join("_internal.ts").is_file());

    write_app(&app_dir, "second_endpoint", None);
    wait_for_api_containing(&out, "secondEndpoint", Duration::from_secs(5));
    stop_dev(child);
}

#[test]
fn dev_debounces_rapid_python_changes_to_final_state() {
    let app_dir = temp_dir("debounce-app");
    let out = temp_dir("debounce-out");
    fs::create_dir_all(&app_dir).expect("create app dir");
    fs::create_dir_all(&out).expect("create out dir");
    let count_file = app_dir.join("count.txt");
    fs::write(&count_file, "0").expect("write count");
    fs::write(
        app_dir.join("app.py"),
        format!(
            "import json\nfrom pathlib import Path\nCOUNT = Path({count_path:?})\nclass Bridge:\n    def dump_schema_json(self):\n        value = int(COUNT.read_text())\n        COUNT.write_text(str(value + 1))\n        endpoint = Path('endpoint.txt').read_text().strip()\n        graph = {{'endpoints': {{endpoint: {{'name': endpoint, 'kind': 'rpc', 'params': [], 'returns': {{'kind': 'primitive', 'name': 'string'}}, 'multiFile': False}}}}, 'models': {{}}, 'enums': {{}}}}\n        return json.dumps(graph)\nbridge = Bridge()\n",
            count_path = count_file.display().to_string(),
        ),
    )
    .expect("write app fixture");
    fs::write(app_dir.join("endpoint.txt"), "initial_endpoint").expect("write endpoint");

    let child = spawn_dev(&app_dir, &out);
    wait_for_api_containing(&out, "initialEndpoint", Duration::from_secs(5));
    fs::write(&count_file, "0").expect("reset count after initial generation");

    fs::write(app_dir.join("endpoint.txt"), "final_endpoint").expect("write final endpoint");
    for index in 0..10 {
        fs::write(app_dir.join("watched.py"), format!("# change {index}\n"))
            .expect("write watched file");
        thread::sleep(Duration::from_millis(5));
    }

    let api = wait_for_api_containing(&out, "finalEndpoint", Duration::from_secs(5));
    thread::sleep(Duration::from_millis(500));
    let count: usize = fs::read_to_string(&count_file)
        .expect("read count")
        .parse()
        .expect("count is numeric");
    assert_eq!(
        count, 1,
        "rapid events should trigger one regeneration after debounce"
    );
    assert!(api.contains("finalEndpoint"));
    stop_dev(child);
}

#[test]
fn dev_sigint_exits_cleanly_during_slow_regeneration() {
    let app_dir = temp_dir("sigint-app");
    let out = temp_dir("sigint-out");
    fs::create_dir_all(&app_dir).expect("create app dir");
    fs::create_dir_all(&out).expect("create out dir");
    write_app(&app_dir, "initial_endpoint", None);

    let mut child = spawn_dev(&app_dir, &out);
    wait_for_file(&out.join("api.ts"), Duration::from_secs(5));

    write_app(&app_dir, "slow_endpoint", Some(10.0));
    thread::sleep(Duration::from_millis(250));
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("send SIGINT");

    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "zynk dev did not exit within 5 seconds after SIGINT"
        );
        thread::sleep(Duration::from_millis(50));
    };

    assert!(
        status.success(),
        "zynk dev should exit successfully on SIGINT, got {status}"
    );

    let ps = Command::new("ps")
        .args(["-o", "pid=", "-o", "ppid=", "-o", "command=", "-ax"])
        .output()
        .expect("run ps");
    let listing = String::from_utf8_lossy(&ps.stdout);
    let leaked = listing.lines().any(|line| {
        line.contains(&app_dir.to_string_lossy().to_string()) && line.contains("python")
    });
    assert!(!leaked, "python subprocess leaked after SIGINT:\n{listing}");
}
