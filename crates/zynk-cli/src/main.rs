use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use zynk_codegen::GenerationResult;
use zynk_schema::ApiGraph;

#[derive(Debug, Parser)]
#[command(name = "zynk", version, about = "Generate Zynk TypeScript clients")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate a TypeScript client from a schema-capable user app.
    Gen(GenArgs),
    /// Watch user source files and regenerate a TypeScript client on changes.
    Dev(DevArgs),
}

#[derive(Debug, Args)]
struct GenArgs {
    /// Client flavor to generate.
    #[arg(value_enum)]
    language: Language,

    /// Backend language to introspect.
    #[arg(long, value_enum)]
    target: Target,

    /// Output directory or api.ts path. Directories receive api.ts plus runtime siblings.
    #[arg(long)]
    out: PathBuf,

    /// Python Bridge object as <module>:<attr>.
    #[arg(long)]
    app: Option<String>,

    /// Python executable used for target=python.
    #[arg(long, default_value = "python")]
    python: OsString,

    /// Rust app launch command used for target=rust. The CLI appends --dump-schema-json.
    #[arg(long, value_name = "COMMAND")]
    rust_cmd: Option<String>,

    /// Additional arguments passed to --rust-cmd before --dump-schema-json.
    #[arg(long = "rust-arg", value_name = "ARG")]
    rust_args: Vec<OsString>,
}

#[derive(Debug, Args)]
struct DevArgs {
    /// Output directory or api.ts path. Directories receive api.ts plus runtime siblings.
    #[arg(long)]
    out: PathBuf,

    /// Client flavor to generate.
    #[arg(long, value_enum, default_value_t = Language::Typescript)]
    language: Language,

    /// Python Bridge object as <module>:<attr>.
    #[arg(long)]
    app: Option<String>,

    /// Python executable used with --app.
    #[arg(long, default_value = "python")]
    python: OsString,

    /// Debounce window for overlapping filesystem events.
    #[arg(long, default_value_t = 250)]
    debounce_ms: u64,

    /// Glob-like watched source patterns. Defaults to recursive *.py under cwd.
    #[arg(long = "watch", value_name = "GLOB")]
    watch_globs: Vec<String>,

    /// Schema-dump command to run if --app is not provided.
    #[arg(
        value_name = "APP_CMD",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    app_cmd: Vec<OsString>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Language {
    Typescript,
    Effect,
}

impl std::fmt::Display for Language {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Typescript => formatter.write_str("typescript"),
            Language::Effect => formatter.write_str("effect"),
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Target {
    Python,
    Rust,
}

#[derive(Debug, Clone)]
enum SchemaDumpCommand {
    PythonApp { python: OsString, app: String },
    Direct(Vec<OsString>),
}

#[derive(Debug)]
struct DevConfig {
    language: Language,
    out: PathBuf,
    dump_command: SchemaDumpCommand,
    watch_root: PathBuf,
    watch_globs: Vec<String>,
    debounce: Duration,
}

#[derive(Debug)]
enum DevMessage {
    FsEvent(notify::Result<Event>),
    Shutdown,
}

#[derive(Debug)]
enum RegenOutcome {
    Completed(Result<()>),
    Canceled,
    Shutdown,
}

#[derive(Debug)]
enum DebounceOutcome {
    Ready,
    Shutdown,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Gen(args) => gen(args),
        Commands::Dev(args) => dev(args),
    }
}

fn gen(args: GenArgs) -> Result<()> {
    let graph = match args.target {
        Target::Python => dump_python_schema(&args)?,
        Target::Rust => dump_rust_schema(&args)?,
    };
    let result = generate_files(args.language, &graph);
    write_generated_files(&result, &args.out)?;
    Ok(())
}

fn dev(args: DevArgs) -> Result<()> {
    let dump_command = dev_dump_command(&args)?;
    let config = DevConfig {
        language: args.language,
        out: args.out,
        dump_command,
        watch_root: std::env::current_dir().context("failed to resolve current directory")?,
        watch_globs: args.watch_globs,
        debounce: Duration::from_millis(args.debounce_ms),
    };

    let (tx, rx) = mpsc::channel();
    spawn_shutdown_listeners(tx.clone());

    match run_regeneration_cancellable(&rx, &config)? {
        RegenOutcome::Completed(Ok(())) => {}
        RegenOutcome::Completed(Err(err)) => return Err(err.context("initial regeneration failed")),
        RegenOutcome::Canceled => unreachable!("no watcher is active during initial regeneration"),
        RegenOutcome::Shutdown => return Ok(()),
    }

    let watcher = create_source_watcher(&config, tx)?;
    eprintln!(
        "zynk dev watching {} for {} client regeneration",
        config.watch_root.display(),
        config.language
    );

    while let Ok(message) = rx.recv() {
        match message {
            DevMessage::Shutdown => break,
            DevMessage::FsEvent(event) => {
                if !watched_event_result(event, &config) {
                    continue;
                }
                if matches!(wait_for_debounce(&rx, &config), DebounceOutcome::Shutdown) {
                    break;
                }
                loop {
                    match run_regeneration_cancellable(&rx, &config)? {
                        RegenOutcome::Completed(Ok(())) => break,
                        RegenOutcome::Completed(Err(err)) => {
                            eprintln!("Error: regeneration failed: {err:#}");
                            break;
                        }
                        RegenOutcome::Canceled => {
                            if matches!(wait_for_debounce(&rx, &config), DebounceOutcome::Shutdown)
                            {
                                drop(watcher);
                                return Ok(());
                            }
                        }
                        RegenOutcome::Shutdown => {
                            drop(watcher);
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    drop(watcher);
    Ok(())
}

fn create_source_watcher(
    config: &DevConfig,
    tx: Sender<DevMessage>,
) -> Result<notify::RecommendedWatcher> {
    let mut watcher = notify::recommended_watcher(move |result| {
        let _ = tx.send(DevMessage::FsEvent(result));
    })
    .context("failed to create file watcher")?;
    watcher
        .watch(&config.watch_root, RecursiveMode::Recursive)
        .with_context(|| format!("failed to watch {}", config.watch_root.display()))?;
    Ok(watcher)
}

fn spawn_shutdown_listeners(tx: Sender<DevMessage>) {
    thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(err) => {
                eprintln!("Error: failed to initialize signal listener: {err}");
                return;
            }
        };
        runtime.block_on(async move {
            let mut terminate = unix_termination_signal();
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    if let Err(err) = result {
                        eprintln!("Error: failed to listen for Ctrl-C: {err}");
                    }
                }
                _ = async {
                    if let Some(signal) = terminate.as_mut() {
                        signal.recv().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {}
            }
            let _ = tx.send(DevMessage::Shutdown);
        });
    });
}

#[cfg(unix)]
fn unix_termination_signal() -> Option<tokio::signal::unix::Signal> {
    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok()
}

#[cfg(not(unix))]
fn unix_termination_signal() -> Option<()> {
    None
}

fn wait_for_debounce(rx: &Receiver<DevMessage>, config: &DevConfig) -> DebounceOutcome {
    let mut deadline = Instant::now() + config.debounce;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return DebounceOutcome::Ready;
        }
        match rx.recv_timeout(deadline.saturating_duration_since(now)) {
            Ok(DevMessage::Shutdown) => return DebounceOutcome::Shutdown,
            Ok(DevMessage::FsEvent(event)) => {
                if watched_event_result(event, config) {
                    deadline = Instant::now() + config.debounce;
                }
            }
            Err(RecvTimeoutError::Timeout) => return DebounceOutcome::Ready,
            Err(RecvTimeoutError::Disconnected) => return DebounceOutcome::Shutdown,
        }
    }
}

fn run_regeneration_cancellable(
    rx: &Receiver<DevMessage>,
    config: &DevConfig,
) -> Result<RegenOutcome> {
    let mut child = spawn_schema_dump(&config.dump_command)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("schema dump stdout pipe was unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("schema dump stderr pipe was unavailable"))?;
    let stdout_reader = thread::spawn(move || read_pipe(stdout, "schema dump stdout"));
    let stderr_reader = thread::spawn(move || read_pipe(stderr, "schema dump stderr"));

    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to poll schema dump subprocess")?
        {
            let result = read_child_output(stdout_reader, stderr_reader)
                .and_then(|(stdout, stderr)| schema_from_output(status.success(), &stdout, &stderr))
                .and_then(|graph| {
                    let result = generate_files(config.language, &graph);
                    write_generated_files(&result, &config.out)
                });
            return Ok(RegenOutcome::Completed(result));
        }

        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(DevMessage::Shutdown) => {
                terminate_child(&mut child);
                let _ = read_child_output(stdout_reader, stderr_reader);
                return Ok(RegenOutcome::Shutdown);
            }
            Ok(DevMessage::FsEvent(event)) => {
                if watched_event_result(event, config) {
                    terminate_child(&mut child);
                    let _ = read_child_output(stdout_reader, stderr_reader);
                    return Ok(RegenOutcome::Canceled);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                terminate_child(&mut child);
                let _ = read_child_output(stdout_reader, stderr_reader);
                return Ok(RegenOutcome::Shutdown);
            }
        }
    }
}

fn terminate_child(child: &mut Child) {
    if matches!(child.try_wait(), Ok(None)) {
        let _ = child.kill();
    }
    let _ = child.wait();
}

fn read_child_output(
    stdout_reader: thread::JoinHandle<Result<Vec<u8>>>,
    stderr_reader: thread::JoinHandle<Result<Vec<u8>>>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow!("schema dump stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow!("schema dump stderr reader panicked"))??;
    Ok((stdout, stderr))
}

fn read_pipe(mut pipe: impl Read, label: &'static str) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    pipe.read_to_end(&mut output)
        .with_context(|| format!("failed to read {label}"))?;
    Ok(output)
}

fn spawn_schema_dump(command: &SchemaDumpCommand) -> Result<Child> {
    let mut command = match command {
        SchemaDumpCommand::PythonApp { python, app } => python_app_command(python, app)?,
        SchemaDumpCommand::Direct(parts) => direct_schema_dump_command(parts)?,
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn schema dump subprocess")
}

fn python_app_command(python: &OsString, app: &str) -> Result<Command> {
    let (module, attr) = parse_app(app)?;
    let script = format!(
        "import sys; sys.path.insert(0, '.'); from {module} import {attr}; print({attr}.dump_schema_json())"
    );
    let mut command = Command::new(python);
    command.arg("-c").arg(script);
    Ok(command)
}

fn direct_schema_dump_command(parts: &[OsString]) -> Result<Command> {
    let Some(program) = parts.first() else {
        bail!("schema dump command after -- must not be empty");
    };
    let mut command = Command::new(program);
    command.args(&parts[1..]);
    Ok(command)
}

fn dev_dump_command(args: &DevArgs) -> Result<SchemaDumpCommand> {
    if let Some(app) = &args.app {
        return Ok(SchemaDumpCommand::PythonApp {
            python: args.python.clone(),
            app: app.clone(),
        });
    }

    if let Some(command) = infer_python_app_command(&args.app_cmd) {
        return Ok(command);
    }

    if args.app_cmd.is_empty() {
        bail!("zynk dev requires either --app <module:attr> or a schema dump command after --");
    }
    Ok(SchemaDumpCommand::Direct(args.app_cmd.clone()))
}

fn infer_python_app_command(parts: &[OsString]) -> Option<SchemaDumpCommand> {
    let python = parts.first()?;
    let script = parts.get(1)?;
    let script_path = Path::new(script);
    if script_path.extension().and_then(|ext| ext.to_str()) != Some("py") {
        return None;
    }
    let module = script_path.file_stem()?.to_str()?;
    if !is_python_identifier(module) {
        return None;
    }
    Some(SchemaDumpCommand::PythonApp {
        python: python.clone(),
        app: format!("{module}:app"),
    })
}

fn dump_python_schema(args: &GenArgs) -> Result<ApiGraph> {
    let app = args
        .app
        .as_deref()
        .ok_or_else(|| anyhow!("--app <module:attr> is required for --target python"))?;
    let mut command = python_app_command(&args.python, app)?;
    let output = command
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to spawn Python executable {:?}", args.python))?;

    schema_from_output(output.status.success(), &output.stdout, &output.stderr)
}

fn dump_rust_schema(args: &GenArgs) -> Result<ApiGraph> {
    let rust_cmd = args
        .rust_cmd
        .as_deref()
        .ok_or_else(|| anyhow!("--rust-cmd <command> is required for --target rust"))?;
    let mut command = Command::new(rust_cmd);
    command.args(&args.rust_args);
    command.arg("--dump-schema-json");

    let output = command
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to spawn Rust app command {rust_cmd:?}"))?;

    schema_from_output(output.status.success(), &output.stdout, &output.stderr)
}

fn parse_app(app: &str) -> Result<(&str, &str)> {
    let (module, attr) = app
        .split_once(':')
        .ok_or_else(|| anyhow!("--app must use <module>:<attr> format"))?;
    if module.is_empty() || attr.is_empty() {
        bail!("--app must use <module>:<attr> format");
    }
    if !module
        .split('.')
        .all(|part| is_python_identifier(part) && !part.is_empty())
    {
        bail!("--app module must be a dotted Python identifier");
    }
    if !is_python_identifier(attr) {
        bail!("--app attr must be a Python identifier");
    }
    Ok((module, attr))
}

fn is_python_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|char| char == '_' || char.is_ascii_alphanumeric())
}

fn schema_from_output(success: bool, stdout: &[u8], stderr: &[u8]) -> Result<ApiGraph> {
    if !stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(stderr));
    }

    if !success {
        bail!("user app schema dump subprocess failed");
    }

    let stdout = std::str::from_utf8(stdout).context("schema dump stdout was not valid UTF-8")?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        bail!("schema dump stdout was empty");
    }

    serde_json::from_str::<ApiGraph>(trimmed).with_context(|| {
        "failed to parse schema dump stdout as ApiGraph JSON; schema dump must be the only stdout"
    })
}

fn generate_files(language: Language, graph: &ApiGraph) -> GenerationResult {
    match language {
        Language::Typescript => zynk_gen_ts::generate(graph),
        Language::Effect => zynk_gen_effect::generate(graph),
    }
}

fn write_generated_files(result: &GenerationResult, out: &Path) -> Result<()> {
    let output_dir = output_directory(out);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    for file in &result.files {
        let destination = generated_file_destination(out, &output_dir, &file.path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        write_file_atomically(&destination, &file.contents)
            .with_context(|| format!("failed to write {}", destination.display()))?;
    }

    Ok(())
}

fn generated_file_destination(out: &Path, output_dir: &Path, generated_path: &Path) -> PathBuf {
    if out.extension().is_some() && generated_path == Path::new("api.ts") && !out.is_dir() {
        out.to_path_buf()
    } else {
        output_dir.join(generated_path)
    }
}

fn write_file_atomically(destination: &Path, contents: &str) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .ok_or_else(|| anyhow!("output path {} has no file name", destination.display()))?
        .to_string_lossy();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = parent.join(format!(".{file_name}.tmp-{}-{nanos}", std::process::id()));

    {
        let mut temp_file = fs::File::create(&temp_path)
            .with_context(|| format!("failed to create temporary file {}", temp_path.display()))?;
        temp_file
            .write_all(contents.as_bytes())
            .with_context(|| format!("failed to write temporary file {}", temp_path.display()))?;
        temp_file
            .sync_all()
            .with_context(|| format!("failed to sync temporary file {}", temp_path.display()))?;
    }
    if let Err(err) = fs::rename(&temp_path, destination) {
        let _ = fs::remove_file(&temp_path);
        return Err(err).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                destination.display(),
                temp_path.display()
            )
        });
    }
    Ok(())
}

fn output_directory(out: &Path) -> PathBuf {
    if out.is_dir() || out.extension().is_none() {
        out.to_path_buf()
    } else {
        out.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    }
}

fn watched_event_result(result: notify::Result<Event>, config: &DevConfig) -> bool {
    match result {
        Ok(event) => watched_event(&event, config),
        Err(err) => {
            eprintln!("Error: file watcher event failed: {err}");
            false
        }
    }
}

fn watched_event(event: &Event, config: &DevConfig) -> bool {
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event
        .paths
        .iter()
        .any(|path| is_watched_source_path(path, &config.watch_globs))
}

fn is_watched_source_path(path: &Path, globs: &[String]) -> bool {
    if globs.is_empty() {
        return path.extension().and_then(|ext| ext.to_str()) == Some("py");
    }

    let path = path.to_string_lossy();
    globs.iter().any(|glob| glob_matches(glob, &path))
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    if pattern == "*" || pattern == "**/*" {
        return true;
    }
    if let Some((prefix, suffix)) = pattern.split_once('*') {
        return path.starts_with(prefix.trim_start_matches("**/")) && path.ends_with(suffix);
    }
    path.contains(pattern)
}

#[cfg(test)]
mod tests {
    use super::{
        generated_file_destination, infer_python_app_command, is_watched_source_path,
        output_directory, parse_app,
    };
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    #[test]
    fn parse_app_accepts_module_attr() {
        assert_eq!(
            parse_app("kitchen_sink.app:bridge").unwrap(),
            ("kitchen_sink.app", "bridge")
        );
    }

    #[test]
    fn parse_app_rejects_invalid_values() {
        assert!(parse_app("missing_separator").is_err());
        assert!(parse_app("bad-module:app").is_err());
        assert!(parse_app("module:bad.attr").is_err());
    }

    #[test]
    fn out_path_can_be_directory_or_api_file() {
        assert_eq!(
            output_directory(Path::new("/tmp/zynk-gen")),
            PathBuf::from("/tmp/zynk-gen")
        );
        assert_eq!(
            output_directory(Path::new("/tmp/zynk-gen/api.ts")),
            PathBuf::from("/tmp/zynk-gen")
        );
    }

    #[test]
    fn api_file_out_path_controls_api_destination_only() {
        let out = Path::new("/tmp/generated/client.ts");
        let output_dir = output_directory(out);
        assert_eq!(
            generated_file_destination(out, &output_dir, Path::new("api.ts")),
            PathBuf::from("/tmp/generated/client.ts")
        );
        assert_eq!(
            generated_file_destination(out, &output_dir, Path::new("_internal.ts")),
            PathBuf::from("/tmp/generated/_internal.ts")
        );
    }

    #[test]
    fn dev_defaults_to_python_sources_but_accepts_simple_globs() {
        assert!(is_watched_source_path(Path::new("src/app.py"), &[]));
        assert!(!is_watched_source_path(Path::new("src/app.pyc"), &[]));
        assert!(is_watched_source_path(
            Path::new("src/app.rs"),
            &["*.rs".to_string()]
        ));
    }

    #[test]
    fn dev_infers_main_py_as_app_bridge_for_trailing_python_command() {
        let command =
            infer_python_app_command(&[OsString::from("python"), OsString::from("main.py")]);
        assert!(command.is_some());
    }
}
