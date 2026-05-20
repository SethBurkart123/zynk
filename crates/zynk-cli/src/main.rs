use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Language {
    Typescript,
    Effect,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Target {
    Python,
    Rust,
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

fn dump_python_schema(args: &GenArgs) -> Result<ApiGraph> {
    let app = args
        .app
        .as_deref()
        .ok_or_else(|| anyhow!("--app <module:attr> is required for --target python"))?;
    let (module, attr) = parse_app(app)?;
    let script = format!(
        "import sys; sys.path.insert(0, '.'); from {module} import {attr}; print({attr}.dump_schema_json())"
    );

    let output = Command::new(&args.python)
        .arg("-c")
        .arg(script)
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
        let destination = output_dir.join(&file.path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        fs::write(&destination, &file.contents)
            .with_context(|| format!("failed to write {}", destination.display()))?;
    }

    Ok(())
}

fn output_directory(out: &Path) -> PathBuf {
    if out.extension().is_some() {
        out.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    } else {
        out.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::{output_directory, parse_app};
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
}
