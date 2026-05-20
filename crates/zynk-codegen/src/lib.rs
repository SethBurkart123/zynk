//! Pure in-memory interfaces shared by Zynk code generators.
//!
//! This crate defines generator-neutral data structures and traits only. Callers
//! own all filesystem, process, and network side effects.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zynk_schema::ApiGraph;

/// Inputs made available to a code generator invocation.
#[derive(Debug, Clone, Copy)]
pub struct GenerationContext<'a> {
    /// Canonical API graph to generate from.
    pub graph: &'a ApiGraph,
    /// Generator-specific options, interpreted by concrete generators.
    pub options: &'a Value,
}

impl<'a> GenerationContext<'a> {
    /// Create a generation context for an API graph and arbitrary options.
    pub fn new(graph: &'a ApiGraph, options: &'a Value) -> Self {
        Self { graph, options }
    }
}

/// One generated file, represented entirely in memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedFile {
    /// Relative output path chosen by the generator.
    pub path: PathBuf,
    /// Complete UTF-8 source contents for the generated file.
    pub contents: String,
}

impl GeneratedFile {
    /// Create an in-memory generated file.
    pub fn new(path: impl Into<PathBuf>, contents: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            contents: contents.into(),
        }
    }
}

/// In-memory output from a generator invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated files. The caller decides whether and where to write them.
    pub files: Vec<GeneratedFile>,
}

impl GenerationResult {
    /// Create a generation result from in-memory files.
    pub fn new(files: Vec<GeneratedFile>) -> Self {
        Self { files }
    }
}

/// A pure code generator from an API graph to generated files.
pub trait Generator {
    /// Generate files in memory from the provided context.
    fn generate(&self, ctx: &GenerationContext) -> GenerationResult;
}

/// Name-based registry for generator implementations.
#[derive(Default)]
pub struct GeneratorRegistry {
    generators: BTreeMap<String, Box<dyn Generator>>,
}

impl GeneratorRegistry {
    /// Create an empty generator registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a generator by name, replacing and returning any previous entry.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        generator: Box<dyn Generator>,
    ) -> Option<Box<dyn Generator>> {
        self.generators.insert(name.into(), generator)
    }

    /// Look up a generator by name.
    pub fn get(&self, name: &str) -> Option<&dyn Generator> {
        self.generators.get(name).map(Box::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;
    use zynk_schema::{ApiGraph, Endpoint, EndpointKind, TypeRef};

    use super::{GeneratedFile, GenerationContext, GenerationResult, Generator, GeneratorRegistry};

    struct TinyGenerator;

    impl Generator for TinyGenerator {
        fn generate(&self, ctx: &GenerationContext) -> GenerationResult {
            let endpoint_count = ctx.graph.endpoints.len();
            let suffix = ctx
                .options
                .get("suffix")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("txt");

            GenerationResult::new(vec![GeneratedFile::new(
                PathBuf::from(format!("api.{suffix}")),
                format!("endpoints={endpoint_count}"),
            )])
        }
    }

    #[test]
    fn generator_trait_uses_context_and_returns_generated_files() {
        let mut graph = ApiGraph::new();
        graph.insert_endpoint(Endpoint::new("ping", EndpointKind::Rpc, TypeRef::void()));
        let options = json!({ "suffix": "ts" });
        let ctx = GenerationContext::new(&graph, &options);

        let result = TinyGenerator.generate(&ctx);

        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].path, PathBuf::from("api.ts"));
        assert_eq!(result.files[0].contents, "endpoints=1");
    }

    #[test]
    fn registry_registers_and_looks_up_generators_by_name() {
        let mut registry = GeneratorRegistry::new();
        assert!(registry.get("typescript").is_none());

        assert!(registry
            .register("typescript", Box::new(TinyGenerator))
            .is_none());

        let graph = ApiGraph::new();
        let options = json!({ "suffix": "txt" });
        let ctx = GenerationContext::new(&graph, &options);
        let generator = registry.get("typescript").expect("registered generator");
        let result = generator.generate(&ctx);

        assert_eq!(
            result.files,
            vec![GeneratedFile::new("api.txt", "endpoints=0")]
        );
        assert!(registry.get("effect").is_none());
    }
}
