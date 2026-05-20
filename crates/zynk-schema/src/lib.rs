//! Canonical Zynk schema types shared by server bindings and generators.
//!
//! This crate intentionally contains only data structures and small naming
//! helpers. It performs no file IO, runtime routing, or code generation.

pub mod endpoint;
pub mod graph;
pub mod naming;
pub mod types;

pub use endpoint::{Endpoint, EndpointKind, Param};
pub use graph::{ApiGraph, EnumDef, Field, ModelDef};
pub use types::{TypeKind, TypeRef};
