//! TypeScript type and Effect Schema expression pairs.

/// A lowered TypeScript type with its matching `effect/Schema` expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeExpr {
    /// TypeScript type expression used in generated signatures.
    pub ts: String,
    /// Effect Schema expression used for runtime validation.
    pub schema: String,
}

impl TypeExpr {
    /// Create a lowered type/schema pair.
    pub fn new(ts: impl Into<String>, schema: impl Into<String>) -> Self {
        Self {
            ts: ts.into(),
            schema: schema.into(),
        }
    }
}
