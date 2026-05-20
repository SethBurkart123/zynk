use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Language-neutral type categories used by Zynk generators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Primitive,
    Array,
    Record,
    Tuple,
    Union,
    Literal,
    Model,
    Enum,
    Any,
    Void,
}

/// A reference to a value type in the Zynk API graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeRef {
    pub kind: TypeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inner: Vec<TypeRef>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub nullable: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_literal_value"
    )]
    pub value: Option<Value>,
}

fn deserialize_optional_literal_value<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Some(Value::deserialize(deserializer)?))
}

impl TypeRef {
    pub fn new(kind: TypeKind) -> Self {
        Self {
            kind,
            name: None,
            inner: Vec::new(),
            optional: false,
            nullable: false,
            value: None,
        }
    }

    pub fn primitive(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::new(TypeKind::Primitive)
        }
    }

    pub fn model(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::new(TypeKind::Model)
        }
    }

    pub fn enum_ref(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::new(TypeKind::Enum)
        }
    }

    pub fn array(item: TypeRef) -> Self {
        Self {
            inner: vec![item],
            ..Self::new(TypeKind::Array)
        }
    }

    pub fn record(key: TypeRef, value: TypeRef) -> Self {
        Self {
            inner: vec![key, value],
            ..Self::new(TypeKind::Record)
        }
    }

    pub fn union(types: Vec<TypeRef>) -> Self {
        Self {
            inner: types,
            ..Self::new(TypeKind::Union)
        }
    }

    pub fn literal(value: Value) -> Self {
        Self {
            value: Some(value),
            ..Self::new(TypeKind::Literal)
        }
    }

    pub fn any() -> Self {
        Self::new(TypeKind::Any)
    }

    pub fn void() -> Self {
        Self::new(TypeKind::Void)
    }
}
