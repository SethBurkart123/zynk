use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Endpoint, TypeRef};

/// A model field exposed to generated clients.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Field {
    pub source_name: String,
    pub wire_name: String,
    pub ty: TypeRef,
    pub required: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub nullable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

impl Field {
    pub fn new(
        source_name: impl Into<String>,
        wire_name: impl Into<String>,
        ty: TypeRef,
        required: bool,
    ) -> Self {
        let optional = ty.optional;
        let nullable = ty.nullable;

        Self {
            source_name: source_name.into(),
            wire_name: wire_name.into(),
            ty,
            required,
            optional,
            nullable,
            doc: None,
            default: None,
        }
    }
}

/// A structured object definition referenced by endpoint signatures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<Field>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

impl ModelDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            doc: None,
        }
    }
}

/// An enum definition referenced by endpoint signatures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnumDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

impl EnumDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            doc: None,
        }
    }
}

/// Complete language-neutral API graph passed between Zynk modules.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGraph {
    #[serde(default)]
    pub endpoints: BTreeMap<String, Endpoint>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelDef>,
    #[serde(default)]
    pub enums: BTreeMap<String, EnumDef>,
}

impl ApiGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_endpoint(&mut self, endpoint: Endpoint) -> Option<Endpoint> {
        self.endpoints.insert(endpoint.name.clone(), endpoint)
    }

    pub fn insert_model(&mut self, model: ModelDef) -> Option<ModelDef> {
        self.models.insert(model.name.clone(), model)
    }

    pub fn insert_enum(&mut self, enum_def: EnumDef) -> Option<EnumDef> {
        self.enums.insert(enum_def.name.clone(), enum_def)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Endpoint, EndpointKind, Param, TypeRef};

    use super::ApiGraph;

    #[test]
    fn serializes_endpoint_graph_without_io() {
        let mut graph = ApiGraph::new();
        let mut endpoint = Endpoint::new("get_user", EndpointKind::Rpc, TypeRef::model("User"));
        endpoint.params.push(Param::new(
            "user_id",
            "userId",
            TypeRef::primitive("number"),
            true,
        ));
        graph.insert_endpoint(endpoint);

        let json = serde_json::to_string(&graph).expect("serialize graph");
        assert!(json.contains("get_user"));
        assert!(json.contains("userId"));
    }

    #[test]
    fn field_new_populates_optional_nullable_from_type_ref() {
        let mut ty = TypeRef::primitive("string");
        ty.optional = true;
        ty.nullable = true;

        let field = super::Field::new("display_name", "displayName", ty, false);

        assert!(field.optional);
        assert!(field.nullable);
    }

    #[test]
    fn field_optional_true_round_trips_and_skips_false_nullable() {
        let mut field = super::Field::new(
            "display_name",
            "displayName",
            TypeRef::primitive("string"),
            false,
        );
        field.optional = true;
        field.nullable = false;

        let json = serde_json::to_value(&field).expect("serialize field");
        let object = json.as_object().expect("field serializes as object");
        assert_eq!(object.get("optional"), Some(&serde_json::Value::Bool(true)));
        assert!(!object.contains_key("nullable"));

        let round_tripped: super::Field = serde_json::from_value(json).expect("deserialize field");
        assert_eq!(round_tripped, field);
    }

    #[test]
    fn field_missing_optional_nullable_defaults_to_false_and_serializes_without_keys() {
        let json = r#"{
            "sourceName": "display_name",
            "wireName": "displayName",
            "ty": { "kind": "primitive", "name": "string" },
            "required": true
        }"#;

        let field: super::Field = serde_json::from_str(json).expect("deserialize field");
        assert!(!field.optional);
        assert!(!field.nullable);

        let serialized = serde_json::to_value(&field).expect("serialize field");
        let object = serialized.as_object().expect("field serializes as object");
        assert!(!object.contains_key("optional"));
        assert!(!object.contains_key("nullable"));
    }
}
