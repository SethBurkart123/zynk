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

    use super::{ApiGraph, EnumDef, Field, ModelDef};

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

    #[test]
    fn api_graph_round_trips_all_endpoint_and_type_features() {
        let mut graph = ApiGraph::new();

        let mut priority = EnumDef::new("Priority");
        priority.doc = Some("Explicit enum values for code generators.".to_string());
        priority.values = vec![
            serde_json::json!("low"),
            serde_json::json!("medium"),
            serde_json::json!("high"),
            serde_json::json!(3),
        ];
        graph.insert_enum(priority);

        let mut metadata = ModelDef::new("Metadata");
        metadata.doc = Some("Nested metadata model.".to_string());
        metadata.fields.push(Field::new(
            "created_at",
            "createdAt",
            TypeRef::primitive("string"),
            true,
        ));
        metadata.fields.push(Field::new(
            "tags",
            "tags",
            TypeRef::array(TypeRef::primitive("string")),
            true,
        ));
        metadata.fields.push(Field::new(
            "attributes",
            "attributes",
            TypeRef::record(TypeRef::primitive("string"), TypeRef::any()),
            false,
        ));
        graph.insert_model(metadata);

        let mut user = ModelDef::new("User");
        user.doc = Some("User model with nested, enum, and literal fields.".to_string());
        user.fields
            .push(Field::new("id", "id", TypeRef::primitive("number"), true));
        user.fields.push(Field::new(
            "metadata",
            "metadata",
            TypeRef::model("Metadata"),
            true,
        ));
        user.fields.push(Field::new(
            "priority",
            "priority",
            TypeRef::enum_ref("Priority"),
            true,
        ));
        user.fields.push(Field::new(
            "role",
            "role",
            TypeRef::literal(serde_json::json!("admin")),
            true,
        ));
        user.fields.push(Field::new(
            "retries",
            "retries",
            TypeRef::literal(serde_json::json!(3)),
            true,
        ));
        user.fields.push(Field::new(
            "enabled",
            "enabled",
            TypeRef::literal(serde_json::json!(true)),
            true,
        ));
        user.fields.push(Field::new(
            "deleted_marker",
            "deletedMarker",
            TypeRef::literal(serde_json::Value::Null),
            false,
        ));

        let mut optional_but_not_nullable = TypeRef::primitive("string");
        optional_but_not_nullable.optional = true;
        user.fields.push(Field::new(
            "display_name",
            "displayName",
            optional_but_not_nullable,
            false,
        ));

        let mut nullable_but_required = TypeRef::model("Metadata");
        nullable_but_required.nullable = true;
        user.fields.push(Field::new(
            "secondary_metadata",
            "secondaryMetadata",
            nullable_but_required,
            true,
        ));

        let mut optional_and_nullable = TypeRef::union(vec![
            TypeRef::literal(serde_json::json!("email")),
            TypeRef::literal(serde_json::json!("sms")),
        ]);
        optional_and_nullable.optional = true;
        optional_and_nullable.nullable = true;
        let mut preferred_contact = Field::new(
            "preferred_contact",
            "preferredContact",
            optional_and_nullable,
            false,
        );
        preferred_contact.doc = Some("Optional nullable union of literal values.".to_string());
        preferred_contact.default = Some(serde_json::json!("email"));
        user.fields.push(preferred_contact);

        graph.insert_model(user);

        let mut rpc = Endpoint::new("get_user", EndpointKind::Rpc, TypeRef::model("User"));
        rpc.module = Some("users".to_string());
        rpc.doc = Some("Fetches a user.".to_string());
        rpc.params.push(Param::new(
            "user_id",
            "userId",
            TypeRef::primitive("number"),
            true,
        ));
        let mut include_deleted = Param::new(
            "include_deleted",
            "includeDeleted",
            TypeRef::primitive("boolean"),
            false,
        );
        include_deleted.default = Some(serde_json::json!(false));
        rpc.params.push(include_deleted);
        graph.insert_endpoint(rpc);

        let mut channel = Endpoint::new("watch_users", EndpointKind::Channel, TypeRef::void());
        channel.channel_item = Some(TypeRef::model("User"));
        channel.params.push(Param::new(
            "priority",
            "priority",
            TypeRef::enum_ref("Priority"),
            false,
        ));
        graph.insert_endpoint(channel);

        let mut upload = Endpoint::new(
            "upload_avatar",
            EndpointKind::Upload,
            TypeRef::model("User"),
        );
        upload.file_param = Some("avatar".to_string());
        upload.multi_file = false;
        upload.max_size = Some(10_485_760);
        upload.allowed_types = vec!["image/png".to_string(), "image/jpeg".to_string()];
        upload.params.push(Param::new(
            "user_id",
            "userId",
            TypeRef::primitive("number"),
            true,
        ));
        graph.insert_endpoint(upload);

        let mut static_endpoint = Endpoint::new(
            "download_report",
            EndpointKind::Static,
            TypeRef::primitive("string"),
        );
        static_endpoint.params.push(Param::new(
            "report_id",
            "reportId",
            TypeRef::primitive("string"),
            true,
        ));
        graph.insert_endpoint(static_endpoint);

        let mut websocket = Endpoint::new("user_socket", EndpointKind::Ws, TypeRef::void());
        websocket.server_events.push(Param::new(
            "user_updated",
            "userUpdated",
            TypeRef::model("User"),
            true,
        ));
        websocket.server_events.push(Param::new(
            "heartbeat",
            "heartbeat",
            TypeRef::literal(serde_json::json!("ok")),
            true,
        ));
        websocket.client_events.push(Param::new(
            "subscribe",
            "subscribe",
            TypeRef::record(TypeRef::primitive("string"), TypeRef::any()),
            true,
        ));
        websocket.client_events.push(Param::new(
            "set_priority",
            "setPriority",
            TypeRef::enum_ref("Priority"),
            false,
        ));
        graph.insert_endpoint(websocket);

        let json = serde_json::to_string_pretty(&graph).expect("serialize graph");
        let round_tripped: ApiGraph = serde_json::from_str(&json).expect("deserialize graph");

        assert_eq!(round_tripped, graph);
        assert_eq!(graph.endpoints.len(), 5);
        assert!(graph
            .endpoints
            .values()
            .any(|endpoint| endpoint.kind == EndpointKind::Rpc));
        assert!(graph
            .endpoints
            .values()
            .any(|endpoint| endpoint.kind == EndpointKind::Channel));
        assert!(graph
            .endpoints
            .values()
            .any(|endpoint| endpoint.kind == EndpointKind::Upload));
        assert!(graph
            .endpoints
            .values()
            .any(|endpoint| endpoint.kind == EndpointKind::Static));
        assert!(graph
            .endpoints
            .values()
            .any(|endpoint| endpoint.kind == EndpointKind::Ws));

        let user = graph.models.get("User").expect("User model exists");
        assert!(user.fields.iter().any(|field| {
            field.wire_name == "displayName" && field.optional && !field.nullable
        }));
        assert!(user.fields.iter().any(|field| {
            field.wire_name == "secondaryMetadata" && !field.optional && field.nullable
        }));
        assert!(user.fields.iter().any(|field| {
            field.wire_name == "preferredContact" && field.optional && field.nullable
        }));
    }
}
