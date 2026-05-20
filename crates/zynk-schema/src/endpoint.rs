use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::TypeRef;

/// Supported endpoint surfaces. These map to Zynk's stable wire routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Rpc,
    Channel,
    Upload,
    Static,
    Ws,
}

/// A callable parameter exposed to generated clients.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Param {
    pub source_name: String,
    pub wire_name: String,
    pub ty: TypeRef,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

impl Param {
    pub fn new(
        source_name: impl Into<String>,
        wire_name: impl Into<String>,
        ty: TypeRef,
        required: bool,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            wire_name: wire_name.into(),
            ty,
            required,
            default: None,
        }
    }
}

/// One server endpoint in the Zynk API graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Endpoint {
    pub name: String,
    pub kind: EndpointKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    pub returns: TypeRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_item: Option<TypeRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_param: Option<String>,
    #[serde(default)]
    pub multi_file: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_events: Vec<Param>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub client_events: Vec<Param>,
}

impl Endpoint {
    pub fn new(name: impl Into<String>, kind: EndpointKind, returns: TypeRef) -> Self {
        Self {
            name: name.into(),
            kind,
            module: None,
            doc: None,
            params: Vec::new(),
            returns,
            channel_item: None,
            file_param: None,
            multi_file: false,
            max_size: None,
            allowed_types: Vec::new(),
            server_events: Vec::new(),
            client_events: Vec::new(),
        }
    }
}
