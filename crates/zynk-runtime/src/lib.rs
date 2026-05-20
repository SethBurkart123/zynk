//! Pure wire-contract types for Zynk server runtimes.
//!
//! This crate deliberately contains no HTTP framework integration. Server
//! bindings such as Axum use these data structures to register endpoints,
//! dispatch type-erased handlers, and preserve Zynk's JSON/SSE/WebSocket wire
//! shapes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use zynk_schema::{TypeKind, TypeRef};

pub use zynk_schema::EndpointKind;

/// Request validation failed.
pub const VALIDATION_ERROR: &str = "VALIDATION_ERROR";
/// Requested RPC/channel command was not registered.
pub const COMMAND_NOT_FOUND: &str = "COMMAND_NOT_FOUND";
/// Endpoint handler returned an execution error.
pub const EXECUTION_ERROR: &str = "EXECUTION_ERROR";
/// Channel operation failed.
pub const CHANNEL_ERROR: &str = "CHANNEL_ERROR";
/// Unexpected internal runtime error.
pub const INTERNAL_ERROR: &str = "INTERNAL_ERROR";
/// WebSocket operation failed.
pub const WEBSOCKET_ERROR: &str = "WEBSOCKET_ERROR";
/// Requested WebSocket message handler was not registered.
pub const HANDLER_NOT_FOUND: &str = "HANDLER_NOT_FOUND";
/// Requested upload handler was not registered.
pub const UPLOAD_HANDLER_NOT_FOUND: &str = "UPLOAD_HANDLER_NOT_FOUND";
/// Upload request or file validation failed.
pub const UPLOAD_VALIDATION_ERROR: &str = "UPLOAD_VALIDATION_ERROR";
/// Requested static file handler was not registered.
pub const STATIC_HANDLER_NOT_FOUND: &str = "STATIC_HANDLER_NOT_FOUND";

/// All Python-compatible error code spellings in canonical order.
pub const ERROR_CODES: [&str; 10] = [
    VALIDATION_ERROR,
    COMMAND_NOT_FOUND,
    EXECUTION_ERROR,
    CHANNEL_ERROR,
    INTERNAL_ERROR,
    WEBSOCKET_ERROR,
    HANDLER_NOT_FOUND,
    UPLOAD_HANDLER_NOT_FOUND,
    UPLOAD_VALIDATION_ERROR,
    STATIC_HANDLER_NOT_FOUND,
];

/// Success envelope for JSON endpoint responses: `{ "result": T }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonResultEnvelope<T> {
    /// Successful endpoint result payload.
    pub result: T,
}

impl<T> JsonResultEnvelope<T> {
    /// Wrap a successful result in Zynk's JSON envelope.
    pub fn new(result: T) -> Self {
        Self { result }
    }
}

/// Error envelope payload for JSON endpoint failures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonErrorEnvelope {
    /// Machine-readable Zynk error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// Optional structured error details, omitted from JSON when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl JsonErrorEnvelope {
    /// Create an error envelope payload.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    /// Attach structured details to an error envelope payload.
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Outer error response shape for surfaces that wrap errors under `error`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonErrorResponseEnvelope {
    /// Error payload.
    pub error: JsonErrorEnvelope,
}

impl JsonErrorResponseEnvelope {
    /// Wrap an error payload in the outer error response envelope.
    pub fn new(error: JsonErrorEnvelope) -> Self {
        Self { error }
    }
}

/// Runtime error type that can be converted into a JSON error envelope.
#[derive(Debug, Clone, PartialEq, Error)]
#[error("{code}: {message}")]
pub struct ZynkError {
    /// Machine-readable Zynk error code.
    pub code: &'static str,
    /// Human-readable error message.
    pub message: String,
    /// Optional structured error details.
    pub details: Option<Value>,
}

impl ZynkError {
    /// Create a runtime error without structured details.
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }

    /// Create a runtime error with structured details.
    pub fn with_details(code: &'static str, message: impl Into<String>, details: Value) -> Self {
        Self {
            code,
            message: message.into(),
            details: Some(details),
        }
    }

    /// Convert this runtime error into the JSON error envelope payload.
    pub fn into_envelope(self) -> JsonErrorEnvelope {
        JsonErrorEnvelope {
            code: self.code.to_string(),
            message: self.message,
            details: self.details,
        }
    }
}

/// Type-erased endpoint handler contract used by server bindings.
pub trait Handler: Send + Sync + 'static {
    /// Invoke the handler with a JSON object payload and return a JSON value.
    fn call(&self, payload: Value) -> Result<Value, ZynkError>;
}

impl<F> Handler for F
where
    F: Fn(Value) -> Result<Value, ZynkError> + Send + Sync + 'static,
{
    fn call(&self, payload: Value) -> Result<Value, ZynkError> {
        self(payload)
    }
}

/// Registration key for a type-erased handler stored by a server binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandlerKey(pub &'static str);

/// Static type metadata used by proc macros in link-time endpoint registration.
#[derive(Debug, Clone, PartialEq)]
pub enum StaticValue {
    /// JSON null literal.
    Null,
    /// Boolean literal.
    Bool(bool),
    /// Signed integer literal.
    I64(i64),
    /// Unsigned integer literal.
    U64(u64),
    /// Floating-point literal.
    F64(f64),
    /// String literal.
    Str(&'static str),
}

impl StaticValue {
    /// Convert static literal metadata into an owned JSON value.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::I64(value) => Value::Number((*value).into()),
            Self::U64(value) => Value::Number((*value).into()),
            Self::F64(value) => serde_json::Number::from_f64(*value)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            Self::Str(value) => Value::String((*value).to_string()),
        }
    }
}

/// Static type metadata used by proc macros in link-time endpoint registration.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeRefStatic {
    /// Language-neutral type category.
    pub kind: TypeKind,
    /// Primitive/model/enum name, when applicable.
    pub name: Option<&'static str>,
    /// Nested type references for arrays, records, tuples, and unions.
    pub inner: &'static [TypeRefStatic],
    /// Whether the value may be omitted.
    pub optional: bool,
    /// Whether the value may be null.
    pub nullable: bool,
    /// Literal value metadata, when applicable.
    pub value: Option<StaticValue>,
}

impl TypeRefStatic {
    /// Return a copy of this metadata with optionality enabled.
    pub const fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Return a copy of this metadata with nullability enabled.
    pub const fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    /// Convert static metadata into the canonical schema type reference.
    pub fn to_schema_type_ref(&self) -> TypeRef {
        TypeRef {
            kind: self.kind.clone(),
            name: self.name.map(str::to_string),
            inner: self
                .inner
                .iter()
                .map(TypeRefStatic::to_schema_type_ref)
                .collect(),
            optional: self.optional,
            nullable: self.nullable,
            value: self.value.as_ref().map(StaticValue::to_json),
        }
    }

    /// Construct static type metadata for a primitive type.
    pub const fn primitive(name: &'static str) -> Self {
        Self {
            kind: TypeKind::Primitive,
            name: Some(name),
            inner: &[],
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for a model reference.
    pub const fn model(name: &'static str) -> Self {
        Self {
            kind: TypeKind::Model,
            name: Some(name),
            inner: &[],
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for an enum reference.
    pub const fn enum_ref(name: &'static str) -> Self {
        Self {
            kind: TypeKind::Enum,
            name: Some(name),
            inner: &[],
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for an array item type.
    ///
    /// Pass a one-element slice containing the item type.
    pub const fn array(item: &'static [TypeRefStatic]) -> Self {
        Self {
            kind: TypeKind::Array,
            name: None,
            inner: item,
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for a union of member types.
    pub const fn union(members: &'static [TypeRefStatic]) -> Self {
        Self {
            kind: TypeKind::Union,
            name: None,
            inner: members,
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for a literal value.
    pub const fn literal(value: StaticValue) -> Self {
        Self {
            kind: TypeKind::Literal,
            name: None,
            inner: &[],
            optional: false,
            nullable: false,
            value: Some(value),
        }
    }

    /// Construct static type metadata for an unconstrained JSON value.
    pub const fn any() -> Self {
        Self {
            kind: TypeKind::Any,
            name: None,
            inner: &[],
            optional: false,
            nullable: false,
            value: None,
        }
    }

    /// Construct static type metadata for a void return type.
    pub const fn void() -> Self {
        Self {
            kind: TypeKind::Void,
            name: None,
            inner: &[],
            optional: false,
            nullable: false,
            value: None,
        }
    }
}

/// Static parameter metadata used by `EndpointMeta`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamMeta {
    /// Source-language parameter name.
    pub source_name: &'static str,
    /// Wire-format parameter name.
    pub wire_name: &'static str,
    /// Static type metadata for this parameter.
    pub ty: TypeRefStatic,
    /// Whether this parameter is required.
    pub required: bool,
    /// Optional static JSON default value.
    pub default: Option<StaticValue>,
}

/// Link-time endpoint registration metadata produced by proc macros.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointMeta {
    /// Stable endpoint name.
    pub name: &'static str,
    /// Endpoint surface kind.
    pub kind: EndpointKind,
    /// Optional source module for introspection/schema output.
    pub module: Option<&'static str>,
    /// Optional documentation text.
    pub doc: Option<&'static str>,
    /// Callable parameter metadata.
    pub params: &'static [ParamMeta],
    /// Return type metadata.
    pub returns: TypeRefStatic,
    /// Channel item type for channel endpoints.
    pub channel_item: Option<TypeRefStatic>,
    /// Upload file parameter name for upload endpoints.
    pub file_param: Option<&'static str>,
    /// Whether an upload endpoint accepts multiple files.
    pub multi_file: bool,
    /// Maximum upload size in bytes.
    pub max_size: Option<u64>,
    /// Allowed upload content types.
    pub allowed_types: &'static [&'static str],
    /// WebSocket server event metadata.
    pub server_events: &'static [ParamMeta],
    /// WebSocket client event metadata.
    pub client_events: &'static [ParamMeta],
    /// Type-erased handler registration key for dispatch tables.
    pub handler_key: Option<HandlerKey>,
}

inventory::collect!(EndpointMeta);

/// One Server-Sent Events frame: `event: <name>` / `data: <json>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SseFrame {
    /// SSE event name.
    pub event: String,
    /// JSON payload for the `data:` line.
    pub data: Value,
}

impl SseFrame {
    /// Create an SSE frame.
    pub fn new(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }

    /// Encode this frame in the exact Zynk SSE wire format.
    pub fn encode(&self) -> String {
        format!("event: {}\ndata: {}\n\n", self.event, self.data)
    }
}

/// One WebSocket JSON text-frame message: `{ "event": name, "data": payload }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WsMessage {
    /// Snake-case event name.
    pub event: String,
    /// JSON payload for this event.
    pub data: Value,
}

impl WsMessage {
    /// Create a WebSocket message.
    pub fn new(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }

    /// Parse a WebSocket JSON frame, defaulting missing `event` to `message`
    /// and missing `data` to `{}` to match the Python runtime.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let mut parsed: Value = serde_json::from_str(json)?;
        let event = parsed
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or("message")
            .to_string();
        let data = parsed
            .as_object_mut()
            .and_then(|object| object.remove("data"))
            .unwrap_or_else(|| serde_json::json!({}));

        Ok(Self { event, data })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use zynk_schema::TypeKind;

    use super::{
        EndpointMeta, Handler, HandlerKey, JsonErrorEnvelope, JsonErrorResponseEnvelope,
        JsonResultEnvelope, ParamMeta, SseFrame, StaticValue, TypeRefStatic, WsMessage,
        CHANNEL_ERROR, COMMAND_NOT_FOUND, ERROR_CODES, EXECUTION_ERROR, HANDLER_NOT_FOUND,
        INTERNAL_ERROR, STATIC_HANDLER_NOT_FOUND, UPLOAD_HANDLER_NOT_FOUND,
        UPLOAD_VALIDATION_ERROR, VALIDATION_ERROR, WEBSOCKET_ERROR,
    };

    #[test]
    fn error_code_constants_match_python_literals_verbatim() {
        assert_eq!(VALIDATION_ERROR, "VALIDATION_ERROR");
        assert_eq!(COMMAND_NOT_FOUND, "COMMAND_NOT_FOUND");
        assert_eq!(EXECUTION_ERROR, "EXECUTION_ERROR");
        assert_eq!(CHANNEL_ERROR, "CHANNEL_ERROR");
        assert_eq!(INTERNAL_ERROR, "INTERNAL_ERROR");
        assert_eq!(WEBSOCKET_ERROR, "WEBSOCKET_ERROR");
        assert_eq!(HANDLER_NOT_FOUND, "HANDLER_NOT_FOUND");
        assert_eq!(UPLOAD_HANDLER_NOT_FOUND, "UPLOAD_HANDLER_NOT_FOUND");
        assert_eq!(UPLOAD_VALIDATION_ERROR, "UPLOAD_VALIDATION_ERROR");
        assert_eq!(STATIC_HANDLER_NOT_FOUND, "STATIC_HANDLER_NOT_FOUND");
        assert_eq!(ERROR_CODES.len(), 10);
        assert_eq!(
            ERROR_CODES,
            [
                "VALIDATION_ERROR",
                "COMMAND_NOT_FOUND",
                "EXECUTION_ERROR",
                "CHANNEL_ERROR",
                "INTERNAL_ERROR",
                "WEBSOCKET_ERROR",
                "HANDLER_NOT_FOUND",
                "UPLOAD_HANDLER_NOT_FOUND",
                "UPLOAD_VALIDATION_ERROR",
                "STATIC_HANDLER_NOT_FOUND",
            ]
        );
    }

    #[test]
    fn json_success_envelope_serializes_to_python_wire_shape() {
        let envelope = JsonResultEnvelope::new(json!({"id": 1, "name": "ada"}));

        let encoded = serde_json::to_string(&envelope).expect("serialize result envelope");

        assert_eq!(encoded, r#"{"result":{"id":1,"name":"ada"}}"#);
    }

    #[test]
    fn json_error_envelope_omits_absent_details_like_python() {
        let envelope = JsonErrorEnvelope::new(VALIDATION_ERROR, "bad input");

        let encoded = serde_json::to_string(&envelope).expect("serialize error envelope");

        assert_eq!(
            encoded,
            r#"{"code":"VALIDATION_ERROR","message":"bad input"}"#
        );
    }

    #[test]
    fn json_error_envelope_includes_details_when_present() {
        let envelope = JsonErrorEnvelope::new(COMMAND_NOT_FOUND, "missing")
            .with_details(json!({"command": "missing"}));

        let encoded = serde_json::to_string(&envelope).expect("serialize error envelope");

        assert_eq!(
            encoded,
            r#"{"code":"COMMAND_NOT_FOUND","message":"missing","details":{"command":"missing"}}"#
        );
    }

    #[test]
    fn json_outer_error_response_envelope_serializes_when_needed() {
        let envelope =
            JsonErrorResponseEnvelope::new(JsonErrorEnvelope::new(VALIDATION_ERROR, "bad input"));

        let encoded = serde_json::to_string(&envelope).expect("serialize wrapped error envelope");

        assert_eq!(
            encoded,
            r#"{"error":{"code":"VALIDATION_ERROR","message":"bad input"}}"#
        );
    }

    #[test]
    fn sse_frame_encodes_event_and_json_data_lines() {
        let frame = SseFrame::new("message", json!({"x": 1}));

        assert_eq!(frame.encode(), "event: message\ndata: {\"x\":1}\n\n");
    }

    #[test]
    fn sse_frame_serializes_to_structure_fields() {
        let frame = SseFrame::new("close", json!({"channelId": "abc"}));

        let encoded = serde_json::to_string(&frame).expect("serialize sse frame");

        assert_eq!(encoded, r#"{"event":"close","data":{"channelId":"abc"}}"#);
    }

    #[test]
    fn sse_frame_json_encodes_string_payloads() {
        let frame = SseFrame::new("message", json!("hello"));

        assert_eq!(frame.encode(), "event: message\ndata: \"hello\"\n\n");
    }

    #[test]
    fn websocket_message_serializes_to_wire_shape() {
        let message = WsMessage::new("chat_message", json!({"body": "hi"}));

        let encoded = serde_json::to_string(&message).expect("serialize websocket message");

        assert_eq!(encoded, r#"{"event":"chat_message","data":{"body":"hi"}}"#);
    }

    #[test]
    fn websocket_message_from_json_defaults_missing_fields_like_python() {
        assert_eq!(
            WsMessage::from_json(r#"{"data":{"x":1}}"#).expect("parse missing event"),
            WsMessage::new("message", json!({"x": 1}))
        );
        assert_eq!(
            WsMessage::from_json(r#"{"event":"join"}"#).expect("parse missing data"),
            WsMessage::new("join", json!({}))
        );
        assert_eq!(
            WsMessage::from_json(r#"{}"#).expect("parse empty object"),
            WsMessage::new("message", json!({}))
        );
    }

    #[test]
    fn handler_trait_accepts_type_erased_json_invocation() {
        let handler = |payload: serde_json::Value| {
            Ok(json!({
                "echo": payload,
            }))
        };

        let result = Handler::call(&handler, json!({"name": "Ada"})).expect("handler succeeds");

        assert_eq!(result, json!({"echo": {"name": "Ada"}}));
    }

    #[test]
    fn endpoint_meta_carries_route_registration_fields() {
        static PARAMS: &[ParamMeta] = &[ParamMeta {
            source_name: "display_name",
            wire_name: "displayName",
            ty: TypeRefStatic::primitive("string"),
            required: true,
            default: None,
        }];
        static RETURNS: TypeRefStatic = TypeRefStatic::model("User");
        static CHANNEL_ITEM: TypeRefStatic = TypeRefStatic::model("User");
        static SERVER_EVENTS: &[ParamMeta] = &[ParamMeta {
            source_name: "user_updated",
            wire_name: "userUpdated",
            ty: TypeRefStatic::model("User"),
            required: true,
            default: None,
        }];
        static CLIENT_EVENTS: &[ParamMeta] = &[ParamMeta {
            source_name: "subscribe_user",
            wire_name: "subscribeUser",
            ty: TypeRefStatic::primitive("number"),
            required: true,
            default: None,
        }];

        let endpoint = EndpointMeta {
            name: "get_user",
            kind: zynk_schema::EndpointKind::Upload,
            module: Some("users"),
            doc: Some("Fetches a user"),
            params: PARAMS,
            returns: RETURNS.clone(),
            channel_item: Some(CHANNEL_ITEM.clone()),
            file_param: Some("avatar"),
            multi_file: false,
            max_size: Some(1_048_576),
            allowed_types: &["image/png", "image/jpeg"],
            server_events: SERVER_EVENTS,
            client_events: CLIENT_EVENTS,
            handler_key: Some(HandlerKey("users::get_user")),
        };

        assert_eq!(endpoint.name, "get_user");
        assert_eq!(endpoint.params[0].source_name, "display_name");
        assert_eq!(endpoint.params[0].wire_name, "displayName");
        assert_eq!(endpoint.params[0].ty.kind, TypeKind::Primitive);
        assert_eq!(endpoint.returns.kind, TypeKind::Model);
        assert_eq!(endpoint.file_param, Some("avatar"));
        assert_eq!(endpoint.allowed_types, ["image/png", "image/jpeg"]);
        assert_eq!(endpoint.server_events[0].wire_name, "userUpdated");
        assert_eq!(endpoint.client_events[0].source_name, "subscribe_user");
        assert_eq!(endpoint.handler_key, Some(HandlerKey("users::get_user")));
    }

    #[test]
    fn type_ref_static_converts_to_schema_type_ref() {
        static INNER: &[TypeRefStatic] = &[
            TypeRefStatic::primitive("string"),
            TypeRefStatic {
                kind: TypeKind::Literal,
                name: None,
                inner: &[],
                optional: false,
                nullable: false,
                value: Some(StaticValue::Str("admin")),
            },
        ];
        let static_ref = TypeRefStatic {
            kind: TypeKind::Union,
            name: None,
            inner: INNER,
            optional: true,
            nullable: true,
            value: None,
        };

        let schema_ref = static_ref.to_schema_type_ref();

        assert_eq!(schema_ref.kind, TypeKind::Union);
        assert!(schema_ref.optional);
        assert!(schema_ref.nullable);
        assert_eq!(schema_ref.inner.len(), 2);
        assert_eq!(schema_ref.inner[1].value, Some(json!("admin")));
    }

    #[test]
    fn zynk_error_converts_into_json_error_envelope() {
        let error = super::ZynkError::with_details(
            CHANNEL_ERROR,
            "channel closed",
            json!({"channel_id": "abc"}),
        );

        let envelope = error.into_envelope();

        assert_eq!(envelope.code, CHANNEL_ERROR);
        assert_eq!(envelope.message, "channel closed");
        assert_eq!(envelope.details, Some(json!({"channel_id": "abc"})));
    }
}
