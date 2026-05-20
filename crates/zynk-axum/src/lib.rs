//! Axum server binding for Zynk Rust endpoints.
//!
//! `ZynkBridge` consumes endpoint metadata collected through `inventory` and
//! registers Zynk wire-contract routes on an existing Axum router.

use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use zynk_runtime::{
    inventory, EndpointKind, EndpointMeta, Handler, HandlerKey, JsonErrorEnvelope,
    JsonResultEnvelope, ParamMeta, ZynkError, COMMAND_NOT_FOUND, EXECUTION_ERROR, INTERNAL_ERROR,
    VALIDATION_ERROR,
};

/// Axum integration point for Zynk endpoints.
#[derive(Clone)]
pub struct ZynkBridge {
    state: Arc<BridgeState>,
}

struct BridgeState {
    endpoints: BTreeMap<String, &'static EndpointMeta>,
    handlers: HashMap<HandlerKey, Arc<dyn Handler>>,
    debug: bool,
}

impl ZynkBridge {
    /// Create a bridge from all link-time registered endpoint metadata.
    pub fn new() -> Self {
        Self::from_inventory()
    }

    /// Create a bridge with debug error output enabled or disabled.
    pub fn with_debug(debug: bool) -> Self {
        Self::from_parts(collect_inventory_endpoints(), HashMap::new(), debug)
    }

    /// Return a copy of this bridge with debug error output enabled or disabled.
    pub fn debug(mut self, debug: bool) -> Self {
        Arc::make_mut(&mut self.state).debug = debug;
        self
    }

    /// Register a type-erased handler for an endpoint metadata `HandlerKey`.
    ///
    /// The current macro layer publishes stable handler keys in inventory. This
    /// method connects those keys to executable handler adapters while keeping
    /// code generation and HTTP framework details out of `zynk-runtime`.
    pub fn register_handler<H>(mut self, key: HandlerKey, handler: H) -> Self
    where
        H: Handler,
    {
        Arc::make_mut(&mut self.state)
            .handlers
            .insert(key, Arc::new(handler));
        self
    }

    /// Register a handler by its endpoint name when that name is unique.
    pub fn register_command<H>(mut self, name: &str, handler: H) -> Self
    where
        H: Handler,
    {
        if let Some(endpoint) = self.state.endpoints.get(name) {
            if let Some(key) = endpoint.handler_key {
                Arc::make_mut(&mut self.state)
                    .handlers
                    .insert(key, Arc::new(handler));
            }
        }
        self
    }

    /// Register Zynk routes on an existing Axum router.
    pub fn configure(self, router: Router) -> Router {
        router.route(
            "/command/{name}",
            post(command_route).with_state(self.state),
        )
    }

    /// Dump the canonical `zynk-schema` API graph JSON for registered endpoints.
    pub fn dump_schema_json(&self) -> String {
        serde_json::to_string(&self.api_graph()).expect("ApiGraph serialization cannot fail")
    }

    /// Build the canonical `zynk-schema` API graph for registered endpoints.
    pub fn api_graph(&self) -> zynk_runtime::zynk_schema::ApiGraph {
        let mut graph = zynk_runtime::zynk_schema::ApiGraph::new();
        for endpoint in self.state.endpoints.values() {
            graph.insert_endpoint(endpoint_to_schema(endpoint));
        }
        graph
    }

    fn from_inventory() -> Self {
        Self::from_parts(collect_inventory_endpoints(), HashMap::new(), false)
    }

    fn from_parts(
        endpoints: BTreeMap<String, &'static EndpointMeta>,
        handlers: HashMap<HandlerKey, Arc<dyn Handler>>,
        debug: bool,
    ) -> Self {
        Self {
            state: Arc::new(BridgeState {
                endpoints,
                handlers,
                debug,
            }),
        }
    }
}

impl Default for ZynkBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BridgeState {
    fn clone(&self) -> Self {
        Self {
            endpoints: self.endpoints.clone(),
            handlers: self.handlers.clone(),
            debug: self.debug,
        }
    }
}

fn collect_inventory_endpoints() -> BTreeMap<String, &'static EndpointMeta> {
    inventory::iter::<EndpointMeta>
        .into_iter()
        .map(|endpoint| (endpoint.name.to_string(), endpoint))
        .collect()
}

async fn command_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Response {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        return error_response(
            StatusCode::NOT_FOUND,
            JsonErrorEnvelope::new(COMMAND_NOT_FOUND, format!("Command '{name}' not found"))
                .with_details(json!({ "command": name })),
        );
    };

    if endpoint.kind != EndpointKind::Rpc {
        return error_response(
            StatusCode::NOT_FOUND,
            JsonErrorEnvelope::new(COMMAND_NOT_FOUND, format!("Command '{name}' not found"))
                .with_details(json!({ "command": name })),
        );
    }

    let payload = match parse_json_body(&body) {
        Ok(payload) => payload,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.into_envelope()),
    };

    let payload = match validate_params(endpoint.params, payload) {
        Ok(payload) => payload,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.into_envelope()),
    };

    let Some(handler_key) = endpoint.handler_key else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                "Registered command is missing a handler key",
            ),
        );
    };

    let Some(handler) = state.handlers.get(&handler_key).cloned() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                format!("No handler registered for command '{name}'"),
            ),
        );
    };

    let result = catch_unwind(AssertUnwindSafe(|| handler.call(payload)));
    match result {
        Ok(Ok(value)) => (StatusCode::OK, Json(JsonResultEnvelope::new(value))).into_response(),
        Ok(Err(error)) => {
            let status = status_for_error(error.code);
            error_response(status, error.into_envelope())
        }
        Err(panic) => {
            let message = if state.debug {
                panic_message(panic)
            } else {
                "An internal error occurred".to_string()
            };
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                JsonErrorEnvelope::new(INTERNAL_ERROR, message),
            )
        }
    }
}

fn parse_json_body(body: &Bytes) -> Result<Value, ZynkError> {
    if body.is_empty() {
        return Ok(Value::Object(Default::default()));
    }

    serde_json::from_slice(body)
        .map_err(|error| ZynkError::new(VALIDATION_ERROR, format!("Invalid JSON body: {error}")))
}

fn validate_params(params: &[ParamMeta], payload: Value) -> Result<Value, ZynkError> {
    let object = payload
        .as_object()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "Request body must be a JSON object"))?;

    for param in params.iter().filter(|param| param.required) {
        if !object.contains_key(param.source_name) && !object.contains_key(param.wire_name) {
            return Err(ZynkError::with_details(
                VALIDATION_ERROR,
                format!("Missing required parameter '{}'", param.source_name),
                json!({ "parameter": param.source_name }),
            ));
        }
    }

    Ok(Value::Object(object.clone()))
}

fn status_for_error(code: &'static str) -> StatusCode {
    match code {
        VALIDATION_ERROR => StatusCode::BAD_REQUEST,
        COMMAND_NOT_FOUND => StatusCode::NOT_FOUND,
        EXECUTION_ERROR | INTERNAL_ERROR => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn error_response(status: StatusCode, error: JsonErrorEnvelope) -> Response {
    (status, Json(error)).into_response()
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "handler panicked".to_string()
    }
}

fn endpoint_to_schema(meta: &EndpointMeta) -> zynk_runtime::zynk_schema::Endpoint {
    let mut endpoint = zynk_runtime::zynk_schema::Endpoint::new(
        meta.name,
        meta.kind,
        meta.returns.to_schema_type_ref(),
    );
    endpoint.module = meta.module.map(str::to_string);
    endpoint.doc = meta.doc.map(str::to_string);
    endpoint.params = params_to_schema(meta.params);
    endpoint.channel_item = meta
        .channel_item
        .as_ref()
        .map(zynk_runtime::TypeRefStatic::to_schema_type_ref);
    endpoint.file_param = meta.file_param.map(str::to_string);
    endpoint.multi_file = meta.multi_file;
    endpoint.max_size = meta.max_size;
    endpoint.allowed_types = meta
        .allowed_types
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    endpoint.server_events = params_to_schema(meta.server_events);
    endpoint.client_events = params_to_schema(meta.client_events);
    endpoint
}

fn params_to_schema(params: &[ParamMeta]) -> Vec<zynk_runtime::zynk_schema::Param> {
    params
        .iter()
        .map(|param| zynk_runtime::zynk_schema::Param {
            source_name: param.source_name.to_string(),
            wire_name: param.wire_name.to_string(),
            ty: param.ty.to_schema_type_ref(),
            required: param.required,
            default: param
                .default
                .as_ref()
                .map(zynk_runtime::StaticValue::to_json),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_body_defaults_to_json_object() {
        assert_eq!(parse_json_body(&Bytes::new()).expect("valid"), json!({}));
    }

    #[test]
    fn validates_required_params_against_snake_or_camel_keys() {
        let params = [ParamMeta {
            source_name: "user_id",
            wire_name: "userId",
            ty: zynk_runtime::TypeRefStatic::primitive("number"),
            required: true,
            default: None,
        }];

        validate_params(&params, json!({"user_id": 7})).expect("snake case accepted");
        validate_params(&params, json!({"userId": 7})).expect("camel case accepted");
        let error = validate_params(&params, json!({})).expect_err("missing param rejected");
        assert_eq!(error.code, VALIDATION_ERROR);
    }
}
