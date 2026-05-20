//! Axum server binding for Zynk Rust endpoints.
//!
//! `ZynkBridge` consumes endpoint metadata collected through `inventory` and
//! registers Zynk wire-contract routes on an existing Axum router.

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::FutureExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use zynk_runtime::{
    inventory, EndpointKind, EndpointMeta, Handler, HandlerKey, JsonErrorEnvelope,
    JsonResultEnvelope, ParamMeta, SseFrame, ZynkError, COMMAND_NOT_FOUND, EXECUTION_ERROR,
    INTERNAL_ERROR, VALIDATION_ERROR,
};

/// Axum integration point for Zynk endpoints.
#[derive(Clone)]
pub struct ZynkBridge {
    state: Arc<BridgeState>,
}

struct BridgeState {
    endpoints: BTreeMap<String, &'static EndpointMeta>,
    handlers: HashMap<HandlerKey, Arc<dyn Handler>>,
    channel_handlers: HashMap<HandlerKey, Arc<dyn ChannelHandler>>,
    debug: bool,
    keepalive_interval: Duration,
}

type BoxChannelFuture = Pin<Box<dyn Future<Output = Result<(), ZynkError>> + Send>>;

/// Type-erased async handler contract for channel endpoints.
pub trait ChannelHandler: Send + Sync + 'static {
    /// Invoke a channel handler with a JSON payload and server-side channel.
    fn call(&self, payload: Value, channel: Channel) -> BoxChannelFuture;
}

impl<F, Fut> ChannelHandler for F
where
    F: Fn(Value, Channel) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), ZynkError>> + Send + 'static,
{
    fn call(&self, payload: Value, channel: Channel) -> BoxChannelFuture {
        Box::pin(self(payload, channel))
    }
}

#[derive(Debug, Clone)]
enum ChannelEvent {
    Data(Value),
    Close,
}

/// Server-side sender passed to channel handlers.
#[derive(Clone, Debug)]
pub struct Channel {
    id: Arc<str>,
    sender: mpsc::UnboundedSender<ChannelEvent>,
    closed: Arc<AtomicBool>,
}

impl Channel {
    fn new(id: String, sender: mpsc::UnboundedSender<ChannelEvent>) -> Self {
        Self {
            id: Arc::from(id),
            sender,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Return this channel's stable identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Send a JSON value as a `message` SSE event.
    pub fn send(&self, data: Value) -> Result<(), ZynkError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ZynkError::new(
                EXECUTION_ERROR,
                format!("Cannot send on closed channel {}", self.id),
            ));
        }
        self.sender.send(ChannelEvent::Data(data)).map_err(|_| {
            ZynkError::new(
                EXECUTION_ERROR,
                format!("Cannot send on closed channel {}", self.id),
            )
        })
    }

    /// Close the channel. A close frame is emitted after queued data frames.
    pub fn close(&self) -> Result<(), ZynkError> {
        if !self.closed.swap(true, Ordering::SeqCst) {
            let _ = self.sender.send(ChannelEvent::Close);
        }
        Ok(())
    }
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

    /// Register an async channel handler for an endpoint metadata `HandlerKey`.
    pub fn register_channel<H>(mut self, key: HandlerKey, handler: H) -> Self
    where
        H: ChannelHandler,
    {
        Arc::make_mut(&mut self.state)
            .channel_handlers
            .insert(key, Arc::new(handler));
        self
    }

    /// Set the idle interval between `keepalive` SSE frames.
    pub fn keepalive_interval(mut self, interval: Duration) -> Self {
        Arc::make_mut(&mut self.state).keepalive_interval = interval;
        self
    }

    /// Register Zynk routes on an existing Axum router.
    pub fn configure(self, router: Router) -> Router {
        router
            .route(
                "/command/{name}",
                post(command_route).with_state(self.state.clone()),
            )
            .route(
                "/channel/{name}",
                post(channel_route).with_state(self.state),
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
                channel_handlers: HashMap::new(),
                debug,
                keepalive_interval: Duration::from_secs(30),
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
            channel_handlers: self.channel_handlers.clone(),
            debug: self.debug,
            keepalive_interval: self.keepalive_interval,
        }
    }
}

fn collect_inventory_endpoints() -> BTreeMap<String, &'static EndpointMeta> {
    inventory::iter::<EndpointMeta>
        .into_iter()
        .map(|endpoint| (endpoint.name.to_string(), endpoint))
        .collect()
}

async fn channel_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Response {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        return command_not_found_response(&name);
    };

    if endpoint.kind != EndpointKind::Channel {
        return command_not_found_response(&name);
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
                "Registered channel is missing a handler key",
            ),
        );
    };

    let Some(handler) = state.channel_handlers.get(&handler_key).cloned() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                format!("No handler registered for channel '{name}'"),
            ),
        );
    };

    let channel_id = next_channel_id();
    let (sender, receiver) = mpsc::unbounded_channel();
    let channel = Channel::new(channel_id, sender);
    let handler_channel = channel.clone();
    let handler_task = tokio::spawn(async move {
        match AssertUnwindSafe(handler.call(payload, handler_channel))
            .catch_unwind()
            .await
        {
            Ok(result) => result,
            Err(panic) => Err(ZynkError::new(INTERNAL_ERROR, panic_message(panic))),
        }
    });

    let stream = futures::stream::unfold(
        ChannelStreamState {
            channel,
            receiver,
            handler_task: Some(handler_task),
            keepalive_interval: state.keepalive_interval,
            emitted_terminal: false,
        },
        next_channel_chunk,
    );

    let mut response = Body::from_stream(stream).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

struct ChannelStreamState {
    channel: Channel,
    receiver: mpsc::UnboundedReceiver<ChannelEvent>,
    handler_task: Option<tokio::task::JoinHandle<Result<(), ZynkError>>>,
    keepalive_interval: Duration,
    emitted_terminal: bool,
}

async fn next_channel_chunk(
    mut state: ChannelStreamState,
) -> Option<(Result<Bytes, Infallible>, ChannelStreamState)> {
    if state.emitted_terminal {
        return None;
    }

    loop {
        if let Some(handler_task) = state.handler_task.as_mut() {
            tokio::select! {
                Some(event) = state.receiver.recv() => {
                    match event {
                        ChannelEvent::Data(data) => {
                            return Some((Ok(Bytes::from(SseFrame::new("message", data).encode())), state));
                        }
                        ChannelEvent::Close => {
                            state.emitted_terminal = true;
                            let frame = close_frame(state.channel.id());
                            return Some((Ok(Bytes::from(frame)), state));
                        }
                    }
                }
                result = handler_task => {
                    state.handler_task = None;
                    match result {
                        Ok(Ok(())) => {
                            let _ = state.channel.close();
                            continue;
                        }
                        Ok(Err(error)) => {
                            state.emitted_terminal = true;
                            let message = error.message;
                            let frame = SseFrame::new("error", json!({ "error": message })).encode();
                            return Some((Ok(Bytes::from(frame)), state));
                        }
                        Err(error) => {
                            state.emitted_terminal = true;
                            let frame = SseFrame::new("error", json!({ "error": error.to_string() })).encode();
                            return Some((Ok(Bytes::from(frame)), state));
                        }
                    }
                }
                () = tokio::time::sleep(state.keepalive_interval) => {
                    return Some((Ok(Bytes::from(SseFrame::new("keepalive", json!({})).encode())), state));
                }
            }
        } else {
            match state.receiver.recv().await {
                Some(ChannelEvent::Data(data)) => {
                    return Some((
                        Ok(Bytes::from(SseFrame::new("message", data).encode())),
                        state,
                    ));
                }
                Some(ChannelEvent::Close) => {
                    state.emitted_terminal = true;
                    let frame = close_frame(state.channel.id());
                    return Some((Ok(Bytes::from(frame)), state));
                }
                None => return None,
            }
        }
    }
}

fn close_frame(channel_id: &str) -> String {
    SseFrame::new("close", json!({ "channelId": channel_id })).encode()
}

static NEXT_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_channel_id() -> String {
    format!("channel-{}", NEXT_CHANNEL_ID.fetch_add(1, Ordering::SeqCst))
}

async fn command_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Response {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        return command_not_found_response(&name);
    };

    if endpoint.kind != EndpointKind::Rpc {
        return command_not_found_response(&name);
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

fn command_not_found_response(name: &str) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        JsonErrorEnvelope::new(COMMAND_NOT_FOUND, format!("Command '{name}' not found"))
            .with_details(json!({ "command": name })),
    )
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
