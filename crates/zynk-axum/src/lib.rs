//! Axum server binding for Zynk Rust endpoints.
//!
//! `ZynkBridge` consumes endpoint metadata collected through `inventory` and
//! registers Zynk wire-contract routes on an existing Axum router.

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::multipart::MultipartRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket as AxumWebSocket, WebSocketUpgrade};
use axum::extract::DefaultBodyLimit;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::FutureExt;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use zynk_runtime::{
    inventory, EndpointKind, EndpointMeta, Handler, HandlerKey, JsonErrorEnvelope,
    JsonResultEnvelope, ParamMeta, SseFrame, WsMessage, ZynkError, COMMAND_NOT_FOUND,
    EXECUTION_ERROR, INTERNAL_ERROR, STATIC_HANDLER_NOT_FOUND, UPLOAD_HANDLER_NOT_FOUND,
    UPLOAD_VALIDATION_ERROR, VALIDATION_ERROR, WEBSOCKET_ERROR,
};

/// Axum integration point for Zynk endpoints.
#[derive(Clone)]
pub struct ZynkBridge {
    state: Arc<BridgeState>,
}

struct BridgeState {
    title: String,
    endpoints: BTreeMap<String, &'static EndpointMeta>,
    handlers: HashMap<HandlerKey, Arc<dyn Handler>>,
    channel_handlers: HashMap<HandlerKey, Arc<dyn ChannelHandler>>,
    upload_handlers: HashMap<HandlerKey, Arc<dyn UploadHandler>>,
    static_handlers: HashMap<HandlerKey, Arc<dyn StaticHandler>>,
    ws_handlers: HashMap<HandlerKey, Arc<dyn WsHandler>>,
    models: BTreeMap<String, zynk_runtime::zynk_schema::ModelDef>,
    enums: BTreeMap<String, zynk_runtime::zynk_schema::EnumDef>,
    debug: bool,
    keepalive_interval: Duration,
}

type BoxChannelFuture = Pin<Box<dyn Future<Output = Result<(), ZynkError>> + Send>>;
type BoxUploadFuture = Pin<Box<dyn Future<Output = Result<Value, ZynkError>> + Send>>;
type BoxStaticFuture = Pin<Box<dyn Future<Output = Result<StaticFile, ZynkError>> + Send>>;
type BoxWsFuture = Pin<Box<dyn Future<Output = Result<(), ZynkError>> + Send>>;

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

/// Uploaded file metadata and bytes passed to upload handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadFile {
    filename: String,
    content_type: String,
    bytes: Bytes,
}

impl UploadFile {
    fn new(filename: String, content_type: String, bytes: Bytes) -> Self {
        Self {
            filename,
            content_type,
            bytes,
        }
    }

    /// Original upload filename from the multipart part.
    pub fn filename(&self) -> &str {
        &self.filename
    }

    /// MIME type from the multipart part, defaulting to `application/octet-stream`.
    pub fn content_type(&self) -> &str {
        &self.content_type
    }

    /// Uploaded file size in bytes.
    pub fn size(&self) -> usize {
        self.bytes.len()
    }

    /// Uploaded file contents.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// File response descriptor returned by static handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticFile {
    path: PathBuf,
    content_type: Option<String>,
    headers: HeaderMap,
}

impl StaticFile {
    /// Create a static-file response for a filesystem path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            content_type: None,
            headers: HeaderMap::new(),
        }
    }

    /// Set an explicit response content type.
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    /// Add a response header. Invalid names or values are ignored.
    pub fn with_header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_ref().as_bytes()),
            HeaderValue::from_str(value.as_ref()),
        ) {
            self.headers.insert(name, value);
        }
        self
    }

    /// File path to serve.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Response content type, guessed from the file extension when absent.
    pub fn content_type(&self) -> String {
        self.content_type
            .clone()
            .unwrap_or_else(|| guess_content_type(&self.path))
    }

    /// Custom response headers.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }
}

/// Type-erased async handler contract for upload endpoints.
pub trait UploadHandler: Send + Sync + 'static {
    /// Invoke an upload handler with JSON args and uploaded files.
    fn call(&self, payload: Value, files: Vec<UploadFile>) -> BoxUploadFuture;
}

impl<F, Fut> UploadHandler for F
where
    F: Fn(Value, Vec<UploadFile>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, ZynkError>> + Send + 'static,
{
    fn call(&self, payload: Value, files: Vec<UploadFile>) -> BoxUploadFuture {
        Box::pin(self(payload, files))
    }
}

/// Type-erased async handler contract for static file endpoints.
pub trait StaticHandler: Send + Sync + 'static {
    /// Invoke a static handler with coerced query args.
    fn call(&self, payload: Value) -> BoxStaticFuture;
}

impl<F, Fut> StaticHandler for F
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<StaticFile, ZynkError>> + Send + 'static,
{
    fn call(&self, payload: Value) -> BoxStaticFuture {
        Box::pin(self(payload))
    }
}

/// Type-erased async handler contract for WebSocket message events.
pub trait WsHandler: Send + Sync + 'static {
    /// Invoke a websocket event handler with the parsed JSON frame and sender.
    fn call(&self, payload: Value, socket: WebSocket) -> BoxWsFuture;
}

impl<F, Fut> WsHandler for F
where
    F: Fn(Value, WebSocket) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), ZynkError>> + Send + 'static,
{
    fn call(&self, payload: Value, socket: WebSocket) -> BoxWsFuture {
        Box::pin(self(payload, socket))
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

/// Server-side sender passed to WebSocket event handlers.
#[derive(Clone, Debug)]
pub struct WebSocket {
    sender: mpsc::UnboundedSender<WsMessage>,
}

impl WebSocket {
    fn new(sender: mpsc::UnboundedSender<WsMessage>) -> Self {
        Self { sender }
    }

    /// Send a JSON value as a WebSocket `{event, data}` text frame.
    pub async fn send(&self, event: impl Into<String>, data: Value) -> Result<(), ZynkError> {
        self.sender
            .send(WsMessage::new(event, data))
            .map_err(|_| ZynkError::new(WEBSOCKET_ERROR, "Cannot send on closed WebSocket"))
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

    /// Return a copy of this bridge with the configured introspection title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.state).title = title.into();
        self
    }

    /// Return a copy of this bridge with debug error output enabled or disabled.
    pub fn debug(mut self, debug: bool) -> Self {
        Arc::make_mut(&mut self.state).debug = debug;
        self
    }

    /// Register or replace endpoint metadata used by routes and schema dumps.
    pub fn register_endpoint_meta(mut self, endpoint: &'static EndpointMeta) -> Self {
        Arc::make_mut(&mut self.state)
            .endpoints
            .insert(endpoint.name.to_string(), endpoint);
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

    /// Register an async upload handler for an endpoint metadata `HandlerKey`.
    pub fn register_upload<H>(mut self, key: HandlerKey, handler: H) -> Self
    where
        H: UploadHandler,
    {
        Arc::make_mut(&mut self.state)
            .upload_handlers
            .insert(key, Arc::new(handler));
        self
    }

    /// Register an async static-file handler for an endpoint metadata `HandlerKey`.
    pub fn register_static<H>(mut self, key: HandlerKey, handler: H) -> Self
    where
        H: StaticHandler,
    {
        Arc::make_mut(&mut self.state)
            .static_handlers
            .insert(key, Arc::new(handler));
        self
    }

    /// Register an async WebSocket event handler for an endpoint metadata `HandlerKey`.
    pub fn register_ws<H>(mut self, key: HandlerKey, handler: H) -> Self
    where
        H: WsHandler,
    {
        Arc::make_mut(&mut self.state)
            .ws_handlers
            .insert(key, Arc::new(handler));
        self
    }

    /// Register a schema model definition for `dump_schema_json()` output.
    pub fn register_model(mut self, model: zynk_runtime::zynk_schema::ModelDef) -> Self {
        Arc::make_mut(&mut self.state)
            .models
            .insert(model.name.clone(), model);
        self
    }

    /// Register a schema enum definition for `dump_schema_json()` output.
    pub fn register_enum(mut self, enum_def: zynk_runtime::zynk_schema::EnumDef) -> Self {
        Arc::make_mut(&mut self.state)
            .enums
            .insert(enum_def.name.clone(), enum_def);
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
            .route("/", get(root_route).with_state(self.state.clone()))
            .route(
                "/commands",
                get(commands_route).with_state(self.state.clone()),
            )
            .route(
                "/command/{name}",
                post(command_route).with_state(self.state.clone()),
            )
            .route(
                "/channel/{name}",
                post(channel_route).with_state(self.state.clone()),
            )
            .route(
                "/upload/{name}",
                post(upload_route)
                    .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
                    .with_state(self.state.clone()),
            )
            .route(
                "/static/{name}",
                get(static_get_route)
                    .head(static_head_route)
                    .with_state(self.state.clone()),
            )
            .route("/ws/{name}", get(ws_route).with_state(self.state))
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
        for model in self.state.models.values() {
            graph.insert_model(model.clone());
        }
        for enum_def in self.state.enums.values() {
            graph.insert_enum(enum_def.clone());
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
                title: "Zynk API".to_string(),
                endpoints,
                handlers,
                channel_handlers: HashMap::new(),
                upload_handlers: HashMap::new(),
                static_handlers: HashMap::new(),
                ws_handlers: HashMap::new(),
                models: BTreeMap::new(),
                enums: BTreeMap::new(),
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
            title: self.title.clone(),
            endpoints: self.endpoints.clone(),
            handlers: self.handlers.clone(),
            channel_handlers: self.channel_handlers.clone(),
            upload_handlers: self.upload_handlers.clone(),
            static_handlers: self.static_handlers.clone(),
            ws_handlers: self.ws_handlers.clone(),
            models: self.models.clone(),
            enums: self.enums.clone(),
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

async fn root_route(State(state): State<Arc<BridgeState>>) -> Response {
    let commands: Vec<_> = state
        .endpoints
        .values()
        .filter(|endpoint| matches!(endpoint.kind, EndpointKind::Rpc | EndpointKind::Channel))
        .map(|endpoint| endpoint.name)
        .collect();

    Json(json!({
        "status": "ok",
        "bridge": state.title,
        "commands": commands,
    }))
    .into_response()
}

async fn commands_route(State(state): State<Arc<BridgeState>>) -> Response {
    let commands: Vec<_> = state
        .endpoints
        .values()
        .filter(|endpoint| matches!(endpoint.kind, EndpointKind::Rpc | EndpointKind::Channel))
        .map(|endpoint| {
            json!({
                "name": endpoint.name,
                "module": endpoint.module.unwrap_or_default(),
                "has_channel": endpoint.kind == EndpointKind::Channel,
                "params": endpoint
                    .params
                    .iter()
                    .map(|param| param.source_name)
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    Json(json!({ "commands": commands })).into_response()
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

async fn upload_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        return upload_not_found_response(&name);
    };

    if endpoint.kind != EndpointKind::Upload {
        return upload_not_found_response(&name);
    }

    let mut multipart = match multipart {
        Ok(multipart) => multipart,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                JsonErrorEnvelope::new(
                    VALIDATION_ERROR,
                    format!("Invalid multipart body: {error}"),
                ),
            )
        }
    };

    let (args, files) = match parse_upload_multipart(endpoint, &mut multipart).await {
        Ok(parsed) => parsed,
        Err(error) => return error_response(status_for_error(error.code), error.into_envelope()),
    };

    let payload = match validate_params(endpoint.params, args) {
        Ok(payload) => payload,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.into_envelope()),
    };

    if !endpoint.multi_file && files.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            JsonErrorEnvelope::new(VALIDATION_ERROR, "No file provided"),
        );
    }

    let Some(handler_key) = endpoint.handler_key else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(INTERNAL_ERROR, "Registered upload is missing a handler key"),
        );
    };

    let Some(handler) = state.upload_handlers.get(&handler_key).cloned() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                format!("No handler registered for upload '{name}'"),
            ),
        );
    };

    match AssertUnwindSafe(handler.call(payload, files))
        .catch_unwind()
        .await
    {
        Ok(Ok(value)) => (StatusCode::OK, Json(JsonResultEnvelope::new(value))).into_response(),
        Ok(Err(error)) => error_response(status_for_error(error.code), error.into_envelope()),
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

async fn parse_upload_multipart(
    endpoint: &EndpointMeta,
    multipart: &mut Multipart,
) -> Result<(Value, Vec<UploadFile>), ZynkError> {
    let mut args = json!({});
    let mut files = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ZynkError::new(VALIDATION_ERROR, format!("Invalid multipart body: {error}"))
    })? {
        let Some(field_name) = field.name().map(str::to_string) else {
            continue;
        };

        if field_name == "_args" {
            let text = field.text().await.map_err(|error| {
                ZynkError::new(VALIDATION_ERROR, format!("Invalid _args field: {error}"))
            })?;
            args = serde_json::from_str(&text).map_err(|error| {
                ZynkError::new(VALIDATION_ERROR, format!("Invalid _args JSON: {error}"))
            })?;
        } else if field_name == "files" {
            let filename = field.file_name().unwrap_or("upload").to_string();
            let content_type = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_string();
            if !content_type_allowed(&content_type, endpoint.allowed_types) {
                return Err(upload_validation_error(
                    format!("File '{filename}' has disallowed content type {content_type}"),
                    filename,
                ));
            }
            let bytes = read_limited_field(field, endpoint.max_size, &filename).await?;
            files.push(UploadFile::new(filename, content_type, bytes));
        }
    }

    Ok((args, files))
}

async fn read_limited_field(
    mut field: axum::extract::multipart::Field<'_>,
    max_size: Option<u64>,
    filename: &str,
) -> Result<Bytes, ZynkError> {
    let mut bytes = Vec::new();
    let mut size: u64 = 0;

    while let Some(chunk) = field.chunk().await.map_err(|error| {
        ZynkError::new(VALIDATION_ERROR, format!("Invalid upload file: {error}"))
    })? {
        size += chunk.len() as u64;
        if let Some(max_size) = max_size {
            if size > max_size {
                return Err(upload_validation_error(
                    format!("File '{filename}' exceeds maximum size of {max_size} bytes"),
                    filename.to_string(),
                ));
            }
        }
        bytes.extend_from_slice(&chunk);
    }

    Ok(Bytes::from(bytes))
}

fn upload_validation_error(message: String, filename: String) -> ZynkError {
    ZynkError::with_details(
        UPLOAD_VALIDATION_ERROR,
        message,
        json!({ "filename": filename }),
    )
}

fn content_type_allowed(content_type: &str, allowed_types: &[&str]) -> bool {
    allowed_types.is_empty()
        || allowed_types.iter().any(|allowed| {
            *allowed == content_type
                || allowed
                    .strip_suffix("/*")
                    .is_some_and(|prefix| content_type.starts_with(&format!("{prefix}/")))
        })
}

async fn static_get_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    static_route(state, name, query, Method::GET).await
}

async fn static_head_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    static_route(state, name, query, Method::HEAD).await
}

async fn static_route(
    state: Arc<BridgeState>,
    name: String,
    query: HashMap<String, String>,
    method: Method,
) -> Response {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        return static_not_found_response(&name);
    };

    if endpoint.kind != EndpointKind::Static {
        return static_not_found_response(&name);
    }

    let payload = match coerce_query_params(endpoint.params, &query) {
        Ok(payload) => payload,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.into_envelope()),
    };

    let Some(handler_key) = endpoint.handler_key else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                "Registered static handler is missing a handler key",
            ),
        );
    };

    let Some(handler) = state.static_handlers.get(&handler_key).cloned() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            JsonErrorEnvelope::new(
                INTERNAL_ERROR,
                format!("No handler registered for static '{name}'"),
            ),
        );
    };

    match AssertUnwindSafe(handler.call(payload)).catch_unwind().await {
        Ok(Ok(file)) => static_file_response(file, method).await,
        Ok(Err(error)) => error_response(status_for_error(error.code), error.into_envelope()),
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

fn coerce_query_params(
    params: &[ParamMeta],
    query: &HashMap<String, String>,
) -> Result<Value, ZynkError> {
    let mut object = Map::new();
    for param in params {
        let Some(raw) = query
            .get(param.source_name)
            .or_else(|| query.get(param.wire_name))
        else {
            if param.required {
                return Err(ZynkError::with_details(
                    VALIDATION_ERROR,
                    format!("Missing required parameter: {}", param.source_name),
                    json!({ "parameter": param.source_name }),
                ));
            }
            continue;
        };
        object.insert(
            param.source_name.to_string(),
            coerce_query_value(raw, param)?,
        );
    }
    Ok(Value::Object(object))
}

fn coerce_query_value(raw: &str, param: &ParamMeta) -> Result<Value, ZynkError> {
    match param.ty.name {
        Some("number") => {
            if let Ok(value) = raw.parse::<i64>() {
                Ok(json!(value))
            } else if let Ok(value) = raw.parse::<f64>() {
                Ok(json!(value))
            } else {
                Err(ZynkError::new(
                    VALIDATION_ERROR,
                    format!("Invalid value for parameter '{}': {raw}", param.source_name),
                ))
            }
        }
        Some("boolean") => Ok(Value::Bool(matches!(
            raw.to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        ))),
        _ => Ok(Value::String(raw.to_string())),
    }
}

async fn static_file_response(file: StaticFile, method: Method) -> Response {
    let metadata = match tokio::fs::metadata(file.path()).await {
        Ok(metadata) => metadata,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                JsonErrorEnvelope::new(VALIDATION_ERROR, error.to_string()),
            )
        }
    };
    let body = if method == Method::HEAD {
        Body::empty()
    } else {
        match tokio::fs::read(file.path()).await {
            Ok(bytes) => Body::from(bytes),
            Err(error) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    JsonErrorEnvelope::new(VALIDATION_ERROR, error.to_string()),
                )
            }
        }
    };

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&file.content_type())
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    if let Ok(value) = HeaderValue::from_str(&metadata.len().to_string()) {
        headers.insert(header::CONTENT_LENGTH, value);
    }
    if let Ok(modified) = metadata.modified() {
        headers.insert(header::LAST_MODIFIED, http_date(modified));
    }
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    for (name, value) in file.headers() {
        headers.insert(name.clone(), value.clone());
    }
    response
}

fn http_date(time: SystemTime) -> HeaderValue {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    let weekday = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
    let month_name = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][(month - 1) as usize];
    HeaderValue::from_str(&format!(
        "{weekday}, {day:02} {month_name} {year:04} {hour:02}:{minute:02}:{second:02} GMT"
    ))
    .unwrap_or_else(|_| HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT"))
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (y + i64::from(m <= 2), m as u32, d as u32)
}

fn guess_content_type(path: &std::path::Path) -> String {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("html") | Some("htm") => "text/html; charset=utf-8",
        Some("txt") => "text/plain; charset=utf-8",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript",
        Some("pdf") => "application/pdf",
        _ => "application/octet-stream",
    }
    .to_string()
}

static NEXT_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_channel_id() -> String {
    format!("channel-{}", NEXT_CHANNEL_ID.fetch_add(1, Ordering::SeqCst))
}

async fn ws_route(
    State(state): State<Arc<BridgeState>>,
    Path(name): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws_socket(state, name, socket))
}

async fn handle_ws_socket(state: Arc<BridgeState>, name: String, mut socket: AxumWebSocket) {
    let Some(endpoint) = state.endpoints.get(&name).copied() else {
        close_ws(&mut socket, 4004, format!("Handler '{name}' not found")).await;
        return;
    };

    if endpoint.kind != EndpointKind::Ws {
        close_ws(&mut socket, 4004, format!("Handler '{name}' not found")).await;
        return;
    }

    let Some(handler_key) = endpoint.handler_key else {
        close_ws(
            &mut socket,
            1011,
            "Registered WebSocket is missing a handler key",
        )
        .await;
        return;
    };

    let Some(handler) = state.ws_handlers.get(&handler_key).cloned() else {
        close_ws(
            &mut socket,
            1011,
            format!("No handler registered for WebSocket '{name}'"),
        )
        .await;
        return;
    };

    let (sender, mut receiver) = mpsc::unbounded_channel();
    let ws_sender = WebSocket::new(sender);

    loop {
        tokio::select! {
            outbound = receiver.recv() => {
                match outbound {
                    Some(message) => {
                        let text = serde_json::to_string(&message)
                            .expect("WebSocket message serialization cannot fail");
                        if socket.send(Message::Text(text.into())).await.is_err() {
                            return;
                        }
                    }
                    None => return,
                }
            }
            inbound = socket.recv() => {
                let Some(inbound) = inbound else {
                    return;
                };
                match inbound {
                    Ok(Message::Text(text)) => {
                        let message = match WsMessage::from_json(text.as_str()) {
                            Ok(message) => message,
                            Err(_) => continue,
                        };
                        if !client_event_known(endpoint, &message.event) {
                            continue;
                        }
                        let payload = json!({ "event": message.event, "data": message.data });
                        let future = match catch_unwind(AssertUnwindSafe(|| {
                            handler.call(payload, ws_sender.clone())
                        })) {
                            Ok(future) => future,
                            Err(panic) => {
                                close_ws(&mut socket, 1011, panic_message(panic)).await;
                                return;
                            }
                        };
                        match AssertUnwindSafe(future).catch_unwind().await {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => {
                                close_ws(&mut socket, 1011, error.message).await;
                                return;
                            }
                            Err(panic) => {
                                close_ws(&mut socket, 1011, panic_message(panic)).await;
                                return;
                            }
                        }
                    }
                    Ok(Message::Close(_)) => return,
                    Ok(Message::Ping(payload)) => {
                        let _ = socket.send(Message::Pong(payload)).await;
                    }
                    Ok(Message::Pong(_)) | Ok(Message::Binary(_)) => {}
                    Err(error) => {
                        close_ws(&mut socket, 1011, error.to_string()).await;
                        return;
                    }
                }
            }
        }
    }
}

fn client_event_known(endpoint: &EndpointMeta, event: &str) -> bool {
    endpoint.client_events.is_empty()
        || endpoint
            .client_events
            .iter()
            .any(|param| param.source_name == event)
}

async fn close_ws(socket: &mut AxumWebSocket, code: u16, reason: impl Into<String>) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into().into(),
        })))
        .await;
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
            if error.code == INTERNAL_ERROR && !state.debug {
                error_response(
                    status,
                    JsonErrorEnvelope::new(INTERNAL_ERROR, "An internal error occurred"),
                )
            } else {
                error_response(status, error.into_envelope())
            }
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
        VALIDATION_ERROR | UPLOAD_VALIDATION_ERROR => StatusCode::BAD_REQUEST,
        COMMAND_NOT_FOUND | UPLOAD_HANDLER_NOT_FOUND | STATIC_HANDLER_NOT_FOUND => {
            StatusCode::NOT_FOUND
        }
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

fn upload_not_found_response(name: &str) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        JsonErrorEnvelope::new(
            UPLOAD_HANDLER_NOT_FOUND,
            format!("Upload handler '{name}' not found"),
        )
        .with_details(json!({ "handler": name })),
    )
}

fn static_not_found_response(name: &str) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        JsonErrorEnvelope::new(
            STATIC_HANDLER_NOT_FOUND,
            format!("Static handler '{name}' not found"),
        )
        .with_details(json!({ "handler": name })),
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

    #[test]
    fn last_modified_header_uses_http_date_format() {
        assert_eq!(
            http_date(UNIX_EPOCH),
            HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT")
        );
    }
}
