use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use axum::Router;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zynk_axum::{Channel, StaticFile, UploadFile, WebSocket, ZynkBridge};
use zynk_runtime::{
    EndpointKind, EndpointMeta, HandlerKey, ParamMeta, TypeRefStatic, ZynkError, EXECUTION_ERROR,
    INTERNAL_ERROR, VALIDATION_ERROR,
};
use zynk_schema::{ApiGraph, EnumDef, Field, ModelDef, TypeRef};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub email: String,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskLabel {
    pub id: i64,
    pub name: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Urgent,
}

impl TaskPriority {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }

    fn rank(&self) -> usize {
        match self {
            Self::Urgent => 0,
            Self::High => 1,
            Self::Medium => 2,
            Self::Low => 3,
        }
    }
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Medium
    }
}

impl std::str::FromStr for TaskPriority {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "urgent" => Ok(Self::Urgent),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Todo,
    InProgress,
    Done,
    Cancelled,
}

impl TaskStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Todo => "todo",
            Self::InProgress => "in_progress",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
        }
    }
}

impl Default for TaskStatus {
    fn default() -> Self {
        Self::Todo
    }
}

impl std::str::FromStr for TaskStatus {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "todo" => Ok(Self::Todo),
            "in_progress" => Ok(Self::InProgress),
            "done" => Ok(Self::Done),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub labels: Vec<TaskLabel>,
    pub created_at: String,
    pub due_date: Option<String>,
    pub assigned_to: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStats {
    pub total: i64,
    pub todo: i64,
    pub in_progress: i64,
    pub done: i64,
    pub cancelled: i64,
    pub by_priority: HashMap<String, i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskWireCheck {
    pub kind: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub numeric_status: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub city: String,
    pub temperature: f64,
    pub humidity: f64,
    pub conditions: String,
    pub wind_speed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherForecast {
    pub day: String,
    pub high: f64,
    pub low: f64,
    pub conditions: String,
    pub precipitation_chance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherUpdate {
    pub timestamp: String,
    pub city: String,
    pub temperature: f64,
    pub conditions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: String,
    pub filename: String,
    pub size: usize,
    pub content_type: String,
    pub checksum: String,
    pub uploaded_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUploadResult {
    pub id: String,
    pub filename: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub thumbnail_base64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentUploadResult {
    pub id: String,
    pub filename: String,
    pub size: usize,
    pub page_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub user: String,
    pub text: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingIndicator {
    pub user: String,
    pub is_typing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserJoined {
    pub user: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLeft {
    pub user: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub connected_users: usize,
    pub uptime_seconds: f64,
}

struct AppState {
    users: HashMap<i64, User>,
    next_user_id: i64,
    labels: HashMap<i64, TaskLabel>,
    next_label_id: i64,
    tasks: HashMap<i64, Task>,
    next_task_id: i64,
    uploaded_files: HashMap<String, Vec<u8>>,
    connected_users: std::collections::BTreeSet<String>,
    start_time: Instant,
}

impl AppState {
    fn new() -> Self {
        let users = HashMap::from([
            (
                1,
                User {
                    id: 1,
                    name: "Alice".to_string(),
                    email: "alice@example.com".to_string(),
                    is_active: true,
                },
            ),
            (
                2,
                User {
                    id: 2,
                    name: "Bob".to_string(),
                    email: "bob@example.com".to_string(),
                    is_active: true,
                },
            ),
            (
                3,
                User {
                    id: 3,
                    name: "Charlie".to_string(),
                    email: "charlie@example.com".to_string(),
                    is_active: true,
                },
            ),
        ]);

        let labels = HashMap::from([
            (
                1,
                TaskLabel {
                    id: 1,
                    name: "Bug".to_string(),
                    color: "#ff0000".to_string(),
                },
            ),
            (
                2,
                TaskLabel {
                    id: 2,
                    name: "Feature".to_string(),
                    color: "#00ff00".to_string(),
                },
            ),
            (
                3,
                TaskLabel {
                    id: 3,
                    name: "Documentation".to_string(),
                    color: "#0000ff".to_string(),
                },
            ),
        ]);

        let sample_tasks = vec![
            Task {
                id: 1,
                title: "Set up project structure".to_string(),
                description: Some(
                    "Create the initial folder structure and configuration files".to_string(),
                ),
                priority: TaskPriority::High,
                status: TaskStatus::Done,
                labels: vec![labels.get(&2).expect("sample label").clone()],
                created_at: "2024-01-01T10:00:00".to_string(),
                due_date: None,
                assigned_to: None,
            },
            Task {
                id: 2,
                title: "Implement user authentication".to_string(),
                description: Some("Add login and signup functionality".to_string()),
                priority: TaskPriority::High,
                status: TaskStatus::InProgress,
                labels: vec![labels.get(&2).expect("sample label").clone()],
                created_at: "2024-01-02T09:00:00".to_string(),
                due_date: Some("2024-01-15".to_string()),
                assigned_to: None,
            },
            Task {
                id: 3,
                title: "Fix login button alignment".to_string(),
                description: None,
                priority: TaskPriority::Low,
                status: TaskStatus::Todo,
                labels: vec![labels.get(&1).expect("sample label").clone()],
                created_at: "2024-01-03T14:00:00".to_string(),
                due_date: None,
                assigned_to: None,
            },
            Task {
                id: 4,
                title: "Write API documentation".to_string(),
                description: Some("Document all endpoints and models".to_string()),
                priority: TaskPriority::Medium,
                status: TaskStatus::Todo,
                labels: vec![labels.get(&3).expect("sample label").clone()],
                created_at: "2024-01-04T11:00:00".to_string(),
                due_date: None,
                assigned_to: None,
            },
        ];
        let tasks = sample_tasks
            .into_iter()
            .map(|task| (task.id, task))
            .collect();

        Self {
            users,
            next_user_id: 4,
            labels,
            next_label_id: 4,
            tasks,
            next_task_id: 5,
            uploaded_files: HashMap::new(),
            connected_users: std::collections::BTreeSet::new(),
            start_time: Instant::now(),
        }
    }
}

static STATE: OnceLock<Arc<Mutex<AppState>>> = OnceLock::new();

fn state() -> Arc<Mutex<AppState>> {
    STATE
        .get_or_init(|| Arc::new(Mutex::new(AppState::new())))
        .clone()
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn state_error() -> ZynkError {
    ZynkError::new(EXECUTION_ERROR, "application state lock was poisoned")
}

fn typed_payload<T: for<'de> Deserialize<'de>>(payload: Value) -> Result<T, ZynkError> {
    serde_json::from_value(payload).map_err(|error| {
        ZynkError::new(
            VALIDATION_ERROR,
            format!("Invalid kitchen-sink payload: {error}"),
        )
    })
}

fn to_value<T: Serialize>(value: T) -> Result<Value, ZynkError> {
    serde_json::to_value(value)
        .map_err(|error| ZynkError::new(EXECUTION_ERROR, format!("Serialization failed: {error}")))
}

fn field(source: &str, wire: &str, ty: TypeRef, required: bool) -> Field {
    Field::new(source, wire, ty, required)
}

fn optional(mut ty: TypeRef) -> TypeRef {
    ty.optional = true;
    ty
}

fn nullable(mut ty: TypeRef) -> TypeRef {
    ty.nullable = true;
    ty
}

fn optional_nullable(ty: TypeRef) -> TypeRef {
    optional(nullable(ty))
}

fn maybe<'a>(
    object: &'a serde_json::Map<String, Value>,
    snake: &str,
    camel: &str,
) -> Option<&'a Value> {
    object.get(snake).or_else(|| object.get(camel))
}

fn get_i64(payload: &Value, snake: &str, camel: &str) -> Result<i64, ZynkError> {
    let object = payload
        .as_object()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "Request body must be a JSON object"))?;
    maybe(object, snake, camel)
        .and_then(Value::as_i64)
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, format!("Missing or invalid {snake}")))
}

fn get_string(payload: &Value, snake: &str, camel: &str) -> Result<String, ZynkError> {
    let object = payload
        .as_object()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "Request body must be a JSON object"))?;
    maybe(object, snake, camel)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, format!("Missing or invalid {snake}")))
}

fn optional_string(payload: &Value, snake: &str, camel: &str) -> Option<String> {
    payload
        .as_object()
        .and_then(|object| maybe(object, snake, camel))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn optional_bool(payload: &Value, snake: &str, camel: &str) -> Option<bool> {
    payload
        .as_object()
        .and_then(|object| maybe(object, snake, camel))
        .and_then(Value::as_bool)
}

fn optional_i64(payload: &Value, snake: &str, camel: &str) -> Option<i64> {
    payload
        .as_object()
        .and_then(|object| maybe(object, snake, camel))
        .and_then(Value::as_i64)
}

fn optional_string_vec(payload: &Value, snake: &str, camel: &str) -> Option<Vec<String>> {
    payload
        .as_object()
        .and_then(|object| maybe(object, snake, camel))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
}

fn optional_i64_vec(payload: &Value, snake: &str, camel: &str) -> Option<Vec<i64>> {
    payload
        .as_object()
        .and_then(|object| maybe(object, snake, camel))
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_i64).collect())
}

fn simulated_weather(city: &str) -> WeatherData {
    let base = city_base_temps().get(city).copied().unwrap_or(15.0);
    WeatherData {
        city: city.to_string(),
        temperature: base + 1.5,
        humidity: 55.0,
        conditions: "Sunny".to_string(),
        wind_speed: 12.0,
    }
}

fn city_base_temps() -> HashMap<&'static str, f64> {
    HashMap::from([
        ("New York", 15.0),
        ("Los Angeles", 22.0),
        ("Chicago", 10.0),
        ("Miami", 28.0),
        ("Seattle", 12.0),
        ("Denver", 8.0),
        ("Tokyo", 18.0),
        ("London", 11.0),
        ("Paris", 14.0),
        ("Sydney", 23.0),
    ])
}

fn file_info(file: &UploadFile, state: &mut AppState, filename: String) -> FileInfo {
    let id = Uuid::new_v4().to_string();
    state
        .uploaded_files
        .insert(id.clone(), file.bytes().to_vec());
    let checksum = format!("{:x}", Sha256::digest(file.bytes()))[..16].to_string();
    FileInfo {
        id,
        filename,
        size: file.size(),
        content_type: file.content_type().to_string(),
        checksum,
        uploaded_at: now_iso(),
    }
}

#[zynk::command]
pub async fn get_user(user_id: i64) -> User {
    let state = state();
    let guard = state.lock().expect("application state lock");
    guard.users.get(&user_id).cloned().unwrap_or(User {
        id: user_id,
        name: format!("User {user_id}"),
        email: format!("user{user_id}@example.com"),
        is_active: true,
    })
}

#[zynk::command]
pub async fn list_users(active_only: Option<bool>) -> Vec<User> {
    let state = state();
    let guard = state.lock().expect("application state lock");
    guard
        .users
        .values()
        .filter(|user| !active_only.unwrap_or(false) || user.is_active)
        .cloned()
        .collect()
}

#[zynk::command]
pub async fn create_user(name: String, email: String) -> User {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    let user = User {
        id: guard.next_user_id,
        name,
        email,
        is_active: true,
    };
    guard.users.insert(user.id, user.clone());
    guard.next_user_id += 1;
    user
}

#[zynk::command]
pub async fn update_user(
    user_id: i64,
    name: Option<String>,
    email: Option<String>,
    is_active: Option<bool>,
) -> User {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    let mut user = guard.users.get(&user_id).cloned().unwrap_or(User {
        id: user_id,
        name: format!("User {user_id}"),
        email: format!("user{user_id}@example.com"),
        is_active: true,
    });
    if let Some(name) = name {
        user.name = name;
    }
    if let Some(email) = email {
        user.email = email;
    }
    if let Some(is_active) = is_active {
        user.is_active = is_active;
    }
    guard.users.insert(user_id, user.clone());
    user
}

#[zynk::command]
pub async fn delete_user(user_id: i64) -> bool {
    state()
        .lock()
        .expect("application state lock")
        .users
        .remove(&user_id)
        .is_some()
}

#[zynk::command]
pub async fn search_users(query: String) -> Vec<User> {
    let query = query.to_lowercase();
    let state = state();
    let guard = state.lock().expect("application state lock");
    guard
        .users
        .values()
        .filter(|user| {
            user.name.to_lowercase().contains(&query) || user.email.to_lowercase().contains(&query)
        })
        .cloned()
        .collect()
}

#[zynk::command]
pub async fn get_task(task_id: i64) -> Task {
    state()
        .lock()
        .expect("application state lock")
        .tasks
        .get(&task_id)
        .cloned()
        .unwrap_or(Task {
            id: task_id,
            title: format!("Task {task_id}"),
            description: None,
            priority: TaskPriority::Medium,
            status: TaskStatus::Todo,
            labels: vec![],
            created_at: now_iso(),
            due_date: None,
            assigned_to: None,
        })
}

#[zynk::command]
pub async fn list_tasks(
    status: Option<String>,
    priority: Option<String>,
    label_id: Option<i64>,
) -> Vec<Task> {
    let state = state();
    let guard = state.lock().expect("application state lock");
    let mut tasks: Vec<_> = guard
        .tasks
        .values()
        .filter(|task| {
            status
                .as_deref()
                .is_none_or(|value| task.status.as_str() == value)
        })
        .filter(|task| {
            priority
                .as_deref()
                .is_none_or(|value| task.priority.as_str() == value)
        })
        .filter(|task| label_id.is_none_or(|id| task.labels.iter().any(|label| label.id == id)))
        .cloned()
        .collect();
    tasks.sort_by_key(|task| (task.priority.rank(), task.created_at.clone()));
    tasks
}

#[zynk::command]
pub async fn create_task(
    title: String,
    description: Option<String>,
    priority: Option<String>,
    due_date: Option<String>,
    label_ids: Option<Vec<i64>>,
) -> Task {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    let priority = priority
        .as_deref()
        .and_then(|value| value.parse().ok())
        .unwrap_or(TaskPriority::Medium);
    let labels = label_ids
        .unwrap_or_default()
        .into_iter()
        .filter_map(|id| guard.labels.get(&id).cloned())
        .collect();
    let task = Task {
        id: guard.next_task_id,
        title,
        description,
        priority,
        status: TaskStatus::Todo,
        labels,
        created_at: now_iso(),
        due_date,
        assigned_to: None,
    };
    guard.tasks.insert(task.id, task.clone());
    guard.next_task_id += 1;
    task
}

#[zynk::command]
pub async fn update_task_status(task_id: i64, status: String) -> Task {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    let status = status.parse().unwrap_or(TaskStatus::Todo);
    let mut task = guard.tasks.get(&task_id).cloned().unwrap_or(Task {
        id: task_id,
        title: format!("Task {task_id}"),
        description: None,
        priority: TaskPriority::Medium,
        status: TaskStatus::Todo,
        labels: vec![],
        created_at: now_iso(),
        due_date: None,
        assigned_to: None,
    });
    task.status = status;
    guard.tasks.insert(task_id, task.clone());
    task
}

#[zynk::command]
pub async fn delete_task(task_id: i64) -> bool {
    state()
        .lock()
        .expect("application state lock")
        .tasks
        .remove(&task_id)
        .is_some()
}

#[zynk::command]
pub async fn echo_task_wire_check(payload: TaskWireCheck) -> TaskWireCheck {
    payload
}

#[zynk::command]
pub async fn get_task_wire_check() -> TaskWireCheck {
    TaskWireCheck {
        kind: "task_wire_check".to_string(),
        priority: TaskPriority::Urgent,
        status: TaskStatus::InProgress,
        numeric_status: 2,
    }
}

#[zynk::command]
pub async fn get_task_stats() -> TaskStats {
    let state = state();
    let guard = state.lock().expect("application state lock");
    let tasks: Vec<_> = guard.tasks.values().collect();
    let mut by_priority = HashMap::from([
        ("low".to_string(), 0),
        ("medium".to_string(), 0),
        ("high".to_string(), 0),
        ("urgent".to_string(), 0),
    ]);
    for task in &tasks {
        *by_priority
            .entry(task.priority.as_str().to_string())
            .or_default() += 1;
    }
    TaskStats {
        total: tasks.len() as i64,
        todo: tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Todo)
            .count() as i64,
        in_progress: tasks
            .iter()
            .filter(|task| task.status == TaskStatus::InProgress)
            .count() as i64,
        done: tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Done)
            .count() as i64,
        cancelled: tasks
            .iter()
            .filter(|task| task.status == TaskStatus::Cancelled)
            .count() as i64,
        by_priority,
    }
}

#[zynk::command]
pub async fn list_labels() -> Vec<TaskLabel> {
    state()
        .lock()
        .expect("application state lock")
        .labels
        .values()
        .cloned()
        .collect()
}

#[zynk::command]
pub async fn create_label(name: String, color: Option<String>) -> TaskLabel {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    let label = TaskLabel {
        id: guard.next_label_id,
        name,
        color: color.unwrap_or_else(|| "#808080".to_string()),
    };
    guard.labels.insert(label.id, label.clone());
    guard.next_label_id += 1;
    label
}

#[zynk::command]
pub async fn get_weather(city: String) -> WeatherData {
    simulated_weather(&city)
}

#[zynk::command]
pub async fn get_forecast(city: String, days: Option<i64>) -> Vec<WeatherForecast> {
    let days = days.unwrap_or(7).clamp(0, 14) as usize;
    let base = city_base_temps()
        .get(city.as_str())
        .copied()
        .unwrap_or(15.0);
    let names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    (0..days)
        .map(|index| WeatherForecast {
            day: names[index % names.len()].to_string(),
            high: base + 5.0 + index as f64,
            low: base - 3.0,
            conditions: "Sunny".to_string(),
            precipitation_chance: 10.0 + index as f64,
        })
        .collect()
}

#[zynk::command]
pub async fn list_cities() -> Vec<String> {
    city_base_temps()
        .keys()
        .map(|city| (*city).to_string())
        .collect()
}

#[zynk::upload]
pub async fn upload_file(file: UploadFile) -> FileInfo {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    file_info(&file, &mut guard, file.filename().to_string())
}

#[zynk::upload]
pub async fn upload_files(files: Vec<UploadFile>) -> Vec<FileInfo> {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    files
        .iter()
        .map(|file| file_info(file, &mut guard, file.filename().to_string()))
        .collect()
}

#[zynk::upload(max_size = "5MB", allowed_types = ["image/*"])]
pub async fn upload_image(file: UploadFile, generate_thumbnail: Option<bool>) -> ImageUploadResult {
    let id = Uuid::new_v4().to_string();
    state()
        .lock()
        .expect("application state lock")
        .uploaded_files
        .insert(id.clone(), file.bytes().to_vec());
    ImageUploadResult {
        id,
        filename: file.filename().to_string(),
        width: None,
        height: None,
        thumbnail_base64: generate_thumbnail.unwrap_or(false).then(|| {
            base64::engine::general_purpose::STANDARD.encode(&file.bytes()[..file.size().min(100)])
        }),
    }
}

#[zynk::upload(
    max_size = "10MB",
    allowed_types = [
        "application/msword",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
)]
pub async fn upload_document(
    file: UploadFile,
    extract_metadata: Option<bool>,
) -> DocumentUploadResult {
    let id = Uuid::new_v4().to_string();
    state()
        .lock()
        .expect("application state lock")
        .uploaded_files
        .insert(id.clone(), file.bytes().to_vec());
    DocumentUploadResult {
        id,
        filename: file.filename().to_string(),
        size: file.size(),
        page_count: extract_metadata
            .unwrap_or(true)
            .then_some(std::cmp::max(1, file.size() / 50_000)),
    }
}

#[zynk::upload(max_size = "50MB", allowed_types = ["audio/*", "image/*", "video/*"])]
pub async fn upload_media(files: Vec<UploadFile>, album_name: Option<String>) -> Vec<FileInfo> {
    let state = state();
    let mut guard = state.lock().expect("application state lock");
    files
        .iter()
        .map(|file| {
            let filename = album_name
                .as_ref()
                .map(|album| format!("{album}/{}", file.filename()))
                .unwrap_or_else(|| file.filename().to_string());
            file_info(file, &mut guard, filename)
        })
        .collect()
}

#[zynk::static_file]
pub async fn download_sample(filename: Option<String>) -> StaticFile {
    let filename = filename.unwrap_or_else(|| "sample.txt".to_string());
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static_assets");
    StaticFile::new(base.join(filename)).with_content_type("text/plain")
}

#[zynk::message]
pub async fn chat(_chat_message: ChatMessage) {}

async fn stream_weather_handler(payload: Value, channel: Channel) -> Result<(), ZynkError> {
    let city = get_string(&payload, "city", "city")?;
    if city == "__parity__" {
        let update = WeatherUpdate {
            timestamp: "2024-01-01T00:00:00".to_string(),
            city: "Tokyo".to_string(),
            temperature: 19.5,
            conditions: "Sunny".to_string(),
        };
        channel.send(to_value(update)?)?;
        return Ok(());
    }
    if city == "__idle_keepalive__" {
        tokio::time::sleep(std::time::Duration::from_millis(35_500)).await;
        channel.send(json!("hello"))?;
        return Ok(());
    }
    if city == "__error__" {
        channel.send(to_value(WeatherUpdate {
            timestamp: "2024-01-01T00:00:00".to_string(),
            city: "Tokyo".to_string(),
            temperature: 19.5,
            conditions: "Sunny".to_string(),
        })?)?;
        channel.send(to_value(WeatherUpdate {
            timestamp: "2024-01-01T00:00:01".to_string(),
            city: "Tokyo".to_string(),
            temperature: 20.5,
            conditions: "Cloudy".to_string(),
        })?)?;
        return Err(ZynkError::new(EXECUTION_ERROR, "boom"));
    }
    let update = WeatherUpdate {
        timestamp: now_iso(),
        city: city.clone(),
        temperature: simulated_weather(&city).temperature,
        conditions: "Sunny".to_string(),
    };
    channel.send(to_value(update)?)?;
    Ok(())
}

async fn stream_multi_city_handler(payload: Value, channel: Channel) -> Result<(), ZynkError> {
    let mut cities = optional_string_vec(&payload, "cities", "cities").unwrap_or_else(|| {
        city_base_temps()
            .keys()
            .take(3)
            .map(|city| (*city).to_string())
            .collect()
    });
    if cities.is_empty() {
        cities = city_base_temps()
            .keys()
            .take(3)
            .map(|city| (*city).to_string())
            .collect();
    }
    for city in cities.into_iter().take(3) {
        let update = WeatherUpdate {
            timestamp: now_iso(),
            temperature: simulated_weather(&city).temperature,
            conditions: "Sunny".to_string(),
            city,
        };
        channel.send(to_value(update)?)?;
    }
    Ok(())
}

async fn chat_handler(payload: Value, socket: WebSocket) -> Result<(), ZynkError> {
    let event = payload
        .get("event")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let data = payload.get("data").cloned().unwrap_or(Value::Null);
    match event {
        "join" => {
            let joined: UserJoined = typed_payload(data.clone())?;
            if joined.user == "__panic__" {
                return Err(ZynkError::new(INTERNAL_ERROR, "super secret stack info"));
            }
            let (connected_users, uptime_seconds) = {
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                guard.connected_users.insert(joined.user.clone());
                (
                    guard.connected_users.len(),
                    guard.start_time.elapsed().as_secs_f64(),
                )
            };
            socket
                .send(
                    "user_joined",
                    to_value(UserJoined {
                        user: joined.user,
                        timestamp: now_iso(),
                    })?,
                )
                .await?;
            socket
                .send(
                    "status",
                    to_value(ServerStatus {
                        connected_users,
                        uptime_seconds,
                    })?,
                )
                .await?;
        }
        "leave" => {
            let left: UserLeft = typed_payload(data.clone())?;
            let state = state();
            state
                .lock()
                .map_err(|_| state_error())?
                .connected_users
                .remove(&left.user);
            socket
                .send(
                    "user_left",
                    to_value(UserLeft {
                        user: left.user,
                        timestamp: now_iso(),
                    })?,
                )
                .await?;
        }
        "chat_message" => {
            let message: ChatMessage = typed_payload(data)?;
            if message.text == "__panic__" {
                return Err(ZynkError::new(INTERNAL_ERROR, "super secret stack info"));
            }
            socket
                .send(
                    "chat_message",
                    to_value(ChatMessage {
                        user: message.user,
                        text: message.text,
                        timestamp: now_iso(),
                    })?,
                )
                .await?;
        }
        "typing" => {
            let typing: TypingIndicator = typed_payload(data)?;
            socket.send("typing", to_value(typing)?).await?;
        }
        _ => {}
    }
    Ok(())
}

fn bridge_with_handlers() -> ZynkBridge {
    ZynkBridge::new()
        .title("Kitchen Sink API")
        .register_endpoint_meta(&CREATE_LABEL_META)
        .register_endpoint_meta(&CREATE_TASK_META)
        .register_endpoint_meta(&ECHO_TASK_WIRE_CHECK_META)
        .register_endpoint_meta(&GET_TASK_WIRE_CHECK_META)
        .register_endpoint_meta(&DOWNLOAD_SAMPLE_META)
        .register_endpoint_meta(&GET_FORECAST_META)
        .register_endpoint_meta(&LIST_USERS_META)
        .register_endpoint_meta(&UPLOAD_DOCUMENT_META)
        .register_endpoint_meta(&UPLOAD_IMAGE_META)
        .register_endpoint_meta(&STREAM_WEATHER_META)
        .register_endpoint_meta(&STREAM_MULTI_CITY_META)
        .register_endpoint_meta(&CHAT_META)
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_user"),
            |payload: Value| {
                let state = state();
                let guard = state.lock().map_err(|_| state_error())?;
                let id = get_i64(&payload, "user_id", "userId")?;
                if id == -500 {
                    return Err(ZynkError::new(INTERNAL_ERROR, "super secret stack info"));
                }
                let user = guard.users.get(&id).cloned().ok_or_else(|| {
                    ZynkError::new(EXECUTION_ERROR, format!("User with ID {id} not found"))
                })?;
                to_value(user)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::list_users"),
            |payload: Value| {
                let active_only =
                    optional_bool(&payload, "active_only", "activeOnly").unwrap_or(false);
                let state = state();
                let guard = state.lock().map_err(|_| state_error())?;
                let users: Vec<_> = guard
                    .users
                    .values()
                    .filter(|u| !active_only || u.is_active)
                    .cloned()
                    .collect();
                to_value(users)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::create_user"),
            |payload: Value| {
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                let user = User {
                    id: guard.next_user_id,
                    name: get_string(&payload, "name", "name")?,
                    email: get_string(&payload, "email", "email")?,
                    is_active: true,
                };
                guard.users.insert(user.id, user.clone());
                guard.next_user_id += 1;
                to_value(user)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::update_user"),
            |payload: Value| {
                let id = get_i64(&payload, "user_id", "userId")?;
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                let mut user = guard.users.get(&id).cloned().ok_or_else(|| {
                    ZynkError::new(EXECUTION_ERROR, format!("User with ID {id} not found"))
                })?;
                if let Some(name) = optional_string(&payload, "name", "name") {
                    user.name = name;
                }
                if let Some(email) = optional_string(&payload, "email", "email") {
                    user.email = email;
                }
                if let Some(active) = optional_bool(&payload, "is_active", "isActive") {
                    user.is_active = active;
                }
                guard.users.insert(id, user.clone());
                to_value(user)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::delete_user"),
            |payload: Value| {
                let deleted = state()
                    .lock()
                    .map_err(|_| state_error())?
                    .users
                    .remove(&get_i64(&payload, "user_id", "userId")?)
                    .is_some();
                to_value(deleted)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::search_users"),
            |payload: Value| {
                let query = get_string(&payload, "query", "query")?.to_lowercase();
                let state = state();
                let guard = state.lock().map_err(|_| state_error())?;
                let users: Vec<_> = guard
                    .users
                    .values()
                    .filter(|u| {
                        u.name.to_lowercase().contains(&query)
                            || u.email.to_lowercase().contains(&query)
                    })
                    .cloned()
                    .collect();
                to_value(users)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_task"),
            |payload: Value| {
                let id = get_i64(&payload, "task_id", "taskId")?;
                let state = state();
                let guard = state.lock().map_err(|_| state_error())?;
                let task = guard.tasks.get(&id).cloned().ok_or_else(|| {
                    ZynkError::new(EXECUTION_ERROR, format!("Task with ID {id} not found"))
                })?;
                to_value(task)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::list_tasks"),
            |payload: Value| {
                let status = optional_string(&payload, "status", "status");
                let priority = optional_string(&payload, "priority", "priority");
                let label_id = optional_i64(&payload, "label_id", "labelId");
                let state = state();
                let guard = state.lock().map_err(|_| state_error())?;
                let mut tasks: Vec<_> = guard
                    .tasks
                    .values()
                    .filter(|task| status.as_deref().is_none_or(|v| task.status.as_str() == v))
                    .filter(|task| {
                        priority
                            .as_deref()
                            .is_none_or(|v| task.priority.as_str() == v)
                    })
                    .filter(|task| {
                        label_id.is_none_or(|id| task.labels.iter().any(|label| label.id == id))
                    })
                    .cloned()
                    .collect();
                tasks.sort_by_key(|task| (task.priority.rank(), task.created_at.clone()));
                to_value(tasks)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::create_task"),
            |payload: Value| {
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                let priority = optional_string(&payload, "priority", "priority")
                    .as_deref()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(TaskPriority::Medium);
                let labels = optional_i64_vec(&payload, "label_ids", "labelIds")
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|id| guard.labels.get(&id).cloned())
                    .collect();
                let task = Task {
                    id: guard.next_task_id,
                    title: get_string(&payload, "title", "title")?,
                    description: optional_string(&payload, "description", "description"),
                    priority,
                    status: TaskStatus::Todo,
                    labels,
                    created_at: now_iso(),
                    due_date: optional_string(&payload, "due_date", "dueDate"),
                    assigned_to: None,
                };
                guard.tasks.insert(task.id, task.clone());
                guard.next_task_id += 1;
                to_value(task)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::update_task_status"),
            |payload: Value| {
                let id = get_i64(&payload, "task_id", "taskId")?;
                let status_value = get_string(&payload, "status", "status")?;
                let status = status_value.parse().map_err(|_| {
                    ZynkError::new(EXECUTION_ERROR, format!("Invalid status: {status_value}"))
                })?;
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                let mut task = guard.tasks.get(&id).cloned().ok_or_else(|| {
                    ZynkError::new(EXECUTION_ERROR, format!("Task with ID {id} not found"))
                })?;
                task.status = status;
                guard.tasks.insert(id, task.clone());
                to_value(task)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::delete_task"),
            |payload: Value| {
                let deleted = state()
                    .lock()
                    .map_err(|_| state_error())?
                    .tasks
                    .remove(&get_i64(&payload, "task_id", "taskId")?)
                    .is_some();
                to_value(deleted)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::echo_task_wire_check"),
            |payload: Value| {
                let object = payload.as_object().ok_or_else(|| {
                    ZynkError::new(VALIDATION_ERROR, "Request body must be a JSON object")
                })?;
                let payload = maybe(object, "payload", "payload")
                    .cloned()
                    .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "Missing payload"))?;
                let check: TaskWireCheck = typed_payload(payload)?;
                to_value(check)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_task_wire_check"),
            |_payload: Value| {
                to_value(TaskWireCheck {
                    kind: "task_wire_check".to_string(),
                    priority: TaskPriority::Urgent,
                    status: TaskStatus::InProgress,
                    numeric_status: 2,
                })
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_task_stats"),
            |_payload: Value| {
                to_value({
                    let state = state();
                    let guard = state.lock().map_err(|_| state_error())?;
                    let tasks: Vec<_> = guard.tasks.values().collect();
                    let mut by_priority = HashMap::from([
                        (String::from("low"), 0),
                        (String::from("medium"), 0),
                        (String::from("high"), 0),
                        (String::from("urgent"), 0),
                    ]);
                    for task in &tasks {
                        *by_priority
                            .entry(task.priority.as_str().to_string())
                            .or_default() += 1;
                    }
                    TaskStats {
                        total: tasks.len() as i64,
                        todo: tasks
                            .iter()
                            .filter(|t| t.status == TaskStatus::Todo)
                            .count() as i64,
                        in_progress: tasks
                            .iter()
                            .filter(|t| t.status == TaskStatus::InProgress)
                            .count() as i64,
                        done: tasks
                            .iter()
                            .filter(|t| t.status == TaskStatus::Done)
                            .count() as i64,
                        cancelled: tasks
                            .iter()
                            .filter(|t| t.status == TaskStatus::Cancelled)
                            .count() as i64,
                        by_priority,
                    }
                })
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::list_labels"),
            |_payload: Value| {
                to_value(
                    state()
                        .lock()
                        .map_err(|_| state_error())?
                        .labels
                        .values()
                        .cloned()
                        .collect::<Vec<_>>(),
                )
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::create_label"),
            |payload: Value| {
                let state = state();
                let mut guard = state.lock().map_err(|_| state_error())?;
                let label = TaskLabel {
                    id: guard.next_label_id,
                    name: get_string(&payload, "name", "name")?,
                    color: optional_string(&payload, "color", "color")
                        .unwrap_or_else(|| "#808080".to_string()),
                };
                guard.labels.insert(label.id, label.clone());
                guard.next_label_id += 1;
                to_value(label)
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_weather"),
            |payload: Value| to_value(simulated_weather(&get_string(&payload, "city", "city")?)),
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::get_forecast"),
            |payload: Value| {
                let city = get_string(&payload, "city", "city")?;
                let days = optional_i64(&payload, "days", "days")
                    .unwrap_or(7)
                    .clamp(0, 14) as usize;
                let base = city_base_temps()
                    .get(city.as_str())
                    .copied()
                    .unwrap_or(15.0);
                let names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
                to_value(
                    (0..days)
                        .map(|index| WeatherForecast {
                            day: names[index % names.len()].to_string(),
                            high: base + 5.0 + index as f64,
                            low: base - 3.0,
                            conditions: "Sunny".to_string(),
                            precipitation_chance: 10.0 + index as f64,
                        })
                        .collect::<Vec<_>>(),
                )
            },
        )
        .register_handler(
            HandlerKey("rust_axum_kitchen_sink::list_cities"),
            |_payload: Value| {
                to_value(
                    city_base_temps()
                        .keys()
                        .map(|city| (*city).to_string())
                        .collect::<Vec<_>>(),
                )
            },
        )
        .register_channel(
            HandlerKey("rust_axum_kitchen_sink::stream_weather"),
            stream_weather_handler,
        )
        .register_channel(
            HandlerKey("rust_axum_kitchen_sink::stream_multi_city"),
            stream_multi_city_handler,
        )
        .register_upload(
            HandlerKey("rust_axum_kitchen_sink::upload_file"),
            upload_file_handler,
        )
        .register_upload(
            HandlerKey("rust_axum_kitchen_sink::upload_files"),
            upload_files_handler,
        )
        .register_upload(
            HandlerKey("rust_axum_kitchen_sink::upload_image"),
            upload_image_handler,
        )
        .register_upload(
            HandlerKey("rust_axum_kitchen_sink::upload_document"),
            upload_document_handler,
        )
        .register_upload(
            HandlerKey("rust_axum_kitchen_sink::upload_media"),
            upload_media_handler,
        )
        .register_static(
            HandlerKey("rust_axum_kitchen_sink::download_sample"),
            download_sample_handler,
        )
        .register_ws(HandlerKey("rust_axum_kitchen_sink::chat"), chat_handler)
        .register_model_models()
}

trait RegisterKitchenSinkModels {
    fn register_model_models(self) -> Self;
}

impl RegisterKitchenSinkModels for ZynkBridge {
    fn register_model_models(self) -> Self {
        let mut bridge = self;
        let graph = kitchen_sink_schema_graph();
        for model in graph.models.into_values() {
            bridge = bridge.register_model(model);
        }
        for enum_def in graph.enums.into_values() {
            bridge = bridge.register_enum(enum_def);
        }
        bridge
    }
}

async fn upload_file_handler(_payload: Value, files: Vec<UploadFile>) -> Result<Value, ZynkError> {
    let file = files
        .first()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "upload_file requires one file"))?;
    let state = state();
    let mut guard = state.lock().map_err(|_| state_error())?;
    to_value(file_info(file, &mut guard, file.filename().to_string()))
}

async fn upload_files_handler(_payload: Value, files: Vec<UploadFile>) -> Result<Value, ZynkError> {
    let state = state();
    let mut guard = state.lock().map_err(|_| state_error())?;
    let infos: Vec<_> = files
        .iter()
        .map(|file| file_info(file, &mut guard, file.filename().to_string()))
        .collect();
    to_value(infos)
}

async fn upload_image_handler(payload: Value, files: Vec<UploadFile>) -> Result<Value, ZynkError> {
    let file = files
        .first()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "upload_image requires one file"))?;
    to_value(
        upload_image(
            file.clone(),
            optional_bool(&payload, "generate_thumbnail", "generateThumbnail"),
        )
        .await,
    )
}

async fn upload_document_handler(
    payload: Value,
    files: Vec<UploadFile>,
) -> Result<Value, ZynkError> {
    let file = files
        .first()
        .ok_or_else(|| ZynkError::new(VALIDATION_ERROR, "upload_document requires one file"))?;
    to_value(
        upload_document(
            file.clone(),
            optional_bool(&payload, "extract_metadata", "extractMetadata"),
        )
        .await,
    )
}

async fn upload_media_handler(payload: Value, files: Vec<UploadFile>) -> Result<Value, ZynkError> {
    to_value(upload_media(files, optional_string(&payload, "album_name", "albumName")).await)
}

async fn download_sample_handler(payload: Value) -> Result<StaticFile, ZynkError> {
    Ok(download_sample(optional_string(&payload, "filename", "filename")).await)
}

static CREATE_LABEL_PARAMS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "name",
        wire_name: "name",
        ty: TypeRefStatic::primitive("string"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "color",
        wire_name: "color",
        ty: TypeRefStatic::primitive("string").optional(),
        required: false,
        default: None,
    },
];

static CREATE_TASK_PARAMS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "title",
        wire_name: "title",
        ty: TypeRefStatic::primitive("string"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "description",
        wire_name: "description",
        ty: TypeRefStatic::primitive("string").optional().nullable(),
        required: false,
        default: None,
    },
    ParamMeta {
        source_name: "priority",
        wire_name: "priority",
        ty: TypeRefStatic::primitive("string").optional(),
        required: false,
        default: None,
    },
    ParamMeta {
        source_name: "due_date",
        wire_name: "dueDate",
        ty: TypeRefStatic::primitive("string").optional().nullable(),
        required: false,
        default: None,
    },
    ParamMeta {
        source_name: "label_ids",
        wire_name: "labelIds",
        ty: TypeRefStatic::array(&[TypeRefStatic::primitive("number")])
            .optional()
            .nullable(),
        required: false,
        default: None,
    },
];

static ECHO_TASK_WIRE_CHECK_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "payload",
    wire_name: "payload",
    ty: TypeRefStatic::model("TaskWireCheck"),
    required: true,
    default: None,
}];

static DOWNLOAD_SAMPLE_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "filename",
    wire_name: "filename",
    ty: TypeRefStatic::primitive("string").optional(),
    required: false,
    default: None,
}];

static GET_FORECAST_PARAMS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "city",
        wire_name: "city",
        ty: TypeRefStatic::primitive("string"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "days",
        wire_name: "days",
        ty: TypeRefStatic::primitive("number").optional(),
        required: false,
        default: None,
    },
];

static LIST_USERS_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "active_only",
    wire_name: "activeOnly",
    ty: TypeRefStatic::primitive("boolean").optional(),
    required: false,
    default: None,
}];

static UPLOAD_DOCUMENT_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "extract_metadata",
    wire_name: "extractMetadata",
    ty: TypeRefStatic::primitive("boolean").optional(),
    required: false,
    default: None,
}];

static UPLOAD_IMAGE_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "generate_thumbnail",
    wire_name: "generateThumbnail",
    ty: TypeRefStatic::primitive("boolean").optional(),
    required: false,
    default: None,
}];

static CREATE_LABEL_META: EndpointMeta = EndpointMeta {
    name: "create_label",
    kind: EndpointKind::Rpc,
    module: Some("tasks"),
    doc: None,
    params: CREATE_LABEL_PARAMS,
    returns: TypeRefStatic::model("TaskLabel"),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::create_label")),
};

static CREATE_TASK_META: EndpointMeta = EndpointMeta {
    name: "create_task",
    kind: EndpointKind::Rpc,
    module: Some("tasks"),
    doc: None,
    params: CREATE_TASK_PARAMS,
    returns: TypeRefStatic::model("Task"),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::create_task")),
};

static ECHO_TASK_WIRE_CHECK_META: EndpointMeta = EndpointMeta {
    name: "echo_task_wire_check",
    kind: EndpointKind::Rpc,
    module: Some("tasks"),
    doc: None,
    params: ECHO_TASK_WIRE_CHECK_PARAMS,
    returns: TypeRefStatic::model("TaskWireCheck"),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::echo_task_wire_check")),
};

static GET_TASK_WIRE_CHECK_META: EndpointMeta = EndpointMeta {
    name: "get_task_wire_check",
    kind: EndpointKind::Rpc,
    module: Some("tasks"),
    doc: None,
    params: &[],
    returns: TypeRefStatic::model("TaskWireCheck"),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::get_task_wire_check")),
};

static DOWNLOAD_SAMPLE_META: EndpointMeta = EndpointMeta {
    name: "download_sample",
    kind: EndpointKind::Static,
    module: Some("uploads"),
    doc: None,
    params: DOWNLOAD_SAMPLE_PARAMS,
    returns: TypeRefStatic {
        kind: zynk_schema::TypeKind::Any,
        name: Some("StaticFile"),
        inner: &[],
        optional: false,
        nullable: false,
        value: None,
    },
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::download_sample")),
};

static GET_FORECAST_META: EndpointMeta = EndpointMeta {
    name: "get_forecast",
    kind: EndpointKind::Rpc,
    module: Some("weather"),
    doc: None,
    params: GET_FORECAST_PARAMS,
    returns: TypeRefStatic::array(&[TypeRefStatic::model("WeatherForecast")]),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::get_forecast")),
};

static LIST_USERS_META: EndpointMeta = EndpointMeta {
    name: "list_users",
    kind: EndpointKind::Rpc,
    module: Some("users"),
    doc: None,
    params: LIST_USERS_PARAMS,
    returns: TypeRefStatic::array(&[TypeRefStatic::model("User")]),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::list_users")),
};

static UPLOAD_DOCUMENT_META: EndpointMeta = EndpointMeta {
    name: "upload_document",
    kind: EndpointKind::Upload,
    module: Some("uploads"),
    doc: None,
    params: UPLOAD_DOCUMENT_PARAMS,
    returns: TypeRefStatic::model("DocumentUploadResult"),
    channel_item: None,
    file_param: Some("file"),
    multi_file: false,
    max_size: Some(10_485_760),
    allowed_types: &[
        "application/msword",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::upload_document")),
};

static UPLOAD_IMAGE_META: EndpointMeta = EndpointMeta {
    name: "upload_image",
    kind: EndpointKind::Upload,
    module: Some("uploads"),
    doc: None,
    params: UPLOAD_IMAGE_PARAMS,
    returns: TypeRefStatic::model("ImageUploadResult"),
    channel_item: None,
    file_param: Some("file"),
    multi_file: false,
    max_size: Some(5_242_880),
    allowed_types: &["image/*"],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::upload_image")),
};

static STREAM_WEATHER_PARAMS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "city",
        wire_name: "city",
        ty: TypeRefStatic::primitive("string"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "interval_seconds",
        wire_name: "intervalSeconds",
        ty: TypeRefStatic::primitive("number"),
        required: true,
        default: None,
    },
];

static STREAM_MULTI_CITY_PARAMS: &[ParamMeta] = &[ParamMeta {
    source_name: "cities",
    wire_name: "cities",
    ty: TypeRefStatic::array(&[TypeRefStatic::primitive("string")]),
    required: true,
    default: None,
}];

static STREAM_WEATHER_META: EndpointMeta = EndpointMeta {
    name: "stream_weather",
    kind: EndpointKind::Channel,
    module: Some("weather"),
    doc: None,
    params: STREAM_WEATHER_PARAMS,
    returns: TypeRefStatic::primitive("undefined"),
    channel_item: Some(TypeRefStatic::model("WeatherUpdate")),
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::stream_weather")),
};

static STREAM_MULTI_CITY_META: EndpointMeta = EndpointMeta {
    name: "stream_multi_city",
    kind: EndpointKind::Channel,
    module: Some("weather"),
    doc: None,
    params: STREAM_MULTI_CITY_PARAMS,
    returns: TypeRefStatic::primitive("undefined"),
    channel_item: Some(TypeRefStatic::model("WeatherUpdate")),
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: &[],
    client_events: &[],
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::stream_multi_city")),
};

static SERVER_EVENTS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "chat_message",
        wire_name: "chatMessage",
        ty: TypeRefStatic::model("ChatMessage"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "status",
        wire_name: "status",
        ty: TypeRefStatic::model("ServerStatus"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "typing",
        wire_name: "typing",
        ty: TypeRefStatic::model("TypingIndicator"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "user_joined",
        wire_name: "userJoined",
        ty: TypeRefStatic::model("UserJoined"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "user_left",
        wire_name: "userLeft",
        ty: TypeRefStatic::model("UserLeft"),
        required: true,
        default: None,
    },
];

static CLIENT_EVENTS: &[ParamMeta] = &[
    ParamMeta {
        source_name: "chat_message",
        wire_name: "chatMessage",
        ty: TypeRefStatic::model("ChatMessage"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "join",
        wire_name: "join",
        ty: TypeRefStatic::model("UserJoined"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "leave",
        wire_name: "leave",
        ty: TypeRefStatic::model("UserLeft"),
        required: true,
        default: None,
    },
    ParamMeta {
        source_name: "typing",
        wire_name: "typing",
        ty: TypeRefStatic::model("TypingIndicator"),
        required: true,
        default: None,
    },
];

static CHAT_META: EndpointMeta = EndpointMeta {
    name: "chat",
    kind: EndpointKind::Ws,
    module: Some("chat"),
    doc: None,
    params: &[],
    returns: TypeRefStatic::void(),
    channel_item: None,
    file_param: None,
    multi_file: false,
    max_size: None,
    allowed_types: &[],
    server_events: SERVER_EVENTS,
    client_events: CLIENT_EVENTS,
    handler_key: Some(HandlerKey("rust_axum_kitchen_sink::chat")),
};

fn kitchen_sink_schema_graph() -> ApiGraph {
    let mut graph = ApiGraph::new();
    let mut priority = EnumDef::new("TaskPriority");
    priority.doc = Some("Priority levels for tasks.".to_string());
    priority.values = vec![
        json!("low"),
        json!("medium"),
        json!("high"),
        json!("urgent"),
    ];
    graph.insert_enum(priority);
    let mut status = EnumDef::new("TaskStatus");
    status.doc = Some("Status of a task.".to_string());
    status.values = vec![
        json!("todo"),
        json!("in_progress"),
        json!("done"),
        json!("cancelled"),
    ];
    graph.insert_enum(status);
    let mut numeric_status = EnumDef::new("NumericTaskStatus");
    numeric_status.doc = Some("Numeric status codes for wire-fidelity checks.".to_string());
    numeric_status.values = vec![json!(1), json!(2), json!(3)];
    graph.insert_enum(numeric_status);

    insert_model(
        &mut graph,
        "User",
        "A user in the system.",
        vec![
            field("id", "id", TypeRef::primitive("number"), true),
            field("name", "name", TypeRef::primitive("string"), true),
            field("email", "email", TypeRef::primitive("string"), true),
            {
                let mut f = field(
                    "is_active",
                    "isActive",
                    optional(TypeRef::primitive("boolean")),
                    false,
                );
                f.default = Some(json!(true));
                f
            },
        ],
    );
    insert_model(
        &mut graph,
        "TaskLabel",
        "A label/tag for categorizing tasks.",
        vec![
            field("id", "id", TypeRef::primitive("number"), true),
            field("name", "name", TypeRef::primitive("string"), true),
            {
                let mut f = field(
                    "color",
                    "color",
                    optional(TypeRef::primitive("string")),
                    false,
                );
                f.default = Some(json!("#808080"));
                f
            },
        ],
    );
    insert_model(
        &mut graph,
        "Task",
        "A task item.",
        vec![
            field("id", "id", TypeRef::primitive("number"), true),
            field("title", "title", TypeRef::primitive("string"), true),
            null_default(field(
                "description",
                "description",
                optional_nullable(TypeRef::primitive("string")),
                false,
            )),
            defaulted(
                field(
                    "priority",
                    "priority",
                    optional(TypeRef::enum_ref("TaskPriority")),
                    false,
                ),
                json!("medium"),
            ),
            defaulted(
                field(
                    "status",
                    "status",
                    optional(TypeRef::enum_ref("TaskStatus")),
                    false,
                ),
                json!("todo"),
            ),
            field(
                "labels",
                "labels",
                optional(TypeRef::array(TypeRef::model("TaskLabel"))),
                false,
            ),
            field(
                "created_at",
                "createdAt",
                TypeRef::primitive("string"),
                true,
            ),
            null_default(field(
                "due_date",
                "dueDate",
                optional_nullable(TypeRef::primitive("string")),
                false,
            )),
            null_default(field(
                "assigned_to",
                "assignedTo",
                optional_nullable(TypeRef::primitive("number")),
                false,
            )),
        ],
    );
    insert_model(
        &mut graph,
        "TaskStats",
        "Statistics about tasks.",
        vec![
            field("total", "total", TypeRef::primitive("number"), true),
            field("todo", "todo", TypeRef::primitive("number"), true),
            field(
                "in_progress",
                "inProgress",
                TypeRef::primitive("number"),
                true,
            ),
            field("done", "done", TypeRef::primitive("number"), true),
            field("cancelled", "cancelled", TypeRef::primitive("number"), true),
            field(
                "by_priority",
                "byPriority",
                TypeRef::record(TypeRef::primitive("string"), TypeRef::primitive("number")),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "TaskWireCheck",
        "A literal-bearing payload used to verify wire fidelity.",
        vec![
            field(
                "kind",
                "kind",
                TypeRef::literal(json!("task_wire_check")),
                true,
            ),
            field(
                "priority",
                "priority",
                TypeRef::enum_ref("TaskPriority"),
                true,
            ),
            field("status", "status", TypeRef::enum_ref("TaskStatus"), true),
            field(
                "numeric_status",
                "numericStatus",
                TypeRef::union(vec![
                    TypeRef::literal(json!(1)),
                    TypeRef::literal(json!(2)),
                    TypeRef::literal(json!(3)),
                ]),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "WeatherData",
        "Current weather conditions.",
        vec![
            field("city", "city", TypeRef::primitive("string"), true),
            field(
                "temperature",
                "temperature",
                TypeRef::primitive("number"),
                true,
            ),
            field("humidity", "humidity", TypeRef::primitive("number"), true),
            field(
                "conditions",
                "conditions",
                TypeRef::primitive("string"),
                true,
            ),
            field(
                "wind_speed",
                "windSpeed",
                TypeRef::primitive("number"),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "WeatherForecast",
        "Weather forecast for a specific day.",
        vec![
            field("day", "day", TypeRef::primitive("string"), true),
            field("high", "high", TypeRef::primitive("number"), true),
            field("low", "low", TypeRef::primitive("number"), true),
            field(
                "conditions",
                "conditions",
                TypeRef::primitive("string"),
                true,
            ),
            field(
                "precipitation_chance",
                "precipitationChance",
                TypeRef::primitive("number"),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "WeatherUpdate",
        "A streaming weather update.",
        vec![
            field("timestamp", "timestamp", TypeRef::primitive("string"), true),
            field("city", "city", TypeRef::primitive("string"), true),
            field(
                "temperature",
                "temperature",
                TypeRef::primitive("number"),
                true,
            ),
            field(
                "conditions",
                "conditions",
                TypeRef::primitive("string"),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "FileInfo",
        "Information about an uploaded file.",
        vec![
            field("id", "id", TypeRef::primitive("string"), true),
            field("filename", "filename", TypeRef::primitive("string"), true),
            field("size", "size", TypeRef::primitive("number"), true),
            field(
                "content_type",
                "contentType",
                TypeRef::primitive("string"),
                true,
            ),
            field("checksum", "checksum", TypeRef::primitive("string"), true),
            field(
                "uploaded_at",
                "uploadedAt",
                TypeRef::primitive("string"),
                true,
            ),
        ],
    );
    insert_model(
        &mut graph,
        "ImageUploadResult",
        "Result of an image upload.",
        vec![
            field("id", "id", TypeRef::primitive("string"), true),
            field("filename", "filename", TypeRef::primitive("string"), true),
            null_default(field(
                "width",
                "width",
                optional_nullable(TypeRef::primitive("number")),
                false,
            )),
            null_default(field(
                "height",
                "height",
                optional_nullable(TypeRef::primitive("number")),
                false,
            )),
            null_default(field(
                "thumbnail_base64",
                "thumbnailBase64",
                optional_nullable(TypeRef::primitive("string")),
                false,
            )),
        ],
    );
    insert_model(
        &mut graph,
        "DocumentUploadResult",
        "Result of a document upload.",
        vec![
            field("id", "id", TypeRef::primitive("string"), true),
            field("filename", "filename", TypeRef::primitive("string"), true),
            field("size", "size", TypeRef::primitive("number"), true),
            null_default(field(
                "page_count",
                "pageCount",
                optional_nullable(TypeRef::primitive("number")),
                false,
            )),
        ],
    );
    insert_model(
        &mut graph,
        "ChatMessage",
        "A chat message sent by a user.",
        vec![
            field("user", "user", TypeRef::primitive("string"), true),
            field("text", "text", TypeRef::primitive("string"), true),
            field("timestamp", "timestamp", TypeRef::primitive("string"), true),
        ],
    );
    insert_model(
        &mut graph,
        "TypingIndicator",
        "Indicates a user is typing.",
        vec![
            field("user", "user", TypeRef::primitive("string"), true),
            field("is_typing", "isTyping", TypeRef::primitive("boolean"), true),
        ],
    );
    insert_model(
        &mut graph,
        "UserJoined",
        "Notification that a user joined the chat.",
        vec![
            field("user", "user", TypeRef::primitive("string"), true),
            field("timestamp", "timestamp", TypeRef::primitive("string"), true),
        ],
    );
    insert_model(
        &mut graph,
        "UserLeft",
        "Notification that a user left the chat.",
        vec![
            field("user", "user", TypeRef::primitive("string"), true),
            field("timestamp", "timestamp", TypeRef::primitive("string"), true),
        ],
    );
    insert_model(
        &mut graph,
        "ServerStatus",
        "Server status information.",
        vec![
            field(
                "connected_users",
                "connectedUsers",
                TypeRef::primitive("number"),
                true,
            ),
            field(
                "uptime_seconds",
                "uptimeSeconds",
                TypeRef::primitive("number"),
                true,
            ),
        ],
    );
    graph
}

fn defaulted(mut field: Field, value: Value) -> Field {
    field.default = Some(value);
    field
}

fn null_default(field: Field) -> Field {
    defaulted(field, Value::Null)
}

fn insert_model(graph: &mut ApiGraph, name: &str, doc: &str, fields: Vec<Field>) {
    let mut model = ModelDef::new(name);
    model.doc = Some(doc.to_string());
    model.fields = fields;
    graph.insert_model(model);
}

pub fn bridge() -> ZynkBridge {
    bridge_with_handlers()
}

pub fn bridge_with_debug(debug: bool) -> ZynkBridge {
    bridge_with_handlers().debug(debug)
}

pub fn router() -> Router {
    bridge_with_debug(false).configure(Router::new())
}

pub fn router_with_debug(debug: bool) -> Router {
    bridge_with_debug(debug).configure(Router::new())
}

pub fn dump_schema_json() -> String {
    bridge().dump_schema_json()
}

pub fn endpoint_names() -> Vec<String> {
    let graph: ApiGraph = serde_json::from_str(&dump_schema_json()).expect("valid schema graph");
    graph.endpoints.keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_contains_python_kitchen_sink_surface() {
        let graph: ApiGraph = serde_json::from_str(&dump_schema_json()).expect("schema json");
        assert!(graph.endpoints.contains_key("get_user"));
        assert!(graph.endpoints.contains_key("stream_weather"));
        assert!(graph.endpoints.contains_key("upload_file"));
        assert!(graph.endpoints.contains_key("download_sample"));
        assert!(graph.endpoints.contains_key("chat"));
        assert!(graph.models.contains_key("Task"));
        assert!(graph.enums.contains_key("TaskPriority"));
    }
}
