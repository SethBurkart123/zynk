use std::{path::PathBuf, process::Command};

use zynk_schema::{ApiGraph, EndpointKind, Field, TypeKind};

#[test]
fn python_bridge_dump_deserializes_without_loss_and_preserves_field_flags() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|crates_dir| crates_dir.parent())
        .expect("crate lives under <repo>/crates/zynk-schema");
    let python_dir = repo_root.join("bindings/python");

    let output = Command::new("uv")
        .args([
            "run",
            "python",
            "tests/fixtures/roundtrip_schema_fixture.py",
        ])
        .current_dir(&python_dir)
        .output()
        .expect("spawn uv to run Python schema fixture");

    assert!(
        output.status.success(),
        "Python fixture failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("schema dump stdout is utf-8");
    let graph: ApiGraph = serde_json::from_str(stdout.trim()).unwrap_or_else(|err| {
        panic!(
            "Python dump did not deserialize as zynk_schema::ApiGraph: {err}\nstdout:\n{}",
            stdout
        )
    });

    assert_endpoint_kinds_survived(&graph);
    assert_models_enums_and_type_refs_survived(&graph);
    assert_optional_nullable_flags_survived(&graph);
}

fn assert_endpoint_kinds_survived(graph: &ApiGraph) {
    assert_eq!(graph.endpoints.len(), 5, "expected one endpoint per kind");

    let update_profile = graph
        .endpoints
        .get("update_profile")
        .expect("rpc endpoint survived");
    assert_eq!(update_profile.kind, EndpointKind::Rpc);
    assert_eq!(update_profile.params.len(), 2);
    assert_eq!(update_profile.params[0].source_name, "profile");
    assert_eq!(update_profile.params[0].wire_name, "profile");
    assert!(update_profile.params[0].required);
    assert_eq!(update_profile.params[0].ty.kind, TypeKind::Model);
    assert_eq!(
        update_profile.params[0].ty.name.as_deref(),
        Some("RoundtripProfile")
    );
    assert_eq!(update_profile.params[1].source_name, "include_deleted");
    assert_eq!(update_profile.params[1].wire_name, "includeDeleted");
    assert!(!update_profile.params[1].required);
    assert_eq!(update_profile.params[1].ty.kind, TypeKind::Primitive);
    assert_eq!(update_profile.params[1].ty.name.as_deref(), Some("boolean"));
    assert_eq!(update_profile.returns.kind, TypeKind::Model);
    assert_eq!(
        update_profile.returns.name.as_deref(),
        Some("RoundtripProfile")
    );

    let stream_profiles = graph
        .endpoints
        .get("stream_profiles")
        .expect("channel endpoint survived");
    assert_eq!(stream_profiles.kind, EndpointKind::Channel);
    let channel_item = stream_profiles
        .channel_item
        .as_ref()
        .expect("channel item type survived");
    assert_eq!(channel_item.kind, TypeKind::Model);
    assert_eq!(channel_item.name.as_deref(), Some("RoundtripProfile"));
    assert_eq!(stream_profiles.params.len(), 1);
    assert_eq!(stream_profiles.params[0].source_name, "query");
    assert_eq!(stream_profiles.params[0].wire_name, "query");

    let upload_avatar = graph
        .endpoints
        .get("upload_avatar")
        .expect("upload endpoint survived");
    assert_eq!(upload_avatar.kind, EndpointKind::Upload);
    assert_eq!(upload_avatar.file_param.as_deref(), Some("file"));
    assert!(!upload_avatar.multi_file);
    assert_eq!(upload_avatar.max_size, Some(5 * 1024 * 1024));
    assert_eq!(upload_avatar.allowed_types, vec!["image/*".to_string()]);
    assert_eq!(upload_avatar.params.len(), 1);
    assert_eq!(upload_avatar.params[0].source_name, "profile_id");
    assert_eq!(upload_avatar.params[0].wire_name, "profileId");

    let avatar_file = graph
        .endpoints
        .get("avatar_file")
        .expect("static endpoint survived");
    assert_eq!(avatar_file.kind, EndpointKind::Static);
    assert_eq!(avatar_file.params.len(), 2);
    assert_eq!(avatar_file.params[0].source_name, "profile_id");
    assert_eq!(avatar_file.params[0].wire_name, "profileId");
    assert!(avatar_file.params[0].required);
    assert_eq!(avatar_file.params[1].source_name, "size");
    assert_eq!(avatar_file.params[1].wire_name, "size");
    assert!(!avatar_file.params[1].required);

    let profile_socket = graph
        .endpoints
        .get("profile_socket")
        .expect("websocket endpoint survived");
    assert_eq!(profile_socket.kind, EndpointKind::Ws);
    assert_eq!(profile_socket.server_events.len(), 2);
    assert_eq!(profile_socket.server_events[0].source_name, "heartbeat");
    assert_eq!(profile_socket.server_events[0].wire_name, "heartbeat");
    assert_eq!(profile_socket.server_events[0].ty.kind, TypeKind::Literal);
    assert_eq!(
        profile_socket.server_events[0].ty.value.as_ref(),
        Some(&serde_json::json!("pong"))
    );
    assert_eq!(
        profile_socket.server_events[1].source_name,
        "profile_updated"
    );
    assert_eq!(profile_socket.server_events[1].wire_name, "profileUpdated");
    assert_eq!(profile_socket.server_events[1].ty.kind, TypeKind::Model);
    assert_eq!(
        profile_socket.server_events[1].ty.name.as_deref(),
        Some("RoundtripProfile")
    );
    assert_eq!(profile_socket.client_events.len(), 1);
    assert_eq!(profile_socket.client_events[0].source_name, "subscribe");
    assert_eq!(profile_socket.client_events[0].wire_name, "subscribe");
    assert_eq!(profile_socket.client_events[0].ty.kind, TypeKind::Enum);
    assert_eq!(
        profile_socket.client_events[0].ty.name.as_deref(),
        Some("RoundtripPriority")
    );
}

fn assert_models_enums_and_type_refs_survived(graph: &ApiGraph) {
    assert_eq!(graph.models.len(), 2);
    assert!(graph.models.contains_key("RoundtripMetadata"));
    assert!(graph.models.contains_key("RoundtripProfile"));

    let priority = graph
        .enums
        .get("RoundtripPriority")
        .expect("enum definition survived");
    assert_eq!(
        priority.values,
        vec![serde_json::json!("low"), serde_json::json!("high")]
    );

    let metadata = graph
        .models
        .get("RoundtripMetadata")
        .expect("metadata model survived");
    assert_field(
        metadata_field(metadata, "tags"),
        false,
        false,
        true,
        TypeKind::Array,
    );
    assert_field(
        metadata_field(metadata, "attributes"),
        true,
        false,
        false,
        TypeKind::Record,
    );

    let profile = graph
        .models
        .get("RoundtripProfile")
        .expect("profile model survived");
    let metadata_ref = metadata_field(profile, "metadata");
    assert_field(metadata_ref, false, false, true, TypeKind::Model);
    assert_eq!(metadata_ref.ty.name.as_deref(), Some("RoundtripMetadata"));

    let priority_ref = metadata_field(profile, "priority");
    assert_field(priority_ref, false, false, true, TypeKind::Enum);
    assert_eq!(priority_ref.ty.name.as_deref(), Some("RoundtripPriority"));

    let status = metadata_field(profile, "status");
    assert_eq!(status.ty.kind, TypeKind::Literal);
    assert_eq!(status.ty.value.as_ref(), Some(&serde_json::json!("active")));
}

fn assert_optional_nullable_flags_survived(graph: &ApiGraph) {
    let profile = graph
        .models
        .get("RoundtripProfile")
        .expect("profile model survived");

    assert_field(
        metadata_field(profile, "id"),
        false,
        false,
        true,
        TypeKind::Primitive,
    );
    assert_field(
        metadata_field(profile, "displayName"),
        true,
        false,
        false,
        TypeKind::Primitive,
    );
    assert_field(
        metadata_field(profile, "nickname"),
        true,
        true,
        false,
        TypeKind::Primitive,
    );
    assert_field(
        metadata_field(profile, "status"),
        true,
        false,
        false,
        TypeKind::Literal,
    );
}

fn metadata_field<'a>(model: &'a zynk_schema::ModelDef, wire_name: &str) -> &'a Field {
    model
        .fields
        .iter()
        .find(|field| field.wire_name == wire_name)
        .unwrap_or_else(|| panic!("field {wire_name} exists on model {}", model.name))
}

fn assert_field(
    field: &Field,
    expected_optional: bool,
    expected_nullable: bool,
    expected_required: bool,
    expected_kind: TypeKind,
) {
    assert_eq!(
        field.optional, expected_optional,
        "{}.optional",
        field.wire_name
    );
    assert_eq!(
        field.nullable, expected_nullable,
        "{}.nullable",
        field.wire_name
    );
    assert_eq!(
        field.required, expected_required,
        "{}.required",
        field.wire_name
    );
    assert_eq!(field.ty.kind, expected_kind, "{}.ty.kind", field.wire_name);
    assert_eq!(
        field.ty.optional, expected_optional,
        "{}.ty.optional",
        field.wire_name
    );
    assert_eq!(
        field.ty.nullable, expected_nullable,
        "{}.ty.nullable",
        field.wire_name
    );
}
