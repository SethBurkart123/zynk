//! Lower Zynk schema type references into TypeScript type expressions.

use std::collections::BTreeMap;

use zynk_schema::{ApiGraph, EnumDef, TypeKind, TypeRef};

use crate::naming::to_pascal_case;

/// Position-sensitive TypeScript lowering context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypePosition {
    /// A field or function parameter type. Void becomes `undefined`.
    Value,
    /// A function return type. Void remains `void`.
    Return,
}

/// Lower a type reference in value position without enum definition context.
pub fn lower(ty: &TypeRef) -> String {
    lower_with_enums(ty, &BTreeMap::new(), TypePosition::Value)
}

/// Lower a type reference in return position without enum definition context.
pub fn lower_return(ty: &TypeRef) -> String {
    lower_with_enums(ty, &BTreeMap::new(), TypePosition::Return)
}

/// Lower a type reference in value position using enum definitions from a graph.
pub fn lower_with_graph(ty: &TypeRef, graph: &ApiGraph) -> String {
    lower_with_enums(ty, &graph.enums, TypePosition::Value)
}

/// Lower a type reference shape without applying its optional/nullable flags.
pub fn lower_required_with_graph(ty: &TypeRef, graph: &ApiGraph) -> String {
    lower_base(ty, &graph.enums, TypePosition::Value)
}

/// Lower a type reference using explicit enum definitions and position.
pub fn lower_with_enums(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    position: TypePosition,
) -> String {
    let base = lower_base(ty, enums, position);
    apply_optional_nullable(base, ty.optional, ty.nullable)
}

/// Apply the crate-wide optional/nullable convention to a lowered TypeScript type.
pub fn apply_optional_nullable(base: String, optional: bool, nullable: bool) -> String {
    let mut parts = vec![base];
    if nullable {
        parts.push("null".to_string());
    }
    if optional {
        parts.push("undefined".to_string());
    }
    dedupe_union(parts)
}

fn lower_base(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>, position: TypePosition) -> String {
    match ty.kind {
        TypeKind::Primitive => lower_primitive(ty.name.as_deref()),
        TypeKind::Array => lower_array(ty, enums),
        TypeKind::Record => lower_record(ty, enums),
        TypeKind::Tuple => lower_tuple(ty, enums),
        TypeKind::Union => lower_union(ty, enums, position),
        TypeKind::Literal => ty
            .value
            .as_ref()
            .map(lower_literal_value)
            .unwrap_or_else(|| "unknown".to_string()),
        TypeKind::Model => lower_named_ref(ty.name.as_deref()),
        TypeKind::Enum => lower_enum(ty, enums),
        TypeKind::Any => "unknown".to_string(),
        TypeKind::Void => match position {
            TypePosition::Value => "undefined".to_string(),
            TypePosition::Return => "void".to_string(),
        },
    }
}

fn lower_primitive(name: Option<&str>) -> String {
    match name {
        Some("string") | Some("str") | Some("bytes") => "string".to_string(),
        Some("number") | Some("int") | Some("float") => "number".to_string(),
        Some("boolean") | Some("bool") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("undefined") => "undefined".to_string(),
        Some("any") | Some("unknown") => "unknown".to_string(),
        _ => "unknown".to_string(),
    }
}

fn lower_array(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> String {
    let inner = ty
        .inner
        .first()
        .map(|item| lower_with_enums(item, enums, TypePosition::Value))
        .unwrap_or_else(|| "unknown".to_string());

    if needs_array_parens(&inner) {
        format!("({inner})[]")
    } else {
        format!("{inner}[]")
    }
}

fn lower_record(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> String {
    let key = ty
        .inner
        .first()
        .map(|item| lower_record_key(item, enums))
        .unwrap_or_else(|| "string".to_string());
    let value = ty
        .inner
        .get(1)
        .map(|item| lower_with_enums(item, enums, TypePosition::Value))
        .unwrap_or_else(|| "unknown".to_string());

    format!("Record<{key}, {value}>")
}

fn lower_record_key(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> String {
    let lowered = lower_with_enums(ty, enums, TypePosition::Value);
    if lowered == "number" {
        "number".to_string()
    } else if lowered == "string" || lowered.contains("\"") {
        "string".to_string()
    } else {
        "string | number".to_string()
    }
}

fn lower_tuple(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> String {
    let items = ty
        .inner
        .iter()
        .map(|item| lower_with_enums(item, enums, TypePosition::Value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

fn lower_union(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>, position: TypePosition) -> String {
    dedupe_union(
        ty.inner
            .iter()
            .map(|item| lower_with_enums(item, enums, position))
            .collect(),
    )
}

fn lower_named_ref(name: Option<&str>) -> String {
    name.map(to_pascal_case)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn lower_enum(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> String {
    let Some(name) = ty.name.as_deref() else {
        return "unknown".to_string();
    };

    match enums.get(name) {
        Some(enum_def) if !enum_def.values.is_empty() => dedupe_union(
            enum_def
                .values
                .iter()
                .map(lower_literal_value)
                .collect::<Vec<_>>(),
        ),
        _ => lower_named_ref(Some(name)),
    }
}

fn lower_literal_value(value: &zynk_schema::Value) -> String {
    match value {
        zynk_schema::Value::String(value) => format!("\"{}\"", escape_ts_string(value)),
        zynk_schema::Value::Number(value) => value.to_string(),
        zynk_schema::Value::Bool(value) => value.to_string(),
        zynk_schema::Value::Null => "null".to_string(),
        zynk_schema::Value::Array(_) | zynk_schema::Value::Object(_) => "unknown".to_string(),
    }
}

fn escape_ts_string(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
}

fn needs_array_parens(lowered: &str) -> bool {
    lowered.contains(" | ") || lowered.starts_with('[')
}

fn dedupe_union(parts: Vec<String>) -> String {
    let mut output = Vec::new();
    for part in parts {
        if part.is_empty() || output.contains(&part) {
            continue;
        }
        output.push(part);
    }

    if output.is_empty() {
        "unknown".to_string()
    } else {
        output.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use zynk_schema::{ApiGraph, EnumDef, TypeKind, TypeRef, Value};

    use super::{apply_optional_nullable, lower, lower_return, lower_with_graph};

    macro_rules! json {
        (null) => {
            Value::Null
        };
        ($value:literal) => {
            Value::from($value)
        };
    }

    #[test]
    fn lowers_string_number_boolean_null_any_unknown_primitives() {
        assert_eq!(lower(&TypeRef::primitive("string")), "string");
        assert_eq!(lower(&TypeRef::primitive("number")), "number");
        assert_eq!(lower(&TypeRef::primitive("boolean")), "boolean");
        assert_eq!(lower(&TypeRef::primitive("null")), "null");
        assert_eq!(lower(&TypeRef::primitive("any")), "unknown");
        assert_eq!(lower(&TypeRef::primitive("unknown")), "unknown");
        assert_eq!(lower(&TypeRef::primitive("mystery")), "unknown");
        assert_eq!(lower(&TypeRef::new(TypeKind::Primitive)), "unknown");
    }

    #[test]
    fn lowers_any_type_kind_to_unknown() {
        assert_eq!(lower(&TypeRef::any()), "unknown");
    }

    #[test]
    fn lowers_void_by_position() {
        assert_eq!(lower(&TypeRef::void()), "undefined");
        assert_eq!(lower_return(&TypeRef::void()), "void");
    }

    #[test]
    fn lowers_arrays_and_parenthesizes_union_items() {
        assert_eq!(
            lower(&TypeRef::array(TypeRef::primitive("number"))),
            "number[]"
        );
        assert_eq!(
            lower(&TypeRef::array(TypeRef::union(vec![
                TypeRef::literal(json!("a")),
                TypeRef::literal(json!("b")),
            ]))),
            "(\"a\" | \"b\")[]"
        );
    }

    #[test]
    fn lowers_records_with_key_coercion() {
        assert_eq!(
            lower(&TypeRef::record(
                TypeRef::primitive("string"),
                TypeRef::primitive("number")
            )),
            "Record<string, number>"
        );
        assert_eq!(
            lower(&TypeRef::record(
                TypeRef::primitive("boolean"),
                TypeRef::primitive("string")
            )),
            "Record<string | number, string>"
        );
    }

    #[test]
    fn lowers_tuples() {
        let ty = TypeRef {
            inner: vec![TypeRef::primitive("string"), TypeRef::primitive("number")],
            ..TypeRef::new(TypeKind::Tuple)
        };

        assert_eq!(lower(&ty), "[string, number]");
    }

    #[test]
    fn lowers_unions_and_deduplicates_members() {
        assert_eq!(
            lower(&TypeRef::union(vec![
                TypeRef::primitive("string"),
                TypeRef::primitive("number"),
                TypeRef::primitive("string"),
            ])),
            "string | number"
        );
    }

    #[test]
    fn lowers_literals_to_exact_ts_values() {
        assert_eq!(lower(&TypeRef::literal(json!("foo"))), "\"foo\"");
        assert_eq!(lower(&TypeRef::literal(json!("a\"b"))), "\"a\\\"b\"");
        assert_eq!(lower(&TypeRef::literal(json!(42))), "42");
        assert_eq!(lower(&TypeRef::literal(json!(true))), "true");
        assert_eq!(lower(&TypeRef::literal(json!(null))), "null");
    }

    #[test]
    fn lowers_literal_unions_without_unknown() {
        let lowered = lower(&TypeRef::union(vec![
            TypeRef::literal(json!("a")),
            TypeRef::literal(json!("b")),
            TypeRef::literal(json!("c")),
        ]));

        assert_eq!(lowered, "\"a\" | \"b\" | \"c\"");
        assert!(!lowered.contains("unknown"));
    }

    #[test]
    fn lowers_enum_with_values_to_value_union() {
        let mut graph = ApiGraph::new();
        let mut enum_def = EnumDef::new("priority");
        enum_def.values = vec![json!("a"), json!("b")];
        graph.insert_enum(enum_def);

        assert_eq!(
            lower_with_graph(&TypeRef::enum_ref("priority"), &graph),
            "\"a\" | \"b\""
        );
    }

    #[test]
    fn lowers_numeric_enum_with_values_to_value_union() {
        let mut graph = ApiGraph::new();
        let mut enum_def = EnumDef::new("status_code");
        enum_def.values = vec![json!(1), json!(2), json!(3)];
        graph.insert_enum(enum_def);

        assert_eq!(
            lower_with_graph(&TypeRef::enum_ref("status_code"), &graph),
            "1 | 2 | 3"
        );
    }

    #[test]
    fn enum_without_values_degrades_to_pascal_case_reference() {
        let mut graph = ApiGraph::new();
        graph.insert_enum(EnumDef::new("priority_level"));

        assert_eq!(
            lower_with_graph(&TypeRef::enum_ref("priority_level"), &graph),
            "PriorityLevel"
        );
    }

    #[test]
    fn lowers_model_references_to_pascal_case() {
        assert_eq!(lower(&TypeRef::model("user_profile")), "UserProfile");
    }

    #[test]
    fn applies_canonical_optional_nullable_union_style() {
        assert_eq!(
            apply_optional_nullable("string".to_string(), false, false),
            "string"
        );
        assert_eq!(
            apply_optional_nullable("string".to_string(), true, false),
            "string | undefined"
        );
        assert_eq!(
            apply_optional_nullable("string".to_string(), false, true),
            "string | null"
        );
        assert_eq!(
            apply_optional_nullable("string".to_string(), true, true),
            "string | null | undefined"
        );
    }

    #[test]
    fn type_ref_optional_nullable_flags_are_applied_uniformly() {
        let mut ty = TypeRef::primitive("string");
        ty.optional = true;
        ty.nullable = true;

        assert_eq!(lower(&ty), "string | null | undefined");
    }
}
