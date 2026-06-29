//! Lower Zynk schema type references into Effect Schema expressions.

use std::collections::{BTreeMap, BTreeSet};

use zynk_schema::{ApiGraph, EnumDef, Field, ModelDef, TypeKind, TypeRef, Value};

use crate::topological;
use crate::types::TypeExpr;

/// Lower a type reference without enum definition context.
pub fn lower(ty: &TypeRef) -> TypeExpr {
    lower_with_enums(ty, &BTreeMap::new(), None)
}

/// Lower a type reference using enum definitions from an API graph.
pub fn lower_with_graph(ty: &TypeRef, graph: &ApiGraph) -> TypeExpr {
    lower_with_enums(ty, &graph.enums, None)
}

/// Lower a type reference while rendering a particular model.
///
/// Self-references to `self_name` are wrapped in `Schema.suspend`.
pub fn lower_for_model(ty: &TypeRef, graph: &ApiGraph, self_name: &str) -> TypeExpr {
    lower_with_enums(ty, &graph.enums, Some(self_name))
}

/// Lower a type reference using explicit enum definitions.
pub fn lower_with_enums(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    let base = lower_base(ty, enums, self_name);
    apply_optional_nullable(base, ty.optional, ty.nullable)
}

/// Apply Effect generator optional/nullable wrapping for a lowered expression.
pub fn apply_optional_nullable(expr: TypeExpr, optional: bool, nullable: bool) -> TypeExpr {
    match (optional, nullable) {
        (true, true) => TypeExpr::new(
            append_ts_flags(expr.ts, true, true),
            format!("Schema.optionalWith({}, {{ nullable: true }})", expr.schema),
        ),
        (true, false) => TypeExpr::new(
            append_ts_flags(expr.ts, true, false),
            format!("Schema.UndefinedOr({})", expr.schema),
        ),
        (false, true) => TypeExpr::new(
            append_ts_flags(expr.ts, false, true),
            format!("Schema.NullOr({})", expr.schema),
        ),
        (false, false) => expr,
    }
}

fn append_ts_flags(base: String, optional: bool, nullable: bool) -> String {
    let mut parts = base.split(" | ").map(str::to_string).collect::<Vec<_>>();
    if nullable && !parts.iter().any(|part| part == "null") {
        parts.push("null".to_string());
    }
    if optional && !parts.iter().any(|part| part == "undefined") {
        parts.push("undefined".to_string());
    }
    parts.join(" | ")
}

fn lower_base(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    match ty.kind {
        TypeKind::Primitive => lower_primitive(ty.name.as_deref()),
        TypeKind::Array => lower_array(ty, enums, self_name),
        TypeKind::Record => lower_record(ty, enums, self_name),
        TypeKind::Tuple => lower_tuple(ty, enums, self_name),
        TypeKind::Union => lower_union(ty, enums, self_name),
        TypeKind::Literal => ty
            .value
            .as_ref()
            .map(lower_literal_value)
            .unwrap_or_else(|| TypeExpr::new("unknown", "Schema.Unknown")),
        TypeKind::Model => lower_model_ref(ty.name.as_deref(), self_name),
        TypeKind::Enum => lower_enum(ty, enums),
        TypeKind::Any => TypeExpr::new("unknown", "Schema.Unknown"),
        TypeKind::Void => TypeExpr::new("void", "Schema.Void"),
    }
}

fn lower_primitive(name: Option<&str>) -> TypeExpr {
    match name {
        Some("string") | Some("str") | Some("bytes") => TypeExpr::new("string", "Schema.String"),
        Some("number") | Some("int") | Some("float") => TypeExpr::new("number", "Schema.Number"),
        Some("boolean") | Some("bool") => TypeExpr::new("boolean", "Schema.Boolean"),
        Some("undefined") => TypeExpr::new("undefined", "Schema.Undefined"),
        Some("null") => TypeExpr::new("null", "Schema.Null"),
        Some("any") | Some("unknown") => TypeExpr::new("unknown", "Schema.Unknown"),
        _ => TypeExpr::new("unknown", "Schema.Unknown"),
    }
}

fn lower_array(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    let inner = ty
        .inner
        .first()
        .map(|item| lower_with_enums(item, enums, self_name))
        .unwrap_or_else(|| TypeExpr::new("void", "Schema.Void"));

    TypeExpr::new(
        format!("ReadonlyArray<{}>", inner.ts),
        format!("Schema.Array({})", inner.schema),
    )
}

fn lower_record(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    let key = ty
        .inner
        .first()
        .map(|item| lower_with_enums(item, enums, self_name))
        .unwrap_or_else(|| TypeExpr::new("void", "Schema.Void"));
    let value = ty
        .inner
        .get(1)
        .map(|item| lower_with_enums(item, enums, self_name))
        .unwrap_or_else(|| TypeExpr::new("void", "Schema.Void"));

    let ts_key = if key.ts == "string" || key.ts == "number" {
        key.ts
    } else {
        "string".to_string()
    };
    let schema_key = if key.schema == "Schema.String" || key.schema == "Schema.Number" {
        key.schema
    } else {
        "Schema.String".to_string()
    };

    TypeExpr::new(
        format!("Readonly<Record<{ts_key}, {}>>", value.ts),
        format!(
            "Schema.Record({{ key: {schema_key}, value: {} }})",
            value.schema
        ),
    )
}

fn lower_tuple(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    let parts = ty
        .inner
        .iter()
        .map(|item| lower_with_enums(item, enums, self_name))
        .collect::<Vec<_>>();
    TypeExpr::new(
        format!(
            "readonly [{}]",
            parts
                .iter()
                .map(|part| part.ts.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!(
            "Schema.Tuple({})",
            parts
                .iter()
                .map(|part| part.schema.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn lower_union(
    ty: &TypeRef,
    enums: &BTreeMap<String, EnumDef>,
    self_name: Option<&str>,
) -> TypeExpr {
    let parts = ty
        .inner
        .iter()
        .map(|item| lower_with_enums(item, enums, self_name))
        .collect::<Vec<_>>();

    if parts.is_empty() {
        return TypeExpr::new("unknown", "Schema.Unknown");
    }

    TypeExpr::new(
        dedupe_union(parts.iter().map(|part| part.ts.clone()).collect()),
        format!(
            "Schema.Union({})",
            parts
                .iter()
                .map(|part| part.schema.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn lower_model_ref(name: Option<&str>, self_name: Option<&str>) -> TypeExpr {
    let name = name
        .map(zynk_schema::naming::to_pascal_case)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let schema = if Some(name.as_str()) == self_name {
        format!("Schema.suspend(() => {name})")
    } else if name == "unknown" {
        "Schema.Unknown".to_string()
    } else {
        name.clone()
    };

    TypeExpr::new(name, schema)
}

fn lower_enum(ty: &TypeRef, enums: &BTreeMap<String, EnumDef>) -> TypeExpr {
    let Some(name) = ty.name.as_deref() else {
        return TypeExpr::new("string", "Schema.String");
    };
    let Some(enum_def) = enums.get(name) else {
        return TypeExpr::new("string", "Schema.String");
    };
    if enum_def.values.is_empty() {
        return TypeExpr::new("string", "Schema.String");
    }

    let literals = enum_def
        .values
        .iter()
        .map(lower_literal_value)
        .collect::<Vec<_>>();
    if literals.iter().any(|literal| literal.ts == "unknown") {
        return TypeExpr::new("string", "Schema.String");
    }

    TypeExpr::new(
        literals
            .iter()
            .map(|literal| literal.ts.clone())
            .collect::<Vec<_>>()
            .join(" | "),
        format!(
            "Schema.Literal({})",
            literals
                .into_iter()
                .map(|literal| literal.ts)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn lower_literal_value(value: &Value) -> TypeExpr {
    match value {
        Value::String(value) => {
            let literal = quote_js_string(value);
            TypeExpr::new(literal.clone(), format!("Schema.Literal({literal})"))
        }
        Value::Number(value) => {
            TypeExpr::new(value.to_string(), format!("Schema.Literal({value})"))
        }
        Value::Bool(value) => TypeExpr::new(value.to_string(), format!("Schema.Literal({value})")),
        Value::Null => TypeExpr::new("null", "Schema.Literal(null)"),
        Value::Array(_) | Value::Object(_) => TypeExpr::new("unknown", "Schema.Unknown"),
    }
}

fn quote_js_string(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a string literal cannot fail")
}

fn dedupe_union(parts: Vec<String>) -> String {
    let mut out = Vec::new();
    for part in parts {
        if !out.contains(&part) {
            out.push(part);
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out.join(" | ")
    }
}

/// Collect every model name reachable from a type reference.
pub fn collect_model_refs(ty: &TypeRef) -> BTreeSet<String> {
    let mut refs = BTreeSet::new();
    walk_model_refs(ty, &mut refs);
    refs
}

fn walk_model_refs(ty: &TypeRef, refs: &mut BTreeSet<String>) {
    if ty.kind == TypeKind::Model {
        if let Some(name) = ty.name.as_deref() {
            refs.insert(zynk_schema::naming::to_pascal_case(name));
        }
    }
    for inner in &ty.inner {
        walk_model_refs(inner, refs);
    }
}

/// Render model schemas in dependency order.
pub fn render_model_schemas(graph: &ApiGraph) -> Vec<String> {
    let deps = graph
        .models
        .iter()
        .map(|(name, model)| {
            let model_name = zynk_schema::naming::to_pascal_case(name);
            let mut model_deps = model
                .fields
                .iter()
                .flat_map(|field| collect_model_refs(&field.ty))
                .collect::<BTreeSet<_>>();
            model_deps.remove(&model_name);
            (model_name, model_deps)
        })
        .collect::<BTreeMap<_, _>>();

    topological::sort_models(&deps)
        .into_iter()
        .filter_map(|name| find_model(graph, &name).map(|model| render_model_schema(model, graph)))
        .collect()
}

fn find_model<'a>(graph: &'a ApiGraph, pascal_name: &str) -> Option<&'a ModelDef> {
    graph
        .models
        .iter()
        .find(|(name, model)| {
            zynk_schema::naming::to_pascal_case(name) == pascal_name
                || zynk_schema::naming::to_pascal_case(&model.name) == pascal_name
        })
        .map(|(_, model)| model)
}

fn render_model_schema(model: &ModelDef, graph: &ApiGraph) -> String {
    let name = zynk_schema::naming::to_pascal_case(&model.name);
    let rendered_fields = model
        .fields
        .iter()
        .map(|field| render_field(field, graph, &name))
        .collect::<Vec<_>>();
    let fields = rendered_fields
        .iter()
        .map(|field| field.schema_line.as_str())
        .collect::<Vec<_>>()
        .join(",\n");
    let referenced = model
        .fields
        .iter()
        .flat_map(|field| collect_model_refs(&field.ty))
        .collect::<BTreeSet<_>>();
    let is_self_recursive = referenced.contains(&name);

    let type_alias = format!("export type {name} = Schema.Schema.Type<typeof {name}>");

    if is_self_recursive {
        let iface = render_recursive_interface(&name, &rendered_fields);
        let schema = if fields.is_empty() {
            format!("export const {name}: Schema.Schema<{name}, any> = Schema.Struct({{}})")
        } else {
            format!(
                "export const {name}: Schema.Schema<{name}, any> = Schema.Struct({{\n{fields}\n}})"
            )
        };
        format!("{iface}\n\n{schema}")
    } else if fields.is_empty() {
        format!("export const {name} = Schema.Struct({{}})\n\n{type_alias}")
    } else {
        format!("export const {name} = Schema.Struct({{\n{fields}\n}})\n\n{type_alias}")
    }
}

#[derive(Debug, Clone)]
struct RenderedField {
    schema_line: String,
    interface_line: String,
}

fn render_recursive_interface(name: &str, fields: &[RenderedField]) -> String {
    if fields.is_empty() {
        return format!("export type {name} = {{}}");
    }

    format!(
        "export interface {name} {{\n{}\n}}",
        fields
            .iter()
            .map(|field| field.interface_line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn render_field(field: &Field, graph: &ApiGraph, self_name: &str) -> RenderedField {
    let mut ty = field.ty.clone();
    ty.optional = ty.optional || field.optional || !field.required;
    ty.nullable = ty.nullable || field.nullable;

    let expr = lower_for_model(&ty, graph, self_name);
    let ts_name = field_ts_name(field);
    let schema = if ts_name != field.wire_name {
        wrap_schema_from_key(&expr.schema, ty.optional, ty.nullable, &field.wire_name)
    } else {
        expr.schema.clone()
    };
    let optional_marker = if ty.optional && ty.nullable { "?" } else { "" };

    RenderedField {
        schema_line: format!("  {ts_name}: {schema}"),
        interface_line: format!("  readonly {ts_name}{optional_marker}: {}", expr.ts),
    }
}

fn field_ts_name(field: &Field) -> String {
    field.wire_name.clone()
}

fn wrap_schema_from_key(schema: &str, optional: bool, nullable: bool, source_name: &str) -> String {
    let key = quote_js_string(source_name);
    if optional && nullable {
        format!("{schema}.pipe(Schema.fromKey({key}))")
    } else {
        format!("Schema.propertySignature({schema}).pipe(Schema.fromKey({key}))")
    }
}

#[cfg(test)]
mod tests {
    use zynk_schema::{ApiGraph, EnumDef, Field, ModelDef, TypeKind, TypeRef, Value};

    use super::{
        collect_model_refs, lower, lower_for_model, lower_with_graph, render_model_schemas,
    };

    macro_rules! json {
        (null) => {
            Value::Null
        };
        ($value:literal) => {
            Value::from($value)
        };
    }

    #[test]
    fn lowers_primitives_and_any_unknown_to_canonical_schema() {
        assert_eq!(lower(&TypeRef::primitive("string")).schema, "Schema.String");
        assert_eq!(lower(&TypeRef::primitive("number")).schema, "Schema.Number");
        assert_eq!(
            lower(&TypeRef::primitive("boolean")).schema,
            "Schema.Boolean"
        );
        assert_eq!(lower(&TypeRef::primitive("null")).schema, "Schema.Null");
        assert_eq!(
            lower(&TypeRef::primitive("unknown")).schema,
            "Schema.Unknown"
        );
        assert_eq!(lower(&TypeRef::primitive("any")).schema, "Schema.Unknown");
        assert_eq!(lower(&TypeRef::any()).schema, "Schema.Unknown");
        assert_eq!(lower(&TypeRef::void()).schema, "Schema.Void");
    }

    #[test]
    fn applies_optional_nullable_flags_like_python_type_expr() {
        let mut optional = TypeRef::primitive("string");
        optional.optional = true;
        assert_eq!(lower(&optional).ts, "string | undefined");
        assert_eq!(lower(&optional).schema, "Schema.UndefinedOr(Schema.String)");

        let mut nullable = TypeRef::primitive("string");
        nullable.nullable = true;
        assert_eq!(lower(&nullable).ts, "string | null");
        assert_eq!(lower(&nullable).schema, "Schema.NullOr(Schema.String)");

        let mut both = TypeRef::primitive("string");
        both.optional = true;
        both.nullable = true;
        assert_eq!(lower(&both).ts, "string | null | undefined");
        assert_eq!(
            lower(&both).schema,
            "Schema.optionalWith(Schema.String, { nullable: true })"
        );
    }

    #[test]
    fn lowers_structural_type_refs_to_effect_schema_and_ts_types() {
        let array = lower(&TypeRef::array(TypeRef::primitive("number")));
        assert_eq!(array.ts, "ReadonlyArray<number>");
        assert_eq!(array.schema, "Schema.Array(Schema.Number)");

        let record = lower(&TypeRef::record(
            TypeRef::primitive("boolean"),
            TypeRef::primitive("string"),
        ));
        assert_eq!(record.ts, "Readonly<Record<string, string>>");
        assert_eq!(
            record.schema,
            "Schema.Record({ key: Schema.String, value: Schema.String })"
        );

        let tuple = TypeRef {
            inner: vec![TypeRef::primitive("string"), TypeRef::primitive("number")],
            ..TypeRef::new(TypeKind::Tuple)
        };
        let tuple = lower(&tuple);
        assert_eq!(tuple.ts, "readonly [string, number]");
        assert_eq!(tuple.schema, "Schema.Tuple(Schema.String, Schema.Number)");

        let union = lower(&TypeRef::union(vec![
            TypeRef::primitive("string"),
            TypeRef::primitive("number"),
        ]));
        assert_eq!(union.ts, "string | number");
        assert_eq!(union.schema, "Schema.Union(Schema.String, Schema.Number)");
    }

    #[test]
    fn lowers_literals_precisely_and_literal_unions_to_schema_union() {
        assert_eq!(
            lower(&TypeRef::literal(json!("a\"b"))).schema,
            "Schema.Literal(\"a\\\"b\")"
        );
        assert_eq!(
            lower(&TypeRef::literal(json!(3))).schema,
            "Schema.Literal(3)"
        );
        assert_eq!(
            lower(&TypeRef::literal(json!(true))).schema,
            "Schema.Literal(true)"
        );
        assert_eq!(
            lower(&TypeRef::literal(json!(null))).schema,
            "Schema.Literal(null)"
        );

        let union = lower(&TypeRef::union(vec![
            TypeRef::literal(json!("a")),
            TypeRef::literal(json!("b")),
        ]));
        assert_eq!(union.ts, "\"a\" | \"b\"");
        assert_eq!(
            union.schema,
            "Schema.Union(Schema.Literal(\"a\"), Schema.Literal(\"b\"))"
        );
    }

    #[test]
    fn lowers_string_enum_values_to_effect_literal_schema() {
        let mut graph = ApiGraph::new();
        let mut priority = EnumDef::new("Priority");
        priority.values = vec![json!("low"), json!("high")];
        graph.insert_enum(priority);

        let expr = lower_with_graph(&TypeRef::enum_ref("Priority"), &graph);

        assert_eq!(expr.ts, "\"low\" | \"high\"");
        assert_eq!(expr.schema, "Schema.Literal(\"low\", \"high\")");
    }

    #[test]
    fn lowers_numeric_enum_values_to_effect_literal_schema() {
        let mut graph = ApiGraph::new();
        let mut status = EnumDef::new("Status");
        status.values = vec![json!(1), json!(2), json!(3)];
        graph.insert_enum(status);

        let expr = lower_with_graph(&TypeRef::enum_ref("Status"), &graph);

        assert_eq!(expr.ts, "1 | 2 | 3");
        assert_eq!(expr.schema, "Schema.Literal(1, 2, 3)");
    }

    #[test]
    fn lowers_model_references_and_self_references() {
        assert_eq!(lower(&TypeRef::model("user_profile")).ts, "UserProfile");
        assert_eq!(lower(&TypeRef::model("user_profile")).schema, "UserProfile");
        assert_eq!(
            lower_for_model(&TypeRef::model("Node"), &ApiGraph::new(), "Node").schema,
            "Schema.suspend(() => Node)"
        );
    }

    #[test]
    fn render_model_schemas_apply_field_optional_nullable_matrix() {
        let mut graph = ApiGraph::new();
        let mut model = ModelDef::new("Matrix");
        model.fields.push(Field::new(
            "required_name",
            "requiredName",
            TypeRef::primitive("string"),
            true,
        ));

        let mut optional = TypeRef::primitive("string");
        optional.optional = true;
        model
            .fields
            .push(Field::new("optional_name", "optionalName", optional, false));

        let mut nullable = TypeRef::primitive("string");
        nullable.nullable = true;
        model
            .fields
            .push(Field::new("nullable_name", "nullableName", nullable, true));

        let mut both = TypeRef::primitive("string");
        both.optional = true;
        both.nullable = true;
        model
            .fields
            .push(Field::new("both_name", "bothName", both, false));
        graph.insert_model(model);

        let rendered = render_model_schemas(&graph).join("\n");

        assert!(rendered.contains("requiredName: Schema.String"));
        assert!(rendered.contains("optionalName: Schema.UndefinedOr(Schema.String)"));
        assert!(rendered.contains("nullableName: Schema.NullOr(Schema.String)"));
        assert!(
            rendered.contains("bothName: Schema.optionalWith(Schema.String, { nullable: true })")
        );
    }

    #[test]
    fn render_model_schemas_emit_self_recursive_interfaces_before_schemas() {
        let mut graph = ApiGraph::new();
        let mut node = ModelDef::new("Node");
        node.fields.push(Field::new(
            "children",
            "children",
            TypeRef::array(TypeRef::model("Node")),
            true,
        ));
        graph.insert_model(node);

        let rendered = render_model_schemas(&graph).join("\n");
        let iface = rendered
            .find("export interface Node")
            .expect("interface emitted");
        let schema = rendered.find("export const Node").expect("schema emitted");

        assert!(
            iface < schema,
            "recursive interface must precede schema: {rendered}"
        );
        assert!(rendered.contains("export const Node: Schema.Schema<Node, any> = Schema.Struct"));
        assert!(rendered.contains("Schema.Array(Schema.suspend(() => Node))"));
    }

    #[test]
    fn render_model_schemas_use_wire_keys_for_effect_schema() {
        let mut graph = ApiGraph::new();
        let mut profile = ModelDef::new("UserProfile");
        profile.fields.push(Field::new(
            "user_id",
            "userId",
            TypeRef::primitive("number"),
            true,
        ));
        profile.fields.push(Field::new(
            "full_name",
            "fullName",
            TypeRef::primitive("string"),
            true,
        ));
        profile.fields.push(Field::new(
            "quoted_\"key",
            "quotedKey",
            TypeRef::primitive("string"),
            true,
        ));
        profile
            .fields
            .push(Field::new("id", "id", TypeRef::primitive("number"), true));
        graph.insert_model(profile);

        let rendered = render_model_schemas(&graph).join("\n");

        assert!(rendered.contains("userId: Schema.Number"));
        assert!(rendered.contains("fullName: Schema.String"));
        assert!(rendered.contains("quotedKey: Schema.String"));
        assert!(rendered.contains("id: Schema.Number"));
        assert!(!rendered.contains("Schema.fromKey"));
    }

    #[test]
    fn collect_refs_and_render_model_schemas_in_dependency_order() {
        let mut graph = ApiGraph::new();
        let mut z = ModelDef::new("Z");
        z.fields
            .push(Field::new("m", "m", TypeRef::model("M"), true));
        z.fields
            .push(Field::new("a", "a", TypeRef::model("A"), true));
        graph.insert_model(z);
        let mut m = ModelDef::new("M");
        m.fields
            .push(Field::new("a", "a", TypeRef::model("A"), true));
        graph.insert_model(m);
        graph.insert_model(ModelDef::new("A"));

        assert_eq!(
            collect_model_refs(&TypeRef::array(TypeRef::model("M")))
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["M"]
        );

        let rendered = render_model_schemas(&graph).join("\n");
        let a = rendered.find("export const A").expect("A emitted");
        let m = rendered.find("export const M").expect("M emitted");
        let z = rendered.find("export const Z").expect("Z emitted");
        assert!(
            a < m && m < z,
            "models rendered in dependency order: {rendered}"
        );
    }
}
