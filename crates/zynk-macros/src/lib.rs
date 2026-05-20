//! Proc macros for registering Rust Zynk endpoints.
//!
//! The attribute macros in this crate preserve the user's function unchanged
//! and add link-time `EndpointMeta` registrations consumed by server bindings.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_macro_input, punctuated::Punctuated, spanned::Spanned, Expr, ExprLit,
    FnArg, GenericArgument, ItemFn, Lit, Meta, PathArguments, ReturnType, Token, Type,
};

#[proc_macro_attribute]
pub fn command(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    expand_endpoint(input, EndpointSurface::Command)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_attribute]
pub fn message(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    expand_endpoint(input, EndpointSurface::Message)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_attribute]
pub fn upload(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = match parse_upload_config(attr) {
        Ok(config) => config,
        Err(error) => return error.into_compile_error().into(),
    };
    let input = parse_macro_input!(item as ItemFn);
    expand_endpoint(input, EndpointSurface::Upload(config))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_attribute]
pub fn static_file(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    expand_endpoint(input, EndpointSurface::StaticFile)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Default)]
struct UploadConfig {
    max_size: Option<u64>,
    allowed_types: Vec<String>,
}

enum EndpointSurface {
    Command,
    Message,
    Upload(UploadConfig),
    StaticFile,
}

struct UploadParts {
    params: Vec<ParamTokens>,
    file_param: String,
    multi_file: bool,
}

fn expand_endpoint(
    function: ItemFn,
    surface: EndpointSurface,
) -> syn::Result<proc_macro2::TokenStream> {
    reject_methods(&function)?;

    let name = function.sig.ident.to_string();
    let params_ident = format_ident!(
        "__ZYNK_{}_PARAMS",
        function.sig.ident.to_string().to_uppercase()
    );
    let returns = return_type_from_signature(&function.sig.output);

    let registration = match surface {
        EndpointSurface::Command => {
            let params = params_from_signature(&function)?;
            let params_tokens = params.iter().map(ParamTokens::to_tokens);

            quote! {
                #[allow(non_upper_case_globals)]
                const #params_ident: &[::zynk_runtime::ParamMeta] = &[#(#params_tokens),*];

                ::zynk_runtime::inventory::submit! {
                    ::zynk_runtime::EndpointMeta {
                        name: #name,
                        kind: ::zynk_runtime::EndpointKind::Rpc,
                        module: Some(module_path!()),
                        doc: None,
                        params: #params_ident,
                        returns: #returns,
                        channel_item: None,
                        file_param: None,
                        multi_file: false,
                        max_size: None,
                        allowed_types: &[],
                        server_events: &[],
                        client_events: &[],
                        handler_key: Some(::zynk_runtime::HandlerKey(concat!(module_path!(), "::", #name))),
                    }
                }
            }
        }
        EndpointSurface::Message => {
            let params = params_from_signature(&function)?;
            let params_tokens = params.iter().map(ParamTokens::to_tokens);

            quote! {
                #[allow(non_upper_case_globals)]
                const #params_ident: &[::zynk_runtime::ParamMeta] = &[#(#params_tokens),*];

                ::zynk_runtime::inventory::submit! {
                    ::zynk_runtime::EndpointMeta {
                        name: #name,
                        kind: ::zynk_runtime::EndpointKind::Ws,
                        module: Some(module_path!()),
                        doc: None,
                        params: &[],
                        returns: #returns,
                        channel_item: None,
                        file_param: None,
                        multi_file: false,
                        max_size: None,
                        allowed_types: &[],
                        server_events: #params_ident,
                        client_events: #params_ident,
                        handler_key: Some(::zynk_runtime::HandlerKey(concat!(module_path!(), "::", #name))),
                    }
                }
            }
        }
        EndpointSurface::Upload(config) => {
            let upload = upload_parts_from_signature(&function)?;
            let params_tokens = upload.params.iter().map(ParamTokens::to_tokens);
            let file_param = upload.file_param;
            let multi_file = upload.multi_file;
            let max_size = option_u64_tokens(config.max_size);
            let allowed_types = config.allowed_types.iter();

            quote! {
                #[allow(non_upper_case_globals)]
                const #params_ident: &[::zynk_runtime::ParamMeta] = &[#(#params_tokens),*];

                ::zynk_runtime::inventory::submit! {
                    ::zynk_runtime::EndpointMeta {
                        name: #name,
                        kind: ::zynk_runtime::EndpointKind::Upload,
                        module: Some(module_path!()),
                        doc: None,
                        params: #params_ident,
                        returns: #returns,
                        channel_item: None,
                        file_param: Some(#file_param),
                        multi_file: #multi_file,
                        max_size: #max_size,
                        allowed_types: &[#(#allowed_types),*],
                        server_events: &[],
                        client_events: &[],
                        handler_key: Some(::zynk_runtime::HandlerKey(concat!(module_path!(), "::", #name))),
                    }
                }
            }
        }
        EndpointSurface::StaticFile => {
            validate_static_file_return(&function.sig.output)?;
            let params = params_from_signature(&function)?;
            let params_tokens = params.iter().map(ParamTokens::to_tokens);

            quote! {
                #[allow(non_upper_case_globals)]
                const #params_ident: &[::zynk_runtime::ParamMeta] = &[#(#params_tokens),*];

                ::zynk_runtime::inventory::submit! {
                    ::zynk_runtime::EndpointMeta {
                        name: #name,
                        kind: ::zynk_runtime::EndpointKind::Static,
                        module: Some(module_path!()),
                        doc: None,
                        params: #params_ident,
                        returns: #returns,
                        channel_item: None,
                        file_param: None,
                        multi_file: false,
                        max_size: None,
                        allowed_types: &[],
                        server_events: &[],
                        client_events: &[],
                        handler_key: Some(::zynk_runtime::HandlerKey(concat!(module_path!(), "::", #name))),
                    }
                }
            }
        }
    };

    Ok(quote! {
        #function
        #registration
    })
}

fn parse_upload_config(attr: TokenStream) -> syn::Result<UploadConfig> {
    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let metas = parser.parse(attr)?;
    let mut config = UploadConfig::default();

    for meta in metas {
        match meta {
            Meta::NameValue(name_value) if name_value.path.is_ident("max_size") => {
                let Expr::Lit(expr_lit) = &name_value.value else {
                    return Err(syn::Error::new_spanned(
                        name_value.value,
                        "max_size must be a string literal like \"10MB\"",
                    ));
                };
                let Lit::Str(size) = &expr_lit.lit else {
                    return Err(syn::Error::new_spanned(
                        &expr_lit.lit,
                        "max_size must be a string literal like \"10MB\"",
                    ));
                };
                config.max_size = Some(parse_size_literal(size)?);
            }
            Meta::NameValue(name_value) if name_value.path.is_ident("allowed_types") => {
                let Expr::Array(array) = &name_value.value else {
                    return Err(syn::Error::new_spanned(
                        name_value.value,
                        "allowed_types must be an array of string literals",
                    ));
                };
                config.allowed_types = parse_allowed_types(array)?;
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported #[zynk::upload] option; expected max_size = \"10MB\" or allowed_types = [\"image/*\"]",
                ));
            }
        }
    }

    Ok(config)
}

fn parse_size_literal(lit: &syn::LitStr) -> syn::Result<u64> {
    let raw = lit.value();
    let compact: String = raw.chars().filter(|ch| !ch.is_whitespace()).collect();
    let digit_len = compact
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit() || *ch == '_')
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);

    if digit_len == 0 {
        return Err(syn::Error::new(
            lit.span(),
            "max_size must start with an integer byte count",
        ));
    }

    let number_text = compact[..digit_len].replace('_', "");
    let number = number_text.parse::<u64>().map_err(|error| {
        syn::Error::new(
            lit.span(),
            format!("max_size integer value is invalid: {error}"),
        )
    })?;
    let suffix = compact[digit_len..].to_ascii_uppercase();
    let factor = match suffix.as_str() {
        "" | "B" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024_u64.pow(2),
        "G" | "GB" | "GIB" => 1024_u64.pow(3),
        "T" | "TB" | "TIB" => 1024_u64.pow(4),
        _ => {
            return Err(syn::Error::new(
                lit.span(),
                "max_size unit must be one of B, KB, MB, GB, or TB",
            ));
        }
    };

    number.checked_mul(factor).ok_or_else(|| {
        syn::Error::new(
            lit.span(),
            "max_size overflows the supported u64 byte count",
        )
    })
}

fn parse_allowed_types(array: &syn::ExprArray) -> syn::Result<Vec<String>> {
    array
        .elems
        .iter()
        .map(|expr| {
            let Expr::Lit(ExprLit {
                lit: Lit::Str(value),
                ..
            }) = expr
            else {
                return Err(syn::Error::new_spanned(
                    expr,
                    "allowed_types entries must be string literals",
                ));
            };
            Ok(value.value())
        })
        .collect()
}

fn option_u64_tokens(value: Option<u64>) -> proc_macro2::TokenStream {
    match value {
        Some(value) => quote! { Some(#value) },
        None => quote! { None },
    }
}

fn reject_methods(function: &ItemFn) -> syn::Result<()> {
    for input in &function.sig.inputs {
        if matches!(input, FnArg::Receiver(_)) {
            return Err(syn::Error::new(
                input.span(),
                "#[zynk::command] and #[zynk::message] can only be applied to free functions; methods with self receivers are not supported",
            ));
        }
    }
    Ok(())
}

struct ParamTokens {
    source_name: String,
    wire_name: String,
    ty: proc_macro2::TokenStream,
    required: bool,
}

impl ParamTokens {
    fn to_tokens(&self) -> proc_macro2::TokenStream {
        let source_name = &self.source_name;
        let wire_name = &self.wire_name;
        let ty = &self.ty;
        let required = self.required;
        quote! {
            ::zynk_runtime::ParamMeta {
                source_name: #source_name,
                wire_name: #wire_name,
                ty: #ty,
                required: #required,
                default: None,
            }
        }
    }
}

fn params_from_signature(function: &ItemFn) -> syn::Result<Vec<ParamTokens>> {
    function.sig.inputs.iter().map(param_from_fn_arg).collect()
}

fn upload_parts_from_signature(function: &ItemFn) -> syn::Result<UploadParts> {
    let mut params = Vec::new();
    let mut file_param = None;

    for input in &function.sig.inputs {
        let (ident, ty) = typed_param_ident_and_type(input)?;
        if let Some(multi_file) = upload_file_type(ty) {
            if file_param.is_some() {
                return Err(syn::Error::new(
                    input.span(),
                    "#[zynk::upload] supports exactly one UploadFile or Vec<UploadFile> parameter",
                ));
            }
            file_param = Some((ident.to_string(), multi_file));
        } else {
            params.push(param_from_ident_and_type(ident, ty));
        }
    }

    let Some((file_param, multi_file)) = file_param else {
        return Err(syn::Error::new(
            function.sig.ident.span(),
            "#[zynk::upload] requires one UploadFile or Vec<UploadFile> parameter",
        ));
    };

    Ok(UploadParts {
        params,
        file_param,
        multi_file,
    })
}

fn param_from_fn_arg(input: &FnArg) -> syn::Result<ParamTokens> {
    let (ident, ty) = typed_param_ident_and_type(input)?;
    Ok(param_from_ident_and_type(ident, ty))
}

fn typed_param_ident_and_type(input: &FnArg) -> syn::Result<(&syn::Ident, &Type)> {
    match input {
        FnArg::Typed(argument) => {
            let ident = match argument.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => &pat_ident.ident,
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "Zynk endpoint parameters must be simple identifiers like `user_id: i64`",
                    ));
                }
            };
            Ok((ident, &argument.ty))
        }
        FnArg::Receiver(receiver) => Err(syn::Error::new(
            receiver.span(),
            "Zynk endpoint macros do not support self receivers",
        )),
    }
}

fn param_from_ident_and_type(ident: &syn::Ident, ty: &Type) -> ParamTokens {
    let lowered = lower_type(ty);
    ParamTokens {
        source_name: ident.to_string(),
        wire_name: to_camel_case(&ident.to_string()),
        ty: lowered.tokens,
        required: !lowered.optional,
    }
}

fn return_type_from_signature(output: &ReturnType) -> proc_macro2::TokenStream {
    match output {
        ReturnType::Default => quote! { ::zynk_runtime::TypeRefStatic::void() },
        ReturnType::Type(_, ty) => lower_type(ty).tokens,
    }
}

fn validate_static_file_return(output: &ReturnType) -> syn::Result<()> {
    let ReturnType::Type(_, ty) = output else {
        return Err(syn::Error::new_spanned(
            output,
            "#[zynk::static_file] functions must return StaticFile",
        ));
    };

    if is_named_type(ty, "StaticFile") {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "#[zynk::static_file] functions must return StaticFile",
        ))
    }
}

struct LoweredType {
    tokens: proc_macro2::TokenStream,
    optional: bool,
}

fn lower_type(ty: &Type) -> LoweredType {
    match ty {
        Type::Tuple(tuple) if tuple.elems.is_empty() => LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::void() },
            optional: false,
        },
        Type::Path(type_path) => lower_path_type(type_path),
        Type::Reference(reference) => lower_type(&reference.elem),
        _ => LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        },
    }
}

fn lower_path_type(type_path: &syn::TypePath) -> LoweredType {
    let Some(segment) = type_path.path.segments.last() else {
        return LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        };
    };

    let ident = segment.ident.to_string();
    match ident.as_str() {
        "String" | "str" => primitive("string"),
        "bool" => primitive("boolean"),
        "f32" | "f64" => primitive("number"),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
        | "usize" => primitive("number"),
        "Option" => lower_option(segment),
        "Vec" => lower_vec(segment),
        "Result" => lower_result(segment),
        "Value" => LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        },
        other => {
            let model_name = other.to_string();
            LoweredType {
                tokens: quote! { ::zynk_runtime::TypeRefStatic::model(#model_name) },
                optional: false,
            }
        }
    }
}

fn primitive(name: &'static str) -> LoweredType {
    LoweredType {
        tokens: quote! { ::zynk_runtime::TypeRefStatic::primitive(#name) },
        optional: false,
    }
}

fn lower_option(segment: &syn::PathSegment) -> LoweredType {
    let inner = first_generic_type(segment)
        .map(lower_type)
        .unwrap_or_else(|| LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        });
    let tokens = inner.tokens;
    LoweredType {
        tokens: quote! { #tokens.optional().nullable() },
        optional: true,
    }
}

fn lower_vec(segment: &syn::PathSegment) -> LoweredType {
    let inner = first_generic_type(segment)
        .map(lower_type)
        .unwrap_or_else(|| LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        });
    let inner_tokens = inner.tokens;
    LoweredType {
        tokens: quote! { ::zynk_runtime::TypeRefStatic::array(&[#inner_tokens]) },
        optional: false,
    }
}

fn lower_result(segment: &syn::PathSegment) -> LoweredType {
    first_generic_type(segment)
        .map(lower_type)
        .unwrap_or_else(|| LoweredType {
            tokens: quote! { ::zynk_runtime::TypeRefStatic::any() },
            optional: false,
        })
}

fn upload_file_type(ty: &Type) -> Option<bool> {
    match ty {
        Type::Reference(reference) => upload_file_type(&reference.elem),
        Type::Path(type_path) => {
            let segment = type_path.path.segments.last()?;
            if segment.ident == "UploadFile" {
                Some(false)
            } else if segment.ident == "Vec" {
                first_generic_type(segment)
                    .filter(|inner| is_named_type(inner, "UploadFile"))
                    .map(|_| true)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_named_type(ty: &Type, expected: &str) -> bool {
    match ty {
        Type::Reference(reference) => is_named_type(&reference.elem, expected),
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == expected),
        _ => false,
    }
}

fn first_generic_type(segment: &syn::PathSegment) -> Option<&Type> {
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };

    arguments.args.iter().find_map(|argument| match argument {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

fn to_camel_case(name: &str) -> String {
    let mut parts = name.split('_');
    let Some(first) = parts.next() else {
        return String::new();
    };

    let mut output = first.to_string();
    for part in parts {
        let mut chars = part.chars();
        if let Some(first_char) = chars.next() {
            output.extend(first_char.to_uppercase());
            output.push_str(chars.as_str());
        }
    }
    output
}
