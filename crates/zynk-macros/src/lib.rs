//! Proc macros for registering Rust Zynk endpoints.
//!
//! The attribute macros in this crate preserve the user's function unchanged
//! and add link-time `EndpointMeta` registrations consumed by server bindings.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, spanned::Spanned, FnArg, GenericArgument, ItemFn, PathArguments, ReturnType,
    Type,
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

enum EndpointSurface {
    Command,
    Message,
}

fn expand_endpoint(
    function: ItemFn,
    surface: EndpointSurface,
) -> syn::Result<proc_macro2::TokenStream> {
    reject_methods(&function)?;

    let name = function.sig.ident.to_string();
    let params = params_from_signature(&function)?;
    let returns = return_type_from_signature(&function.sig.output);
    let params_ident = format_ident!(
        "__ZYNK_{}_PARAMS",
        function.sig.ident.to_string().to_uppercase()
    );
    let params_tokens = params.iter().map(ParamTokens::to_tokens);

    let registration = match surface {
        EndpointSurface::Command => quote! {
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
        },
        EndpointSurface::Message => quote! {
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
        },
    };

    Ok(quote! {
        #function
        #registration
    })
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
    function
        .sig
        .inputs
        .iter()
        .map(|input| match input {
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
                let lowered = lower_type(&argument.ty);
                Ok(ParamTokens {
                    source_name: ident.to_string(),
                    wire_name: to_camel_case(&ident.to_string()),
                    ty: lowered.tokens,
                    required: !lowered.optional,
                })
            }
            FnArg::Receiver(receiver) => Err(syn::Error::new(
                receiver.span(),
                "Zynk endpoint macros do not support self receivers",
            )),
        })
        .collect()
}

fn return_type_from_signature(output: &ReturnType) -> proc_macro2::TokenStream {
    match output {
        ReturnType::Default => quote! { ::zynk_runtime::TypeRefStatic::void() },
        ReturnType::Type(_, ty) => lower_type(ty).tokens,
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
