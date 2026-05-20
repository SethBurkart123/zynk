/// Convert a snake_case backend name into the camelCase wire/client name.
pub fn to_camel_case(name: &str) -> String {
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

/// Convert a snake_case backend name into PascalCase.
pub fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first_char) => {
                    let mut output = String::new();
                    output.extend(first_char.to_uppercase());
                    output.push_str(chars.as_str());
                    output
                }
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{to_camel_case, to_pascal_case};

    #[test]
    fn converts_names_like_python_helpers() {
        assert_eq!(to_camel_case("get_user"), "getUser");
        assert_eq!(to_camel_case("user"), "user");
        assert_eq!(to_pascal_case("get_user"), "GetUser");
    }
}
