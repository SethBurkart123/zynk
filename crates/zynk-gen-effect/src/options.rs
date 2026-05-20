//! Configuration options for Effect generator endpoint surfaces.

use std::collections::BTreeSet;
use std::fmt;

use zynk_schema::{EndpointKind, Value};

/// Generated API surface for an endpoint kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Return raw `Effect.Effect` / `Stream.Stream` values.
    Effect,
    /// Return Promise / AsyncIterable convenience wrappers.
    Promise,
}

impl Default for Surface {
    fn default() -> Self {
        Self::Effect
    }
}

impl Surface {
    /// Parse the Python-compatible surface string.
    pub fn parse(value: &str) -> Result<Self, EffectGeneratorOptionsError> {
        match value {
            "effect" => Ok(Self::Effect),
            "promise" => Ok(Self::Promise),
            other => Err(EffectGeneratorOptionsError::InvalidSurface(
                other.to_string(),
            )),
        }
    }
}

/// Effect generator options matching the Python dataclass surface.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EffectGeneratorOptions {
    /// Default surface used when a per-kind override is absent.
    pub default: Surface,
    /// Command/RPC surface override.
    pub commands: Option<Surface>,
    /// Upload surface override.
    pub uploads: Option<Surface>,
    /// Static endpoint surface override.
    pub statics: Option<Surface>,
    /// Channel surface override.
    pub channels: Option<Surface>,
    /// WebSocket override accepted for Python parity, but not consulted.
    pub websockets: Option<Surface>,
}

impl EffectGeneratorOptions {
    /// Resolve the configured surface for an endpoint kind.
    pub fn resolve(&self, kind: EndpointKind) -> Surface {
        match kind {
            EndpointKind::Rpc => self.commands,
            EndpointKind::Channel => self.channels,
            EndpointKind::Upload => self.uploads,
            EndpointKind::Static => self.statics,
            EndpointKind::Ws => self.websockets,
        }
        .unwrap_or(self.default)
    }

    /// Build options from the JSON object used by `GenerationContext::options`.
    pub fn from_mapping(raw: &Value) -> Result<Self, EffectGeneratorOptionsError> {
        let Some(object) = raw.as_object() else {
            return if raw.is_null() {
                Ok(Self::default())
            } else {
                Err(EffectGeneratorOptionsError::ExpectedObject)
            };
        };

        let valid: BTreeSet<&str> = [
            "default",
            "commands",
            "uploads",
            "statics",
            "channels",
            "websockets",
        ]
        .into_iter()
        .collect();
        let unknown = object
            .keys()
            .filter(|key| !valid.contains(key.as_str()))
            .cloned()
            .collect::<Vec<_>>();

        if !unknown.is_empty() {
            return Err(EffectGeneratorOptionsError::UnknownKeys(unknown));
        }

        Ok(Self {
            default: parse_required_surface(object.iter(), "default")?.unwrap_or_default(),
            commands: parse_optional_surface(object.iter(), "commands")?,
            uploads: parse_optional_surface(object.iter(), "uploads")?,
            statics: parse_optional_surface(object.iter(), "statics")?,
            channels: parse_optional_surface(object.iter(), "channels")?,
            websockets: parse_optional_surface(object.iter(), "websockets")?,
        })
    }
}

fn parse_required_surface<'a, I>(
    object: I,
    key: &str,
) -> Result<Option<Surface>, EffectGeneratorOptionsError>
where
    I: IntoIterator<Item = (&'a String, &'a Value)>,
{
    object
        .into_iter()
        .find(|(candidate, _)| candidate.as_str() == key)
        .map(|(_, value)| parse_surface_value(key, value))
        .transpose()
}

fn parse_optional_surface<'a, I>(
    object: I,
    key: &str,
) -> Result<Option<Surface>, EffectGeneratorOptionsError>
where
    I: IntoIterator<Item = (&'a String, &'a Value)>,
{
    match object
        .into_iter()
        .find(|(candidate, _)| candidate.as_str() == key)
        .map(|(_, value)| value)
    {
        None | Some(Value::Null) => Ok(None),
        Some(value) => parse_surface_value(key, value).map(Some),
    }
}

fn parse_surface_value(key: &str, value: &Value) -> Result<Surface, EffectGeneratorOptionsError> {
    let Some(surface) = value.as_str() else {
        return Err(EffectGeneratorOptionsError::InvalidSurfaceValue(
            key.to_string(),
        ));
    };
    Surface::parse(surface)
}

/// Error returned when generator options cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectGeneratorOptionsError {
    /// Options must be a JSON object or `null`.
    ExpectedObject,
    /// One or more unknown option keys were supplied.
    UnknownKeys(Vec<String>),
    /// Surface string was not `effect` or `promise`.
    InvalidSurface(String),
    /// Surface value had the wrong JSON type.
    InvalidSurfaceValue(String),
}

impl fmt::Display for EffectGeneratorOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedObject => write!(f, "effect generator options must be an object"),
            Self::UnknownKeys(keys) => {
                write!(f, "Unknown effect generator options: {}", keys.join(", "))
            }
            Self::InvalidSurface(surface) => {
                write!(f, "invalid effect generator surface: {surface}")
            }
            Self::InvalidSurfaceValue(key) => {
                write!(f, "effect generator option '{key}' must be a string")
            }
        }
    }
}

impl std::error::Error for EffectGeneratorOptionsError {}

#[cfg(test)]
mod tests {
    use zynk_schema::{EndpointKind, Value};

    use super::{EffectGeneratorOptions, Surface};

    #[test]
    fn resolve_uses_per_kind_override_or_default() {
        let opts = EffectGeneratorOptions {
            default: Surface::Effect,
            commands: Some(Surface::Promise),
            uploads: None,
            statics: Some(Surface::Promise),
            channels: None,
            websockets: Some(Surface::Promise),
        };

        assert_eq!(opts.resolve(EndpointKind::Rpc), Surface::Promise);
        assert_eq!(opts.resolve(EndpointKind::Upload), Surface::Effect);
        assert_eq!(opts.resolve(EndpointKind::Static), Surface::Promise);
        assert_eq!(opts.resolve(EndpointKind::Channel), Surface::Effect);
        assert_eq!(opts.resolve(EndpointKind::Ws), Surface::Promise);
    }

    #[test]
    fn from_mapping_rejects_unknown_keys_sorted_alphabetically() {
        let raw = Value::Object(
            [
                ("zeta".to_string(), Value::String("effect".to_string())),
                ("alpha".to_string(), Value::String("effect".to_string())),
            ]
            .into_iter()
            .collect(),
        );

        let error = EffectGeneratorOptions::from_mapping(&raw).expect_err("unknown keys fail");

        assert_eq!(
            error.to_string(),
            "Unknown effect generator options: alpha, zeta"
        );
    }

    #[test]
    fn from_mapping_parses_json_surfaces_and_null_overrides() {
        let raw = Value::Object(
            [
                ("default".to_string(), Value::String("promise".to_string())),
                ("commands".to_string(), Value::String("effect".to_string())),
                ("channels".to_string(), Value::Null),
                (
                    "websockets".to_string(),
                    Value::String("promise".to_string()),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let opts = EffectGeneratorOptions::from_mapping(&raw).expect("valid options");

        assert_eq!(opts.default, Surface::Promise);
        assert_eq!(opts.commands, Some(Surface::Effect));
        assert_eq!(opts.channels, None);
        assert_eq!(opts.resolve(EndpointKind::Upload), Surface::Promise);
        assert_eq!(opts.websockets, Some(Surface::Promise));
    }
}
