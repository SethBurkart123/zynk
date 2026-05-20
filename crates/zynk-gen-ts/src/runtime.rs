//! Embedded TypeScript runtime helpers shipped with generated clients.

const INTERNAL_TS: &str = include_str!("runtime/_internal.ts");

/// Return the embedded `_internal.ts` runtime source.
pub fn internal_ts() -> &'static str {
    INTERNAL_TS
}

#[cfg(test)]
mod tests {
    #[test]
    fn embeds_internal_runtime_via_include_str() {
        let source = super::internal_ts();

        assert!(source.contains("export function initBridge"));
        assert!(source.contains("export async function request"));
        assert!(source.contains("export function createChannel"));
        assert!(source.contains("export function createUpload"));
    }
}
