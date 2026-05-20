//! Embedded Effect connector runtime shipped with generated clients.

const EFFECT_INTERNAL_TS: &str = include_str!("runtime/_effect_internal.ts");

/// Return the embedded `_effect_internal.ts` runtime source.
pub fn effect_internal_ts() -> &'static str {
    EFFECT_INTERNAL_TS
}

#[cfg(test)]
mod tests {
    #[test]
    fn embeds_effect_internal_runtime_via_include_str() {
        let source = super::effect_internal_ts();

        assert!(source.contains("export class ZynkNetworkError"));
        assert!(source.contains("export class ZynkHttpError"));
        assert!(source.contains("export const callCommand"));
        assert!(source.contains("export const callChannel"));
        assert!(source.contains("export const callUpload"));
        assert!(source.contains("export const buildStaticUrl"));
        assert!(source.contains("export const openWebSocket"));
        assert!(source.contains("export const runPromise"));
        assert!(source.contains("export const toAsyncIterable"));
        assert!(source.contains("ManagedRuntime"));
    }
}
