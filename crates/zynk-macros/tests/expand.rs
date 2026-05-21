#[test]
fn zynk_macro_expand_fixtures_compile() {
    let test = trybuild::TestCases::new();
    test.pass("tests/expand/command_success.rs");
    test.pass("tests/expand/message_success.rs");
    test.pass("tests/expand/upload_success.rs");
    test.pass("tests/expand/static_success.rs");
    test.compile_fail("tests/expand/command_rejects_self.rs");
    test.compile_fail("tests/expand/upload_missing_file.rs");
    test.compile_fail("tests/expand/upload_sync_fn.rs");
    test.compile_fail("tests/expand/upload_multiple_files.rs");
    test.compile_fail("tests/expand/static_bad_return.rs");
    test.compile_fail("tests/expand/static_result_return.rs");
}
