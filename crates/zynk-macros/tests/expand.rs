#[test]
fn zynk_macro_expand_fixtures_compile() {
    let test = trybuild::TestCases::new();
    test.pass("tests/expand/command_success.rs");
    test.pass("tests/expand/message_success.rs");
    test.compile_fail("tests/expand/command_rejects_self.rs");
}
