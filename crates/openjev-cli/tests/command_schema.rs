#[test]
fn command_schema_export_is_valid_json_with_all_m6_tags() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(root.join("schemas/commands-v1.schema.json")).unwrap();
    let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    let serialized = schema.to_string();
    for filename in ["eval-v1.schema.json", "bench-v1.schema.json"] {
        let schema: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("schemas").join(filename)).unwrap(),
        )
        .unwrap();
        jsonschema::validator_for(&schema).unwrap();
    }
    for tag in [
        "openjev-models-v1",
        "openjev-model-path-v1",
        "openjev-help-v1",
        "openjev-version-v1",
        "openjev-write-summary-v1",
        "openjev-eval-command-v1",
        "openjev-bench-v1",
    ] {
        assert!(serialized.contains(tag), "missing schema tag {tag}");
    }
}
