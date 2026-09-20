#[test]
fn jev_http_schema_compiles_and_accepts_supported_request_response_and_error() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let schema: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("schemas/jev-http-v1.schema.json")).unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    for value in [
        serde_json::json!({
            "model":"jev-latest",
            "state":{"ticket":"duplicate charge"},
            "questions":{
                "route":{"type":"choice","criteria":{"billing":null,"support":"general"}},
                "review":{"type":"noul"},
                "urgency":{"type":"score","criteria":["low","high"]}
            }
        }),
        serde_json::json!({
            "model":"qwen3-0.6b",
            "answers":{
                "route":{"type":"choice","choice":"billing","confidence":0.75,"probabilities":{"billing":0.88,"support":0.13}},
                "review":{"type":"noul","noul":0.25},
                "urgency":{"type":"score","score":1.42,"confidence":0.31,"legend":{"0":"low","1":{"level":"medium"},"2":null},"probabilities":{"0":0.13,"1":0.33,"2":0.54}}
            },
            "usage":{"input_tokens":42,"output_tokens":0}
        }),
        serde_json::json!({"error_type":"validation_error","message":"invalid request"}),
    ] {
        assert!(validator.is_valid(&value), "schema rejected {value}");
    }
    assert!(!validator.is_valid(&serde_json::json!({
        "state":"s",
        "questions":{},
        "unknown":true
    })));
}
