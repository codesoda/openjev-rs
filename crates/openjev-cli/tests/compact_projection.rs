use openjev_cli::output::{write_json, write_readout};
use openjev_core::{
    Device, ErrorRecord, ExecutionMetadata, ExecutionMode, GpuLayersRequested, GpuLayersStatus,
    Integrity, ModelMetadata, Primitive, PromptProfile, Readout, TemplateMetadataStatus,
};

fn readout(primitive: Primitive) -> Readout {
    Readout {
        schema: "openjev-readout-v1".to_owned(),
        id: "decision-1".to_owned(),
        primitive,
        choice: "b".to_owned(),
        choice_index: 1,
        option_ids: vec!["a".to_owned(), "b".to_owned()],
        probabilities: vec![0.25, 0.75],
        option_logits: vec![-1.0, 1.0],
        answer_token_ids: vec![32, 33],
        allowed_token_mass: 0.5,
        full_vocab_argmax_id: 33,
        full_vocab_log_normalizer: 2.0,
        input_tokens: 10,
        forward_seconds: Some(0.1),
        total_seconds: Some(0.2),
        prompt_sha256: "0".repeat(64),
        prompt_version: "direct-options-v1".to_owned(),
        model: ModelMetadata {
            id: "test".to_owned(),
            source: "local".to_owned(),
            revision: "sha256:test".to_owned(),
            file: "test.gguf".to_owned(),
            quant: "test".to_owned(),
            backend: "test".to_owned(),
            artifact_sha256: "1".repeat(64),
            integrity: Integrity::LocalUnverified,
            dtype: "test".to_owned(),
            native_reference: None,
            template_profile: PromptProfile::Qwen3,
            template_sha256: None,
            template_override: true,
            template_status: TemplateMetadataStatus::OverrideUnverified,
            template_equivalence_evidence: None,
            serving_config: None,
            adapter: None,
            adapter_sha256: None,
            adapter_revision: None,
            torch_version: None,
            transformers_version: None,
        },
        readout: openjev_core::DIRECT_READOUT.to_owned(),
        probability_status: openjev_core::PROBABILITY_STATUS.to_owned(),
        limitations: openjev_core::standard_limitations(),
        execution: ExecutionMetadata {
            requested_mode: ExecutionMode::Direct,
            effective_mode: ExecutionMode::Direct,
            fallback_reason: None,
            device: Device::Cpu,
            device_name: "test".to_owned(),
            gpu_layers_requested: GpuLayersRequested::Count(0),
            gpu_layers_actual: Some(0),
            gpu_layers_status: GpuLayersStatus::KnownDisabled,
            threads: 1,
            n_ctx_requested: None,
            n_ctx_actual: 128,
            max_tokens: 128,
            n_batch: 128,
            n_ubatch: 128,
            n_seq_max: 1,
            kv_unified: true,
            waves: 1,
            probe_id: None,
            run_id: "test".to_owned(),
            group_id: None,
        },
        confidence: None,
        confidence_status: None,
        p_yes: None,
        level_values: None,
        expected_value: None,
        argmax_level: None,
        cache_hit: Some(false),
        prefix_tokens: None,
        prefix_sha256: None,
        prefill_seconds: None,
        copy_seconds: None,
        suffix_forward_seconds: None,
        shared_timing: None,
        postprocess: None,
    }
}

fn compact_value(row: &Readout, pretty: bool) -> serde_json::Value {
    let mut bytes = Vec::new();
    write_readout(&mut bytes, row, pretty, true).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn compact_projection_is_typed_and_excludes_verbose_fields() {
    let row = readout(Primitive::Choice);
    let value = compact_value(&row, false);
    assert_eq!(value["schema"], "openjev-compact-v1");
    assert_eq!(value["id"], "decision-1");
    assert_eq!(value["choice"], "b");
    assert_eq!(value["option_ids"], serde_json::json!(["a", "b"]));
    assert_eq!(value["probabilities"], serde_json::json!([0.25, 0.75]));
    assert_eq!(
        value["probability_status"],
        openjev_core::PROBABILITY_STATUS
    );
    for omitted in [
        "primitive",
        "option_logits",
        "answer_token_ids",
        "model",
        "prompt_sha256",
        "forward_seconds",
        "total_seconds",
        "execution",
    ] {
        assert!(value.get(omitted).is_none(), "unexpected {omitted}");
    }
}

#[test]
fn compact_projection_covers_noul_score_confidence_and_fallback() {
    let mut noul = readout(Primitive::Noul);
    noul.option_ids = vec!["yes".to_owned(), "no".to_owned()];
    noul.choice = "no".to_owned();
    noul.p_yes = Some(0.25);
    noul.confidence = Some(0.5);
    noul.confidence_status = Some("normalized margin; uncalibrated".to_owned());
    let value = compact_value(&noul, true);
    assert_eq!(value["p_yes"], 0.25);
    assert_eq!(value["confidence"], 0.5);
    assert_eq!(
        value["confidence_status"],
        "normalized margin; uncalibrated"
    );

    let mut score = readout(Primitive::Score);
    score.level_values = Some(vec![-2.0, 6.0]);
    score.expected_value = Some(4.0);
    score.argmax_level = Some("b".to_owned());
    score.execution.requested_mode = ExecutionMode::Shared;
    score.execution.effective_mode = ExecutionMode::Serial;
    score.execution.fallback_reason = Some("probe did not pass".to_owned());
    let value = compact_value(&score, false);
    assert_eq!(value["expected_value"], 4.0);
    assert_eq!(value["argmax_level"], "b");
    assert_eq!(value["level_values"], serde_json::json!([-2.0, 6.0]));
    assert_eq!(value["execution"]["requested_mode"], "shared");
    assert_eq!(value["execution"]["effective_mode"], "serial");
    assert_eq!(value["execution"]["fallback_reason"], "probe did not pass");
}

#[test]
fn error_records_are_not_compact_projected_and_schema_validates() {
    let mut bytes = Vec::new();
    write_json(&mut bytes, &ErrorRecord::new("decode", "failed"), false).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["schema"], "openjev-error-v1");

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let schema: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("schemas/compact-v1.schema.json")).unwrap(),
    )
    .unwrap();
    let projected = compact_value(&readout(Primitive::Choice), false);
    jsonschema::validator_for(&schema)
        .unwrap()
        .validate(&projected)
        .unwrap();
}
