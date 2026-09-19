use std::{fs, path::Path};

use openjev_core::{
    CONFIDENCE_STATUS, DIRECT_READOUT, Device, ExecutionMetadata, ExecutionMode,
    GpuLayersRequested, Integrity, ModelMetadata, PROBABILITY_STATUS, Postprocess, Primitive,
    PromptProfile, RawSample, Readout, SharedTiming, normalized_margin, read_logits,
    standard_limitations,
};

type Mutation = (fn(&mut Readout), bool);

fn execution() -> ExecutionMetadata {
    ExecutionMetadata {
        requested_mode: ExecutionMode::Direct,
        effective_mode: ExecutionMode::Direct,
        fallback_reason: None,
        device: Device::Cpu,
        device_name: "test-cpu".to_owned(),
        gpu_layers_requested: GpuLayersRequested::Count(0),
        gpu_layers_actual: 0,
        threads: 1,
        n_ctx_requested: None,
        n_ctx_actual: 4096,
        max_tokens: 4096,
        n_batch: 512,
        n_ubatch: 512,
        n_seq_max: 1,
        kv_unified: true,
        waves: 1,
        probe_id: None,
        run_id: "test-run".to_owned(),
        group_id: None,
    }
}

fn model() -> ModelMetadata {
    ModelMetadata {
        id: "test-model".to_owned(),
        source: "local".to_owned(),
        revision: format!("sha256:{}", "1".repeat(64)),
        file: "/tmp/test.gguf".to_owned(),
        quant: "Q8_0".to_owned(),
        backend: "test-backend".to_owned(),
        artifact_sha256: "1".repeat(64),
        integrity: Integrity::LocalUnverified,
        dtype: "Q8_0".to_owned(),
        native_reference: None,
        template_profile: PromptProfile::Qwen3,
        template_sha256: None,
        template_override: false,
        serving_config: Some("llama-direct-v1".to_owned()),
        adapter: None,
        adapter_sha256: None,
        adapter_revision: None,
        torch_version: None,
        transformers_version: None,
    }
}

fn valid_readout() -> Readout {
    let numeric = read_logits(&[0.0, 1.0e-20], &[0.0, 1.0e-20]).unwrap();
    let option_ids = vec!["a".to_owned(), "b".to_owned()];
    Readout {
        schema: "openjev-readout-v1".to_owned(),
        id: "decision".to_owned(),
        primitive: Primitive::Choice,
        choice: option_ids[numeric.choice_index].clone(),
        choice_index: numeric.choice_index,
        option_ids,
        probabilities: numeric.probabilities.clone(),
        option_logits: numeric.option_logits,
        answer_token_ids: vec![10, 11],
        allowed_token_mass: numeric.allowed_token_mass,
        full_vocab_argmax_id: numeric.full_vocab_argmax_id,
        full_vocab_log_normalizer: numeric.full_vocab_log_normalizer,
        input_tokens: 10,
        forward_seconds: Some(0.01),
        total_seconds: Some(0.02),
        prompt_sha256: "2".repeat(64),
        prompt_version: "direct-options-v1".to_owned(),
        model: model(),
        readout: DIRECT_READOUT.to_owned(),
        probability_status: PROBABILITY_STATUS.to_owned(),
        limitations: standard_limitations(),
        execution: execution(),
        confidence: Some(normalized_margin(&numeric.probabilities).unwrap()),
        confidence_status: Some(CONFIDENCE_STATUS.to_owned()),
        p_yes: None,
        level_values: None,
        expected_value: None,
        argmax_level: None,
        cache_hit: None,
        prefix_tokens: None,
        prefix_sha256: None,
        prefill_seconds: None,
        copy_seconds: None,
        suffix_forward_seconds: None,
        shared_timing: None,
        postprocess: None,
    }
}

#[test]
fn real_valid_readouts_serialize_and_pass_exported_schema() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/readout-v1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();

    let choice = valid_readout();
    choice.validate().unwrap();
    let choice_json = serde_json::to_value(&choice).unwrap();
    assert!(validator.is_valid(&choice_json));

    let mut noul = choice.clone();
    noul.primitive = Primitive::Noul;
    noul.option_ids = vec!["yes".to_owned(), "no".to_owned()];
    noul.choice = "yes".to_owned();
    noul.p_yes = Some(noul.probabilities[0]);
    noul.validate().unwrap();
    assert!(validator.is_valid(&serde_json::to_value(&noul).unwrap()));

    let mut score = choice;
    score.primitive = Primitive::Score;
    score.option_ids = vec!["low".to_owned(), "high".to_owned()];
    score.choice = "low".to_owned();
    score.level_values = Some(vec![-1.0, 3.0]);
    score.expected_value = Some(1.0);
    score.argmax_level = Some("low".to_owned());
    score.validate().unwrap();
    assert!(validator.is_valid(&serde_json::to_value(&score).unwrap()));
}

#[test]
fn rejects_contract_and_primitive_mutations() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/readout-v1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mutations: [Mutation; 7] = [
        (|value| value.input_tokens = 0, true),
        (|value| value.execution.threads = 0, true),
        (|value| value.prompt_sha256 = "bad".to_owned(), true),
        (|value| value.prompt_version = "wrong".to_owned(), true),
        (
            |value| value.probability_status = "confidence".to_owned(),
            true,
        ),
        (|value| value.confidence = Some(0.9), false),
        (|value| value.model.artifact_sha256 = "ABC".repeat(21), true),
    ];
    for (mutate, schema_invalid) in mutations {
        let mut value = valid_readout();
        mutate(&mut value);
        assert!(value.validate().is_err());
        if schema_invalid {
            assert!(!validator.is_valid(&serde_json::to_value(&value).unwrap()));
        }
    }

    let mut bad_noul = valid_readout();
    bad_noul.primitive = Primitive::Noul;
    bad_noul.p_yes = Some(bad_noul.probabilities[0]);
    assert!(bad_noul.validate().is_err());

    let mut bad_score = valid_readout();
    bad_score.primitive = Primitive::Score;
    bad_score.level_values = Some(vec![0.0, 1.0]);
    bad_score.expected_value = Some(0.5);
    bad_score.argmax_level = Some("a".to_owned());
    bad_score.p_yes = Some(0.5);
    assert!(bad_score.validate().is_err());
}

#[test]
fn validates_nested_timing_and_postprocess_provenance() {
    let mut timing = valid_readout();
    timing.shared_timing = Some(SharedTiming {
        total_seconds: -1.0,
        encode_seconds: 0.0,
        prefix_tokens: 1,
        prefill_seconds: 0.0,
        replicate_seconds: 0.0,
        suffix_forward_seconds: 0.0,
        batch_size: 1,
        true_suffix_tokens: 1,
        padded_suffix_tokens: 1,
    });
    assert!(timing.validate().is_err());

    let mut postprocessed = valid_readout();
    let sample = RawSample {
        permutation: vec![0, 1],
        option_ids: postprocessed.option_ids.clone(),
        answer_token_ids: postprocessed.answer_token_ids.clone(),
        option_logits: postprocessed.option_logits.clone(),
        probabilities: postprocessed.probabilities.clone(),
        prompt_sha256: postprocessed.prompt_sha256.clone(),
        prompt_version: postprocessed.prompt_version.clone(),
        input_tokens: postprocessed.input_tokens,
        allowed_token_mass: postprocessed.allowed_token_mass,
        full_vocab_argmax_id: postprocessed.full_vocab_argmax_id,
        full_vocab_log_normalizer: postprocessed.full_vocab_log_normalizer,
        forward_seconds: postprocessed.forward_seconds,
        total_seconds: postprocessed.total_seconds,
        execution: postprocessed.execution.clone(),
        shared_timing: None,
    };
    postprocessed.postprocess = Some(Postprocess {
        version: "openjev-postprocess-v1".to_owned(),
        temperature: 1.0,
        calibration_id: None,
        permutation_count: 1,
        seed: 0,
        permutation_algorithm: "sha256-factoradic-v1".to_owned(),
        aggregation: "mean-id-aligned-probabilities".to_owned(),
        raw_fields_reference: 0,
        base_probabilities: postprocessed.probabilities.clone(),
        samples: vec![sample],
    });
    postprocessed.validate().unwrap();

    postprocessed
        .postprocess
        .as_mut()
        .unwrap()
        .permutation_count = 0;
    assert!(postprocessed.validate().is_err());
}
