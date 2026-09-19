use std::{fs, path::Path};

use openjev_core::{Decision, PromptProfile, prepare_prompt};
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracles {
    cases: Vec<Oracle>,
}

#[derive(Deserialize)]
struct Oracle {
    decision: serde_json::Value,
    qwen3_and_qwen3_5_render: String,
    qwen3_and_qwen3_5_sha256: String,
    minicpm5_sha256: String,
}

#[test]
fn restricted_profile_oracle_matches() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/prompt-oracles.json");
    let oracles: Oracles = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    for oracle in oracles.cases {
        let decision: Decision = serde_json::from_value(oracle.decision).unwrap();
        for profile in [PromptProfile::Qwen3, PromptProfile::Qwen35] {
            let prompt = prepare_prompt(&decision, profile).unwrap();
            assert_eq!(prompt.text, oracle.qwen3_and_qwen3_5_render);
            assert_eq!(prompt.prompt_sha256, oracle.qwen3_and_qwen3_5_sha256);
        }
        let prompt = prepare_prompt(&decision, PromptProfile::MiniCpm5).unwrap();
        assert_eq!(
            prompt.text,
            format!("<s>{}", oracle.qwen3_and_qwen3_5_render)
        );
        assert_eq!(prompt.prompt_sha256, oracle.minicpm5_sha256);
    }
}
