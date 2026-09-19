use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use openjev_core::{Decision, PromptProfile, prepare_prompt};
use serde_json::Value;

fn reference(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/semif-py")
        .join(path)
}

fn jsonl(path: &Path) -> Vec<String> {
    BufReader::new(File::open(path).unwrap())
        .lines()
        .map(|line| line.unwrap())
        .filter(|line| !line.trim().is_empty())
        .collect()
}

fn golden_hashes() -> HashMap<String, String> {
    let rows = jsonl(&reference("browser-ladder-qwen3-0.6b.predictions.jsonl"));
    let mut result = HashMap::new();
    for line in rows {
        let value: Value = serde_json::from_str(&line).unwrap();
        let id = value["id"].as_str().unwrap().to_owned();
        let hash = value["prompt_sha256"].as_str().unwrap().to_owned();
        assert!(result.insert(id, hash).is_none());
    }
    assert_eq!(result.len(), 252);
    result
}

fn verify_fixture(path: &str, expected_count: usize) {
    let goldens = golden_hashes();
    let mut ids = HashSet::new();
    let rows = jsonl(&reference(path));
    assert_eq!(rows.len(), expected_count);
    for line in rows {
        let decision = Decision::from_json_str(&line).unwrap();
        assert!(ids.insert(decision.id.clone()));
        let prompt = prepare_prompt(&decision, PromptProfile::Qwen3).unwrap();
        assert_eq!(
            prompt.prompt_sha256, goldens[&decision.id],
            "prompt mismatch for {}",
            decision.id
        );
    }
}

#[test]
fn authored_144_prompt_hashes_match_by_id() {
    verify_fixture("benchmarks/data/authored144.jsonl", 144);
}

#[test]
fn perturbation_108_prompt_hashes_match_by_id() {
    verify_fixture("benchmarks/data/perturbations108.jsonl", 108);
}
