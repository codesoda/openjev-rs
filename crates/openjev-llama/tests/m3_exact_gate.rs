#![cfg(feature = "integration")]

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use openjev_core::{Decision, Device, GpuLayersRequested};
use openjev_llama::{EngineHandle, EngineOptions, ModelCache, ModelRegistry};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReferenceEncoding {
    id: String,
    option_ids: Vec<String>,
    answer_token_ids: Vec<u32>,
    input_tokens: usize,
    prompt_sha256: String,
}

#[test]
fn qwen_authored144_exact_encoded_prompt_gate() {
    if std::env::var("OPENJEV_INTEGRATION").as_deref() != Ok("1") {
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let references_path =
        root.join("reference/semif-py/browser-ladder-qwen3-0.6b.predictions.jsonl");
    let mut references = HashMap::new();
    for line in BufReader::new(File::open(references_path).unwrap()).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let row: ReferenceEncoding = serde_json::from_str(&line).unwrap();
        let id = row.id.clone();
        assert!(references.insert(id, row).is_none());
    }
    assert_eq!(references.len(), 252);

    let fixture_path = root.join("reference/semif-py/benchmarks/data/authored144.jsonl");
    let mut decisions = Vec::new();
    let mut ids = HashSet::new();
    for line in BufReader::new(File::open(fixture_path).unwrap()).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let decision: Decision = serde_json::from_str(&line).unwrap();
        assert!(ids.insert(decision.id.clone()));
        assert!(references.contains_key(&decision.id));
        decisions.push(decision);
    }
    assert_eq!(decisions.len(), 144);

    // Every fixture/reference input is parsed and joined before the model is loaded.
    let registry = ModelRegistry::bundled().unwrap();
    let spec = registry.resolve("qwen3-0.6b").unwrap().clone();
    let cache = ModelCache::from_precedence(None).unwrap();
    let artifact = registry.path(&cache, "qwen3-0.6b").unwrap();
    let (device, gpu_layers) = if cfg!(feature = "metal") {
        (Device::Metal, GpuLayersRequested::All)
    } else if cfg!(feature = "cuda") {
        (Device::Cuda, GpuLayersRequested::All)
    } else {
        (Device::Cpu, GpuLayersRequested::Count(0))
    };
    let engine = EngineHandle::spawn(
        spec,
        artifact,
        EngineOptions {
            device,
            gpu_layers,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    for decision in decisions {
        let reference = &references[&decision.id];
        let encoded = engine.encode_direct(decision).unwrap();
        encoded
            .validate_reference(
                &reference.id,
                &reference.option_ids,
                &reference.prompt_sha256,
                reference.input_tokens,
                &reference.answer_token_ids,
            )
            .unwrap();
    }
    engine.shutdown().unwrap();
}
