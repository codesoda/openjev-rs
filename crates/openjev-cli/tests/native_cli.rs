#![cfg(feature = "integration")]

use std::{
    fs,
    io::Write as _,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

use openjev_core::{Device, GpuLayersRequested, PromptProfile};
use openjev_llama::{
    ModelCache, ModelRegistry, NATIVE_PIN, PROBE_SUITE_VERSION, ProbeCaseResult, ProbeCaseStatus,
    ProbeConfiguration, ProbeMode, ProbeReceipt, load_passing_receipt, write_receipt,
};

const MODEL: &str = "qwen3-0.6b";
const SHA256: &str = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";

fn enabled() -> bool {
    std::env::var("OPENJEV_INTEGRATION").as_deref() == Ok("1")
}

fn invoke(args: &[&str], stdin: Option<&[u8]>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_openjev"));
    command
        .args(["--offline", "--model", MODEL])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    let mut child = command.spawn().unwrap();
    if let Some(bytes) = stdin {
        child.stdin.as_mut().unwrap().write_all(bytes).unwrap();
        drop(child.stdin.take());
    }
    child.wait_with_output().unwrap()
}

fn rows(output: &Output) -> Vec<serde_json::Value> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn assert_success(output: &Output) -> Vec<serde_json::Value> {
    assert!(output.status.success(), "{output:?}");
    let parsed = rows(output);
    assert!(!parsed.is_empty());
    assert!(parsed.iter().all(|row| row.is_object()));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("openjev-readout-v1"));
    parsed
}

#[test]
fn probe_child_crash_durably_revokes_a_preexisting_synthetic_pass() {
    if !enabled() {
        return;
    }

    let registry = ModelRegistry::bundled().unwrap();
    let user_cache = ModelCache::from_precedence(None).unwrap();
    let model_argument = user_cache.model_path(registry.resolve(MODEL).unwrap());
    let model_path =
        fs::canonicalize(&model_argument).expect("the opt-in native gate requires cached Qwen");
    let isolated_root =
        std::env::temp_dir().join(format!("openjev-m5-probe-crash-{}", std::process::id()));
    let _ = fs::remove_dir_all(&isolated_root);
    let isolated_cache = ModelCache::new(isolated_root.clone());
    let device = if cfg!(all(target_os = "macos", feature = "metal")) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let configuration = ProbeConfiguration {
        artifact_sha256: SHA256.to_owned(),
        native_pin: NATIVE_PIN.to_owned(),
        probe_suite_version: PROBE_SUITE_VERSION.to_owned(),
        device,
        gpu_layers_requested: if device == Device::Cpu {
            GpuLayersRequested::Count(0)
        } else {
            GpuLayersRequested::All
        },
        offload_kqv: device != Device::Cpu,
        op_offload: device != Device::Cpu,
        threads: std::thread::available_parallelism()
            .map(|value| u32::try_from(value.get()).unwrap())
            .unwrap_or(1),
        n_ctx: None,
        max_tokens: 4096,
        max_context_tokens: 32_768,
        n_batch: 512,
        n_ubatch: 512,
        n_seq_max: 32,
        kv_unified: true,
        profile: PromptProfile::Qwen3,
    };
    let case_rows = [1, 2, 21, 2, configuration.n_seq_max + 1];
    let case_ids = [
        "binary-short-1-branch",
        "three-way-ragged-2-branches-multichunk",
        "sixteen-way-long-state-21-branches",
        "changed-state-isolation",
        "repeated-copy-clear-cycles",
    ];
    // State-machine fixture only in an isolated temporary cache. This synthetic
    // pass is test data and is not native parity evidence or a user-cache receipt.
    let receipt = ProbeReceipt::new(
        model_path.display().to_string(),
        ProbeMode::Shared,
        configuration.clone(),
        case_ids
            .into_iter()
            .zip(case_rows)
            .map(|(id, rows)| ProbeCaseResult {
                id: id.to_owned(),
                status: ProbeCaseStatus::Passed,
                rows,
                max_abs_slot_logit: Some(0.0),
                max_probability_delta: Some(0.0),
                same_first_argmax: Some(true),
                detail: None,
            })
            .collect(),
        None,
    )
    .unwrap();
    write_receipt(&isolated_cache, &receipt).unwrap();
    load_passing_receipt(
        &isolated_cache,
        &model_path.display().to_string(),
        ProbeMode::Shared,
        &configuration,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_openjev"))
        .args([
            "--offline",
            "--cache-dir",
            isolated_root.to_str().unwrap(),
            "--model-sha256",
            SHA256,
            "--template-profile",
            "qwen3",
            "models",
            "probe",
            model_argument.to_str().unwrap(),
            "--mode",
            "shared",
        ])
        .env("OPENJEV_INTERNAL_PROBE_CRASH", "1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "openjev-probe-report-v1");
    assert_eq!(report["enabled"], false);
    assert!(report["receipt"].is_null());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    assert!(
        load_passing_receipt(
            &isolated_cache,
            &model_path.display().to_string(),
            ProbeMode::Shared,
            &configuration,
        )
        .is_err(),
        "a crashed reprobe must leave the preexisting pass ineligible"
    );
    fs::remove_dir_all(isolated_root).unwrap();
}

#[test]
fn cached_qwen_exercises_m4_native_cli_surfaces_and_json_streams() {
    if !enabled() {
        return;
    }

    let decide = assert_success(&invoke(
        &[
            "--confidence",
            "decide",
            "--state",
            "state",
            "--question",
            "Choose",
            "--option",
            "Alpha",
            "--option",
            "Beta",
        ],
        None,
    ));
    assert_eq!(decide[0]["schema"], "openjev-readout-v1");
    assert_eq!(decide[0]["cache_hit"], false);
    assert_eq!(decide[0]["execution"]["requested_mode"], "direct");
    assert_eq!(
        decide[0]["confidence_status"],
        "normalized margin; uncalibrated"
    );

    let noul = assert_success(&invoke(
        &["noul", "--question", "Is this acceptable?"],
        Some(b"state from stdin"),
    ));
    assert_eq!(noul[0]["primitive"], "noul");
    assert!(noul[0]["p_yes"].is_number());

    let score = assert_success(&invoke(
        &[
            "score",
            "--state-json",
            "{\"ordered\":1}",
            "--question",
            "Urgency",
            "--level",
            "low",
            "--level",
            "high",
            "--level-value",
            "-2.5",
            "--level-value",
            "7.5",
        ],
        None,
    ));
    assert_eq!(score[0]["primitive"], "score");
    assert_eq!(score[0]["level_values"], serde_json::json!([-2.5, 7.5]));
    assert!(score[0]["expected_value"].is_number());

    let decision = br#"{"id":"ask-1","state":"ask state","question":"Choose","options":[{"id":"a","description":"Alpha"},{"id":"b","description":"Beta"}]}"#;
    let ask = assert_success(&invoke(&["ask"], Some(decision)));
    assert_eq!(ask[0]["id"], "ask-1");

    let run_input = concat!(
        "{\"id\":\"run-1\",\"state\":\"s\",\"question\":\"Q1\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
        "{\"id\":\"run-2\",\"state\":\"s\",\"question\":\"Q2\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
    );
    let run = assert_success(&invoke(&["run"], Some(run_input.as_bytes())));
    assert_eq!(run.len(), 2);
    assert_eq!(run[0]["id"], "run-1");
    assert_eq!(run[1]["id"], "run-2");

    let multi_output = invoke(
        &[
            "--quiet",
            "decide",
            "--state",
            "shared state",
            "--question",
            "Q1",
            "--question",
            "Q2",
            "--option",
            "A",
            "--option",
            "B",
        ],
        None,
    );
    assert!(
        String::from_utf8_lossy(&multi_output.stderr)
            .contains("using fresh serial full-prompt fallback"),
        "quiet must not suppress a semantic-path warning: {multi_output:?}"
    );
    let multi = assert_success(&multi_output);
    assert_eq!(multi.len(), 2);
    for row in multi {
        assert_eq!(row["execution"]["requested_mode"], "shared");
        assert_eq!(row["execution"]["effective_mode"], "serial");
        assert!(
            row["execution"]["fallback_reason"]
                .as_str()
                .unwrap()
                .contains("probe")
        );
        assert!(row["execution"]["probe_id"].is_null());
        assert!(row.get("shared_timing").is_none());
        assert_eq!(row["cache_hit"], false);
    }

    let required = invoke(
        &[
            "--require-shared",
            "decide",
            "--state",
            "state",
            "--question",
            "Q1",
            "--question",
            "Q2",
            "--option",
            "A",
            "--option",
            "B",
        ],
        None,
    );
    assert_eq!(required.status.code(), Some(2), "{required:?}");
    assert!(required.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&required.stderr).unwrap();
    assert_eq!(error["error"]["code"], "unsupported");
}

#[test]
fn run_continues_after_runtime_row_failure_and_output_is_create_only() {
    if !enabled() {
        return;
    }
    let long_state = "x".repeat(10_000);
    let input = format!(
        "{{\"id\":\"ok-1\",\"state\":\"s\",\"question\":\"Q\",\"options\":[{{\"id\":\"a\",\"description\":\"A\"}},{{\"id\":\"b\",\"description\":\"B\"}}]}}\n{{\"id\":\"too-long\",\"state\":{long_state:?},\"question\":\"Q\",\"options\":[{{\"id\":\"a\",\"description\":\"A\"}},{{\"id\":\"b\",\"description\":\"B\"}}]}}\n{{\"id\":\"ok-2\",\"state\":\"s\",\"question\":\"Q\",\"options\":[{{\"id\":\"a\",\"description\":\"A\"}},{{\"id\":\"b\",\"description\":\"B\"}}]}}\n"
    );
    let output = invoke(&["--max-tokens", "128", "run"], Some(input.as_bytes()));
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let parsed = rows(&output);
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0]["id"], "ok-1");
    assert_eq!(parsed[1]["id"], "too-long");
    assert_eq!(parsed[1]["schema"], "openjev-error-v1");
    assert_eq!(parsed[2]["id"], "ok-2");

    let path = std::env::temp_dir().join(format!("openjev-m4-output-{}.jsonl", std::process::id()));
    let _ = fs::remove_file(&path);
    let path_text = path.to_str().unwrap();
    let first = invoke(
        &["run", "--output", path_text],
        Some(
            b"{\"id\":\"file-1\",\"state\":\"s\",\"question\":\"Q\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
        ),
    );
    let summary = assert_success(&first);
    assert_eq!(summary[0]["schema"], "openjev-write-summary-v1");
    assert_eq!(summary[0]["written"], 1);
    assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);

    let second = invoke(
        &["run", "--output", path_text],
        Some(
            b"{\"id\":\"file-2\",\"state\":\"s\",\"question\":\"Q\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
        ),
    );
    assert_eq!(second.status.code(), Some(2));
    assert!(second.stdout.is_empty());
    let _ = fs::remove_file(path);
}

#[test]
fn run_output_file_is_incremental_before_process_completion() {
    if !enabled() {
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "openjev-m4-streaming-output-{}.jsonl",
        std::process::id()
    ));
    let _ = fs::remove_file(&path);
    let input: String = (1..=20)
        .map(|index| {
            format!(
                "{{\"id\":\"stream-{index}\",\"state\":\"state\",\"question\":\"Q{index}\",\"options\":[{{\"id\":\"a\",\"description\":\"A\"}},{{\"id\":\"b\",\"description\":\"B\"}}]}}\n"
            )
        })
        .collect();
    let mut child = Command::new(env!("CARGO_BIN_EXE_openjev"))
        .args([
            "--offline",
            "--model",
            MODEL,
            "run",
            "--output",
            path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    drop(child.stdin.take());

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if fs::read_to_string(&path).is_ok_and(|contents| contents.lines().count() >= 1) {
            assert!(
                child.try_wait().unwrap().is_none(),
                "first output row must be visible before later scoring completes"
            );
            break;
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("process exited before an incremental row was observed: {status}");
        }
        assert!(Instant::now() < deadline, "timed out waiting for first row");
        std::thread::sleep(Duration::from_millis(5));
    }

    let output = child.wait_with_output().unwrap();
    let summary = assert_success(&output);
    assert_eq!(summary[0]["schema"], "openjev-write-summary-v1");
    assert_eq!(summary[0]["written"], 20);
    assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 20);
    fs::remove_file(path).unwrap();
}

#[test]
fn custom_local_artifact_requires_and_records_explicit_profile_override() {
    if !enabled() {
        return;
    }
    let path = openjev_llama::ModelCache::from_precedence(None)
        .unwrap()
        .model_path(
            openjev_llama::ModelRegistry::bundled()
                .unwrap()
                .resolve(MODEL)
                .unwrap(),
        );
    let output = Command::new(env!("CARGO_BIN_EXE_openjev"))
        .args([
            "--offline",
            "--model",
            path.to_str().unwrap(),
            "--model-sha256",
            SHA256,
            "--template-profile",
            "qwen3",
            "decide",
            "--state",
            "s",
            "--question",
            "Q",
            "--option",
            "A",
            "--option",
            "B",
        ])
        .output()
        .unwrap();
    let parsed = assert_success(&output);
    assert_eq!(parsed[0]["model"]["source"], "local");
    assert_eq!(parsed[0]["model"]["integrity"], "caller-sha256");
    assert_eq!(parsed[0]["model"]["template_override"], true);
    assert_eq!(parsed[0]["model"]["template_status"], "override-unverified");
    assert!(parsed[0]["model"].get("native_reference").is_none());
}
