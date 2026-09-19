#![cfg(not(feature = "native"))]

use std::{
    io::Write as _,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_openjev")
}

fn parse_one(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).expect("stream must contain exactly one JSON value")
}

#[test]
fn help_version_and_every_command_help_are_json_stdout_only() {
    for args in [
        vec!["--help"],
        vec!["--version"],
        vec!["decide", "--help"],
        vec!["noul", "--help"],
        vec!["score", "--help"],
        vec!["ask", "--help"],
        vec!["run", "--help"],
        vec!["models", "--help"],
        vec!["models", "list", "--help"],
        vec!["models", "pull", "--help"],
        vec!["models", "path", "--help"],
        vec!["models", "probe", "--help"],
        vec!["eval", "--help"],
        vec!["bench", "--help"],
        vec!["calibrate", "--help"],
    ] {
        let output = Command::new(binary()).args(&args).output().unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{args:?}");
        let json = parse_one(&output.stdout);
        if args == ["--version"] {
            assert_eq!(json["schema"], "openjev-version-v1");
        } else {
            assert_eq!(json["schema"], "openjev-help-v1");
            let text = json["text"].as_str().unwrap();
            assert!(text.contains("Usage:"));
            assert!(text.contains("Example"), "{args:?}: {text}");
        }
    }
}

#[test]
fn help_metadata_uses_parsed_command_context_not_option_values() {
    for (args, expected_command, expected_usage) in [
        (
            vec!["models", "pull", "--help"],
            "openjev models pull",
            "Usage: openjev models pull [OPTIONS] <ID>",
        ),
        (
            vec!["--model", "run", "models", "pull", "--help"],
            "openjev models pull",
            "Usage: openjev models pull [OPTIONS] <ID>",
        ),
        (
            vec!["models", "pull", "--model", "run", "--help"],
            "openjev models pull",
            "Usage: openjev models pull [OPTIONS] <ID>",
        ),
        (
            vec!["run", "--model", "models", "--help"],
            "openjev run",
            "Usage: openjev run [OPTIONS]",
        ),
        (
            vec!["--model", "models", "run", "--help"],
            "openjev run",
            "Usage: openjev run [OPTIONS]",
        ),
        (
            vec!["help", "models", "pull"],
            "openjev models pull",
            "Usage: openjev models pull [OPTIONS] <ID>",
        ),
        (
            vec!["models", "pull", "-h"],
            "openjev models pull",
            "Usage: openjev models pull [OPTIONS] <ID>",
        ),
    ] {
        let output = Command::new(binary()).args(&args).output().unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{args:?}");
        let json = parse_one(&output.stdout);
        assert_eq!(json["schema"], "openjev-help-v1", "{args:?}");
        assert_eq!(json["command"], expected_command, "{args:?}");
        assert_eq!(json["usage"], expected_usage, "{args:?}");
        assert!(
            json["text"].as_str().unwrap().contains(expected_usage),
            "{args:?}"
        );
    }
}

#[test]
fn compact_is_documented_and_rejected_for_nondecision_commands() {
    let help = Command::new(binary()).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help = parse_one(&help.stdout);
    assert!(help["text"].as_str().unwrap().contains("--compact"));
    assert!(
        help["text"]
            .as_str()
            .unwrap()
            .contains("decide, noul, score, ask, and run")
    );

    for args in [
        vec!["--compact", "models", "list"],
        vec!["--compact", "eval", "--fixture", "authored144"],
        vec![
            "--compact",
            "bench",
            "--state-file",
            "missing",
            "--questions",
            "missing",
        ],
        vec!["--compact", "calibrate", "--input", "missing"],
    ] {
        let output = Command::new(binary()).args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
        let error = parse_one(&output.stderr);
        assert_eq!(error["error"]["code"], "validation", "{args:?}");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("decision commands")
        );
    }

    let output = Command::new(binary())
        .args(["--compact", "--pretty", "run"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(parse_one(&output.stderr)["error"]["code"], "validation");
}

#[test]
fn validation_and_backend_disabled_errors_have_empty_stdout_and_stable_exits() {
    let output = Command::new(binary()).arg("unknown").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert_eq!(parse_one(&output.stderr)["error"]["code"], "usage");

    let output = Command::new(binary())
        .args([
            "decide",
            "--question",
            "q",
            "--option",
            "a",
            "--option",
            "b",
            "--state",
            "s",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(
        parse_one(&output.stderr)["error"]["code"],
        "backend_unavailable"
    );
}

#[test]
fn explicit_state_does_not_wait_for_open_piped_stdin() {
    let mut child = Command::new(binary())
        .args([
            "decide",
            "--question",
            "q",
            "--option",
            "a",
            "--option",
            "b",
            "--state",
            "untrimmed state",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("explicit state command blocked reading stdin");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(
        parse_one(&output.stderr)["error"]["code"],
        "backend_unavailable"
    );
}

#[test]
fn malformed_jsonl_fails_before_backend_and_models_list_is_json_not_paths() {
    let mut child = Command::new(binary())
        .arg("run")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(b"{bad}\n").unwrap();
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let error = parse_one(&output.stderr);
    assert_eq!(error["parse_status"], "unparsed");

    let cache = std::env::temp_dir().join(format!("openjev-m4-model-list-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    let output = Command::new(binary())
        .args(["--cache-dir", cache.to_str().unwrap(), "models", "list"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
    let json = parse_one(&output.stdout);
    assert_eq!(json["schema"], "openjev-models-v1");
    assert_eq!(json["models"].as_array().unwrap().len(), 3);
    assert!(json["models"].as_array().unwrap().iter().all(|row| {
        row["cached"] == false && row["verified"] == false && row["path"].is_null()
    }));

    let output = Command::new(binary())
        .args(["--device", "cpu", "models", "path", "qwen3-0.6b"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert_eq!(parse_one(&output.stderr)["error"]["code"], "validation");
    let _ = std::fs::remove_dir_all(cache);
}

#[test]
fn create_only_run_reserves_an_empty_file_before_backend_startup() {
    let path = std::env::temp_dir().join(format!(
        "openjev-m4-empty-on-startup-failure-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut child = Command::new(binary())
        .args(["run", "--output", path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(
            b"{\"id\":\"d1\",\"state\":\"s\",\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
        )
        .unwrap();
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    assert_eq!(
        parse_one(&output.stderr)["error"]["code"],
        "backend_unavailable"
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn eval_import_accepts_floats_counts_invalid_and_missing_without_backend() {
    let path = std::env::temp_dir().join(format!("openjev-m6-import-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    std::fs::write(
        &path,
        concat!(
            "{\"id\":\"a3f18f3a63d45345942b\",\"option_ids\":[\"supported\",\"insufficient\",\"contradicted\"],\"probabilities\":[0.1,0.2,0.7]}\n",
            "{\"id\":\"f40beba9088c8db8bbd6\",\"option_ids\":[\"supported\",\"insufficient\",\"contradicted\"],\"probabilities\":[true,0.0,0.0]}\n"
        ),
    )
    .unwrap();
    let output = Command::new(binary())
        .args([
            "eval",
            "--fixture",
            "authored144",
            "--predictions",
            path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
    let report = parse_one(&output.stdout);
    assert_eq!(report["schema"], "openjev-eval-command-v1");
    assert_eq!(report["quality"]["available_gold"], 144);
    assert_eq!(report["quality"]["scored"], 2);
    assert_eq!(report["quality"]["invalid"], 1);
    assert_eq!(report["quality"]["missing"], 142);
    assert_eq!(report["quality"]["overall"]["probability_rows"], 1);
    assert!(report["quality"]["overall"]["nll"].is_null());
    assert!(report["quality"]["overall"]["brier"].is_null());
    let schema_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/eval-v1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(schema_path).unwrap()).unwrap();
    jsonschema::validator_for(&schema)
        .unwrap()
        .validate(&report)
        .unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bench_prevalidates_repeats_inputs_and_create_only_outputs() {
    let root = std::env::temp_dir().join(format!("openjev-m6-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    let state = root.join("state.txt");
    let questions = root.join("questions.jsonl");
    let report = root.join("report.json");
    let samples = root.join("samples.jsonl");
    std::fs::write(&state, "owned benchmark state").unwrap();
    std::fs::write(
        &questions,
        "{\"id\":\"q1\",\"question\":\"Which?\",\"options\":[{\"id\":\"a\",\"description\":\"A\"},{\"id\":\"b\",\"description\":\"B\"}]}\n",
    )
    .unwrap();
    let output = Command::new(binary())
        .args([
            "bench",
            "--state-file",
            state.to_str().unwrap(),
            "--questions",
            questions.to_str().unwrap(),
            "--repeats",
            "4",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(parse_one(&output.stderr)["error"]["code"], "validation");

    let output = Command::new(binary())
        .args([
            "bench",
            "--state-file",
            state.to_str().unwrap(),
            "--questions",
            questions.to_str().unwrap(),
            "--output",
            report.to_str().unwrap(),
            "--samples-output",
            samples.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    assert_eq!(
        parse_one(&output.stderr)["error"]["code"],
        "backend_unavailable"
    );
    assert_eq!(std::fs::metadata(report).unwrap().len(), 0);
    assert_eq!(std::fs::metadata(samples).unwrap().len(), 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn closed_stdout_is_a_clean_nonzero_exit_without_panic_text() {
    let mut child = Command::new(binary())
        .arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("panicked"), "{stderr}");
}
