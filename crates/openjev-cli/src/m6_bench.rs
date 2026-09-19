use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Write},
    process::Command,
    time::Instant,
};

use openjev_core::{Decision, ExecutionMode, Question, Readout, StateValue};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    CliError,
    args::{BenchArgs, GlobalArgs},
    commands::{self, Adapter, DecisionScorer},
    load_scorer,
    m6::{normalized_target, sha256},
    output::{self, WriteSummary},
};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchSample {
    pub repeat: u32,
    pub order_in_repeat: u8,
    pub requested_mode: ExecutionMode,
    pub effective_mode: ExecutionMode,
    pub fallback_reason: Option<String>,
    pub probe_id: Option<String>,
    pub group_wall_seconds: f64,
    pub decisions: usize,
    pub decisions_per_second: f64,
    pub input_tokens_sum: u64,
    pub prefix_tokens: Option<u64>,
    pub true_suffix_tokens: Option<u64>,
    pub encoder_seconds: Option<f64>,
    pub context_seconds: Option<f64>,
    pub forward_seconds_sum: Option<f64>,
    pub shared_group_forward_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ModeStats {
    samples: usize,
    median_seconds: f64,
    p95_seconds: f64,
    median_decisions_per_second: f64,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    schema: &'static str,
    state_file: String,
    state_sha256: String,
    state_bytes: usize,
    questions_file: String,
    questions_sha256: String,
    questions: usize,
    repeats: u32,
    warmup: &'static str,
    model_load_seconds: f64,
    model: Value,
    execution: Value,
    configuration: Value,
    host: Value,
    native_sha: &'static str,
    direct: ModeStats,
    requested_shared: ModeStats,
    shared_speedup: Option<f64>,
    shared_gate_failure: Option<String>,
    samples: Vec<BenchSample>,
    limitations: Vec<&'static str>,
}

pub fn execute_bench<W: Write, E: Write>(
    global: &GlobalArgs,
    args: BenchArgs,
    pretty: bool,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    commands::reject_unimplemented_postprocessing(global)?;
    if global.require_shared {
        return Err(CliError::validation(
            "--require-shared is invalid for bench because M6 measures the requested-shared fallback path",
        ));
    }
    if args.repeats < 5 {
        return Err(CliError::validation("--repeats must be at least 5"));
    }
    preflight_outputs(&args)?;
    let state_bytes = std::fs::read(&args.state_file).map_err(|error| {
        CliError::validation(format!(
            "cannot read state {}: {error}",
            args.state_file.display()
        ))
    })?;
    let state_text = String::from_utf8(state_bytes.clone())
        .map_err(|error| CliError::validation(format!("state must be UTF-8 text: {error}")))?;
    if state_text.is_empty() {
        return Err(CliError::validation("state file must not be empty"));
    }
    let state = StateValue::string(state_text).map_err(CliError::from_core_validation)?;
    let question_bytes = std::fs::read(&args.questions).map_err(|error| {
        CliError::validation(format!(
            "cannot read questions {}: {error}",
            args.questions.display()
        ))
    })?;
    let questions = parse_questions(&question_bytes)?;
    let decisions: Vec<_> = questions
        .into_iter()
        .map(|question| {
            Decision::new(
                question.id,
                state.clone(),
                question.question,
                question.options,
            )
            .map_err(CliError::from_core_validation)
        })
        .collect::<Result<_, _>>()?;
    let config = commands::scoring_config(global)?;

    let mut report_file = args
        .output
        .as_deref()
        .map(output::create_jsonl_new)
        .transpose()?;
    let mut samples_file = args
        .samples_output
        .as_deref()
        .map(output::create_jsonl_new)
        .transpose()?;

    let load_start = Instant::now();
    let mut scorer = load_scorer(&config)?;
    let model_load_seconds = load_start.elapsed().as_secs_f64();

    let warmup = scorer.score_direct(decisions[0].clone())?;
    let shared_probe = scorer.probe_id(ExecutionMode::Shared);
    let shared_gate_failure = shared_probe.as_ref().err().cloned();
    if let Some(reason) = &shared_gate_failure {
        writeln!(
            stderr,
            "bench: requested shared is ineligible; measuring serial fallback: {reason}"
        )
        .map_err(|error| CliError::runtime("stderr_io", error.to_string()))?;
    }

    let mut samples = Vec::with_capacity(args.repeats as usize * 2);
    for repeat in 0..args.repeats {
        let order = if repeat % 2 == 0 {
            [ExecutionMode::Direct, ExecutionMode::Shared]
        } else {
            [ExecutionMode::Shared, ExecutionMode::Direct]
        };
        for (order_index, requested) in order.into_iter().enumerate() {
            samples.push(run_sample(
                scorer.as_mut(),
                &decisions,
                repeat,
                u8::try_from(order_index + 1).expect("order is 1 or 2"),
                requested,
                shared_probe.as_deref().map_err(|reason| reason.as_str()),
                config.confidence,
            )?);
        }
    }
    scorer.shutdown()?;

    let direct_samples: Vec<_> = samples
        .iter()
        .filter(|sample| sample.requested_mode == ExecutionMode::Direct)
        .collect();
    let shared_samples: Vec<_> = samples
        .iter()
        .filter(|sample| sample.requested_mode == ExecutionMode::Shared)
        .collect();
    let direct = stats(&direct_samples);
    let requested_shared = stats(&shared_samples);
    let all_shared_effective = shared_samples
        .iter()
        .all(|sample| sample.effective_mode == ExecutionMode::Shared);
    let shared_speedup =
        all_shared_effective.then(|| direct.median_seconds / requested_shared.median_seconds);

    let first = samples_metadata(&warmup);
    let report = BenchReport {
        schema: "openjev-bench-v1",
        state_file: args.state_file.display().to_string(),
        state_sha256: sha256(&state_bytes),
        state_bytes: state_bytes.len(),
        questions_file: args.questions.display().to_string(),
        questions_sha256: sha256(&question_bytes),
        questions: decisions.len(),
        repeats: args.repeats,
        warmup: "one untimed direct decision after one model load",
        model_load_seconds,
        model: first.0,
        execution: first.1,
        configuration: configuration_json(&config),
        host: host_json(),
        native_sha: openjev_llama::NATIVE_PIN,
        direct,
        requested_shared,
        shared_speedup,
        shared_gate_failure,
        samples,
        limitations: vec![
            "Timed group wall excludes model load, warmup, input validation, and output writes.",
            "All scoring uses one owner worker sequentially and one loaded model per process.",
            "A requested-shared serial fallback is measured as the emitted path but is not called a shared speedup.",
            "Per-row native timing is summed only for direct/serial rows; shared group timing is read once and never duplicated across rows.",
        ],
    };

    if let (Some(file), Some(path)) = (samples_file.as_mut(), args.samples_output.as_deref()) {
        output::write_jsonl(file, &report.samples)
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
        output::sync_jsonl(file, path)?;
    }
    if let (Some(file), Some(path)) = (report_file.as_mut(), args.output.as_deref()) {
        output::write_json(file, &report, pretty)
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
        output::sync_jsonl(file, path)?;
        output::write_json(
            stdout,
            &WriteSummary {
                schema: "openjev-write-summary-v1",
                path: path.display().to_string(),
                written: report.samples.len(),
                failed: 0,
            },
            false,
        )
        .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
    } else {
        output::write_json(stdout, &report, pretty)
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
    }
    Ok(0)
}

fn run_sample(
    scorer: &mut dyn DecisionScorer,
    decisions: &[Decision],
    repeat: u32,
    order_in_repeat: u8,
    requested: ExecutionMode,
    shared_eligibility: Result<&str, &str>,
    confidence: bool,
) -> Result<BenchSample, CliError> {
    let started = Instant::now();
    let rows = if requested == ExecutionMode::Shared {
        match shared_eligibility {
            Ok(probe_id) => scorer.score_shared(decisions.to_vec(), probe_id.to_owned())?,
            Err(gate_failure) => {
                let group_id = format!("bench-{repeat}-shared");
                decisions
                    .iter()
                    .map(|decision| {
                        commands::score_item_with_reason(
                            scorer,
                            &Adapter::Choice(decision.clone()),
                            ExecutionMode::Shared,
                            confidence,
                            Some(&group_id),
                            Some(gate_failure),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
    } else {
        decisions
            .iter()
            .map(|decision| {
                commands::score_item_with_reason(
                    scorer,
                    &Adapter::Choice(decision.clone()),
                    ExecutionMode::Direct,
                    confidence,
                    None,
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let wall = started.elapsed().as_secs_f64();
    sample_from_rows(repeat, order_in_repeat, requested, wall, &rows)
}

fn sample_from_rows(
    repeat: u32,
    order_in_repeat: u8,
    requested: ExecutionMode,
    wall: f64,
    rows: &[Readout],
) -> Result<BenchSample, CliError> {
    let first = rows
        .first()
        .ok_or_else(|| CliError::runtime("benchmark", "benchmark produced no rows"))?;
    let effective = first.execution.effective_mode;
    if rows.iter().any(|row| {
        row.execution.requested_mode != requested || row.execution.effective_mode != effective
    }) {
        return Err(CliError::runtime(
            "benchmark",
            "benchmark group reported inconsistent execution modes",
        ));
    }
    let direct_timing = effective != ExecutionMode::Shared;
    let forward = direct_timing.then(|| rows.iter().filter_map(|row| row.forward_seconds).sum());
    let shared = (effective == ExecutionMode::Shared)
        .then_some(first.shared_timing.as_ref())
        .flatten();
    Ok(BenchSample {
        repeat,
        order_in_repeat,
        requested_mode: requested,
        effective_mode: effective,
        fallback_reason: first.execution.fallback_reason.clone(),
        probe_id: first.execution.probe_id.clone(),
        group_wall_seconds: wall,
        decisions: rows.len(),
        decisions_per_second: rows.len() as f64 / wall,
        input_tokens_sum: rows.iter().map(|row| row.input_tokens).sum(),
        prefix_tokens: shared.map(|timing| timing.prefix_tokens),
        true_suffix_tokens: shared.map(|timing| timing.true_suffix_tokens),
        encoder_seconds: shared.map(|timing| timing.encode_seconds),
        context_seconds: shared.map(|timing| timing.prefill_seconds + timing.replicate_seconds),
        forward_seconds_sum: forward,
        shared_group_forward_seconds: shared.map(|timing| timing.suffix_forward_seconds),
    })
}

fn parse_questions(bytes: &[u8]) -> Result<Vec<Question>, CliError> {
    let reader = BufReader::new(bytes);
    let mut questions = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| CliError::validation(error.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let question: Question = serde_json::from_str(&line).map_err(|error| {
            CliError::validation(format!("questions line {} is invalid: {error}", index + 1))
        })?;
        if !ids.insert(question.id.clone()) {
            return Err(CliError::validation(format!(
                "duplicate question ID {:?}",
                question.id
            )));
        }
        questions.push(question);
    }
    if questions.is_empty() {
        return Err(CliError::validation(
            "questions file must contain at least one row",
        ));
    }
    Ok(questions)
}

fn stats(samples: &[&BenchSample]) -> ModeStats {
    let mut seconds: Vec<_> = samples
        .iter()
        .map(|sample| sample.group_wall_seconds)
        .collect();
    let mut throughput: Vec<_> = samples
        .iter()
        .map(|sample| sample.decisions_per_second)
        .collect();
    ModeStats {
        samples: samples.len(),
        median_seconds: percentile(&mut seconds, 0.5),
        p95_seconds: percentile(&mut seconds, 0.95),
        median_decisions_per_second: percentile(&mut throughput, 0.5),
    }
}

fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = (percentile * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn samples_metadata(readout: &Readout) -> (Value, Value) {
    (
        serde_json::to_value(&readout.model).expect("model serializes"),
        serde_json::to_value(&readout.execution).expect("execution serializes"),
    )
}

fn configuration_json(config: &commands::ScoringConfig) -> Value {
    json!({
        "device": config.device,
        "gpu_layers_requested": config.gpu_layers,
        "threads": config.threads,
        "n_ctx_requested": config.n_ctx,
        "max_tokens": config.max_tokens,
        "max_context_tokens": config.max_context_tokens,
        "n_batch": config.n_batch,
        "n_ubatch": config.n_ubatch,
        "max_sequences": config.max_sequences,
        "offline": config.offline,
        "compiled_features": {
            "native": cfg!(feature = "native"),
            "metal": cfg!(feature = "metal"),
            "cuda": cfg!(feature = "cuda")
        }
    })
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn host_json() -> Value {
    json!({
        "os": command_output("sw_vers", &["-productVersion"]),
        "kernel": command_output("uname", &["-srvmp"]),
        "cpu": command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "ram_bytes": command_output("sysctl", &["-n", "hw.memsize"]).and_then(|value| value.parse::<u64>().ok()),
        "gpu": command_output("system_profiler", &["SPDisplaysDataType", "-detailLevel", "mini"]),
        "rustc": command_output("rustc", &["--version"]),
        "cargo": command_output("cargo", &["--version"])
    })
}

fn preflight_outputs(args: &BenchArgs) -> Result<(), CliError> {
    for output_path in [&args.output, &args.samples_output].into_iter().flatten() {
        commands::preflight_output(output_path, Some(&args.state_file))?;
        if normalized_target(output_path)?
            == std::fs::canonicalize(&args.questions).map_err(|error| {
                CliError::validation(format!("cannot resolve questions: {error}"))
            })?
        {
            return Err(CliError::validation(
                "benchmark output must not overwrite the questions input",
            ));
        }
    }
    if let (Some(left), Some(right)) = (&args.output, &args.samples_output)
        && normalized_target(left)? == normalized_target(right)?
    {
        return Err(CliError::validation(
            "--output and --samples-output must identify different files",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BenchSample, stats};
    use openjev_core::ExecutionMode;

    fn sample(seconds: f64) -> BenchSample {
        BenchSample {
            repeat: 0,
            order_in_repeat: 1,
            requested_mode: ExecutionMode::Direct,
            effective_mode: ExecutionMode::Direct,
            fallback_reason: None,
            probe_id: None,
            group_wall_seconds: seconds,
            decisions: 21,
            decisions_per_second: 21.0 / seconds,
            input_tokens_sum: 210,
            prefix_tokens: None,
            true_suffix_tokens: None,
            encoder_seconds: None,
            context_seconds: None,
            forward_seconds_sum: Some(seconds),
            shared_group_forward_seconds: None,
        }
    }

    #[test]
    fn injected_clock_samples_have_deterministic_nearest_rank_stats() {
        let samples: Vec<_> = [5.0, 1.0, 4.0, 2.0, 3.0].into_iter().map(sample).collect();
        let refs: Vec<_> = samples.iter().collect();
        let summary = stats(&refs);
        assert_eq!(summary.median_seconds, 3.0);
        assert_eq!(summary.p95_seconds, 5.0);
        assert_eq!(summary.median_decisions_per_second, 7.0);
    }
}
