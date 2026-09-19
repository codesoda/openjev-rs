#[cfg(feature = "integration")]
mod integration {
    use std::{
        collections::{HashMap, HashSet},
        fs::{self, OpenOptions},
        io::{BufRead, BufReader, Write as _},
        path::{Path, PathBuf},
        str::FromStr as _,
    };

    use openjev_core::{Decision, Device, GpuLayersRequested, Readout};
    use openjev_llama::{EngineHandle, EngineOptions, ModelCache, ModelRegistry, hash_file};
    use serde::{Deserialize, Serialize};
    use tracing_subscriber::EnvFilter;

    const QWEN_ID: &str = "qwen3-0.6b";

    #[derive(Debug)]
    struct Args {
        cache_dir: Option<PathBuf>,
        device: Device,
        threads: Option<u32>,
        n_ctx: Option<u32>,
        authored_output: PathBuf,
        perturbations_output: PathBuf,
        report: PathBuf,
    }

    #[derive(Clone, Debug, Deserialize)]
    struct ReferencePrediction {
        id: String,
        option_ids: Vec<String>,
        answer_token_ids: Vec<u32>,
        input_tokens: u64,
        prompt_sha256: String,
        option_logits: Vec<f64>,
        probabilities: Vec<f64>,
    }

    #[derive(Clone, Debug)]
    struct FixtureSet {
        name: &'static str,
        path: PathBuf,
        rows: Vec<Decision>,
    }

    #[derive(Debug, Serialize)]
    struct FileRecord {
        path: String,
        bytes: u64,
        sha256: String,
        rows: usize,
    }

    #[derive(Debug, Serialize)]
    struct MismatchRow {
        id: String,
        reference_choice: String,
        local_choice: String,
        reference_margin: f64,
        local_margin: f64,
    }

    #[derive(Debug, Serialize)]
    struct NumericalComparison {
        rows: usize,
        slot_values: usize,
        argmax_agreement: usize,
        argmax_agreement_rate: f64,
        logit_mae: f64,
        logit_rmse: f64,
        logit_max_abs: f64,
        probability_mae: f64,
        probability_rmse: f64,
        probability_max_abs: f64,
        mismatch_ids: Vec<MismatchRow>,
    }

    #[derive(Debug, Serialize)]
    struct ExactGate {
        rows: usize,
        prompt_sha256_exact: usize,
        input_tokens_exact: usize,
        option_ids_exact: usize,
        answer_token_ids_exact: usize,
        all_slot_boundaries_verified: usize,
        finite_readouts: usize,
        status: &'static str,
    }

    #[derive(Debug, Serialize)]
    struct SetReport {
        fixture: FileRecord,
        predictions: FileRecord,
        exact_gate: ExactGate,
        numerical_comparison_vs_native_bf16: NumericalComparison,
    }

    #[derive(Debug, Serialize)]
    struct RunConfig {
        device: Device,
        gpu_layers_requested: GpuLayersRequested,
        threads: u32,
        n_ctx_requested: Option<u32>,
        max_tokens: u32,
        max_context_tokens: u32,
        n_batch: u32,
        n_ubatch: u32,
        offline: bool,
        one_prefill_per_decision: bool,
        generation: bool,
    }

    #[derive(Debug, Serialize)]
    struct Report {
        schema: &'static str,
        status: &'static str,
        reference_predictions: FileRecord,
        model: serde_json::Value,
        config: RunConfig,
        authored144: SetReport,
        perturbations108: SetReport,
        limitations: Vec<&'static str>,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        schema: &'static str,
        status: &'static str,
        authored_predictions: FileRecord,
        perturbation_predictions: FileRecord,
        report: FileRecord,
    }

    #[derive(Debug, Serialize)]
    struct ErrorOutput {
        schema: &'static str,
        status: &'static str,
        error: String,
    }

    pub fn run() -> i32 {
        if std::env::var("OPENJEV_INTEGRATION").as_deref() != Ok("1") {
            return emit_error("m3_parity requires OPENJEV_INTEGRATION=1", 2);
        }
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .init();
        match run_inner() {
            Ok(summary) => match write_stdout(&summary) {
                Ok(()) => 0,
                Err(error) => emit_error(&error, 1),
            },
            Err(error) => emit_error(&error, 1),
        }
    }

    fn run_inner() -> Result<Summary, String> {
        let args = parse_args()?;
        preflight_outputs([
            args.authored_output.as_path(),
            args.perturbations_output.as_path(),
            args.report.as_path(),
        ])?;

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let authored = load_fixture(
            "authored144",
            root.join("reference/semif-py/benchmarks/data/authored144.jsonl"),
            144,
        )?;
        let perturbations = load_fixture(
            "perturbations108",
            root.join("reference/semif-py/benchmarks/data/perturbations108.jsonl"),
            108,
        )?;
        let reference_path =
            root.join("reference/semif-py/browser-ladder-qwen3-0.6b.predictions.jsonl");
        let references = load_references(&reference_path)?;
        validate_reference_coverage(&authored, &perturbations, &references)?;

        let registry = ModelRegistry::bundled().map_err(|error| error.to_string())?;
        let spec = registry
            .resolve(QWEN_ID)
            .map_err(|error| error.to_string())?
            .clone();
        let cache = ModelCache::from_precedence(args.cache_dir.as_deref())
            .map_err(|error| error.to_string())?;
        // Offline-only: this gate never downloads and reuses the M2-verified cache.
        let artifact = registry
            .path(&cache, QWEN_ID)
            .map_err(|error| error.to_string())?;
        let mut options = EngineOptions {
            device: args.device,
            gpu_layers: match args.device {
                Device::Cpu => GpuLayersRequested::Count(0),
                Device::Metal | Device::Cuda => GpuLayersRequested::All,
            },
            n_ctx: args.n_ctx,
            ..EngineOptions::default()
        };
        if let Some(threads) = args.threads {
            options.threads = threads;
        }
        let config = RunConfig {
            device: options.device,
            gpu_layers_requested: options.gpu_layers,
            threads: options.threads,
            n_ctx_requested: options.n_ctx,
            max_tokens: options.max_tokens,
            max_context_tokens: options.max_context_tokens,
            n_batch: options.n_batch,
            n_ubatch: options.n_ubatch,
            offline: true,
            one_prefill_per_decision: true,
            generation: false,
        };
        let engine =
            EngineHandle::spawn(spec, artifact, options).map_err(|error| error.to_string())?;
        let model = serde_json::to_value(engine.model_info()).map_err(|error| error.to_string())?;

        let authored_rows = score_set(&engine, &authored, &references)?;
        let perturbation_rows = score_set(&engine, &perturbations, &references)?;
        engine.shutdown().map_err(|error| error.to_string())?;

        let authored_predictions = write_jsonl_create_new(&args.authored_output, &authored_rows)?;
        let perturbation_predictions =
            write_jsonl_create_new(&args.perturbations_output, &perturbation_rows)?;
        let reference_predictions = file_record(&reference_path, references.len())?;
        let authored_report =
            set_report(&authored, authored_predictions, &authored_rows, &references)?;
        let perturbations_report = set_report(
            &perturbations,
            perturbation_predictions,
            &perturbation_rows,
            &references,
        )?;
        let report = Report {
            schema: "openjev-m3-parity-report-v1",
            status: "passed",
            reference_predictions,
            model,
            config,
            authored144: authored_report,
            perturbations108: perturbations_report,
            limitations: vec![
                "The numerical comparison is local Q8_0 GGUF versus committed native-BF16 reference rows, not same-artifact backend parity.",
                "No numerical acceptance threshold is asserted; Astra must adjudicate the measured quantization delta.",
                "The perturbations108 result is extended coverage and does not weaken the mandatory authored144 gate.",
            ],
        };
        write_json_create_new(&args.report, &report)?;
        let report_record = file_record(&args.report, 1)?;
        Ok(Summary {
            schema: "openjev-m3-parity-summary-v1",
            status: "passed",
            authored_predictions: report.authored144.predictions,
            perturbation_predictions: report.perturbations108.predictions,
            report: report_record,
        })
    }

    fn load_fixture(
        name: &'static str,
        path: PathBuf,
        expected: usize,
    ) -> Result<FixtureSet, String> {
        let lines = read_nonempty_lines(&path)?;
        let mut ids = HashSet::with_capacity(lines.len());
        let mut rows = Vec::with_capacity(lines.len());
        for (index, line) in lines.into_iter().enumerate() {
            let decision: Decision = serde_json::from_str(&line)
                .map_err(|error| format!("{} line {}: {error}", path.display(), index + 1))?;
            decision.validate().map_err(|error| error.to_string())?;
            if !ids.insert(decision.id.clone()) {
                return Err(format!("duplicate fixture ID {:?}", decision.id));
            }
            rows.push(decision);
        }
        if rows.len() != expected {
            return Err(format!(
                "{name} has {} rows, expected {expected}",
                rows.len()
            ));
        }
        Ok(FixtureSet { name, path, rows })
    }

    fn load_references(path: &Path) -> Result<HashMap<String, ReferencePrediction>, String> {
        let lines = read_nonempty_lines(path)?;
        let mut references = HashMap::with_capacity(lines.len());
        for (index, line) in lines.into_iter().enumerate() {
            let row: ReferencePrediction = serde_json::from_str(&line)
                .map_err(|error| format!("{} line {}: {error}", path.display(), index + 1))?;
            if row.option_ids.len() != row.option_logits.len()
                || row.option_ids.len() != row.probabilities.len()
                || row.option_ids.len() != row.answer_token_ids.len()
            {
                return Err(format!("reference row {:?} has misaligned vectors", row.id));
            }
            let id = row.id.clone();
            if references.insert(id.clone(), row).is_some() {
                return Err(format!("duplicate reference ID {id:?}"));
            }
        }
        if references.len() != 252 {
            return Err(format!(
                "reference predictions have {} unique rows, expected 252",
                references.len()
            ));
        }
        Ok(references)
    }

    fn validate_reference_coverage(
        authored: &FixtureSet,
        perturbations: &FixtureSet,
        references: &HashMap<String, ReferencePrediction>,
    ) -> Result<(), String> {
        let mut selected = HashSet::with_capacity(252);
        for decision in authored.rows.iter().chain(&perturbations.rows) {
            if !selected.insert(decision.id.as_str()) {
                return Err(format!("fixture ID {:?} occurs in both sets", decision.id));
            }
            let reference = references
                .get(&decision.id)
                .ok_or_else(|| format!("missing reference prediction for {:?}", decision.id))?;
            let ids: Vec<_> = decision
                .options
                .iter()
                .map(|option| option.id.clone())
                .collect();
            if ids != reference.option_ids {
                return Err(format!(
                    "fixture/reference option_ids differ for {:?}",
                    decision.id
                ));
            }
        }
        if selected.len() != 252 {
            return Err(format!(
                "selected {} fixture IDs, expected 252",
                selected.len()
            ));
        }
        Ok(())
    }

    fn score_set(
        engine: &EngineHandle,
        fixture: &FixtureSet,
        references: &HashMap<String, ReferencePrediction>,
    ) -> Result<Vec<Readout>, String> {
        let mut rows = Vec::with_capacity(fixture.rows.len());
        for decision in &fixture.rows {
            let reference = &references[&decision.id];
            let readout = engine
                .score_direct(decision.clone())
                .map_err(|error| format!("{} {:?}: {error}", fixture.name, decision.id))?;
            readout.validate().map_err(|error| error.to_string())?;
            require_exact(&readout, reference)?;
            rows.push(readout);
        }
        Ok(rows)
    }

    fn require_exact(readout: &Readout, reference: &ReferencePrediction) -> Result<(), String> {
        let mismatch = if readout.id != reference.id {
            Some("id")
        } else if readout.option_ids != reference.option_ids {
            Some("option_ids")
        } else if readout.prompt_sha256 != reference.prompt_sha256 {
            Some("prompt_sha256")
        } else if readout.input_tokens != reference.input_tokens {
            Some("input_tokens")
        } else if readout.answer_token_ids != reference.answer_token_ids {
            Some("answer_token_ids")
        } else if readout.cache_hit != Some(false) {
            Some("cache_hit (direct scoring must report fresh inference prefill)")
        } else {
            None
        };
        if let Some(field) = mismatch {
            Err(format!(
                "mandatory exact gate refused row {:?}: {field} mismatch",
                reference.id
            ))
        } else {
            Ok(())
        }
    }

    fn set_report(
        fixture: &FixtureSet,
        predictions: FileRecord,
        rows: &[Readout],
        references: &HashMap<String, ReferencePrediction>,
    ) -> Result<SetReport, String> {
        let mut logit_abs_sum = 0.0;
        let mut logit_sq_sum = 0.0;
        let mut logit_max = 0.0_f64;
        let mut probability_abs_sum = 0.0;
        let mut probability_sq_sum = 0.0;
        let mut probability_max = 0.0_f64;
        let mut slot_values = 0usize;
        let mut agreements = 0usize;
        let mut mismatches = Vec::new();
        for row in rows {
            let reference = &references[&row.id];
            for (local, native) in row.option_logits.iter().zip(&reference.option_logits) {
                let delta = local - native;
                logit_abs_sum += delta.abs();
                logit_sq_sum += delta * delta;
                logit_max = logit_max.max(delta.abs());
                slot_values += 1;
            }
            for (local, native) in row.probabilities.iter().zip(&reference.probabilities) {
                let delta = local - native;
                probability_abs_sum += delta.abs();
                probability_sq_sum += delta * delta;
                probability_max = probability_max.max(delta.abs());
            }
            let reference_index = first_argmax(&reference.probabilities)?;
            if row.choice_index == reference_index {
                agreements += 1;
            } else {
                mismatches.push(MismatchRow {
                    id: row.id.clone(),
                    reference_choice: reference.option_ids[reference_index].clone(),
                    local_choice: row.choice.clone(),
                    reference_margin: top_margin(&reference.probabilities, reference_index),
                    local_margin: top_margin(&row.probabilities, row.choice_index),
                });
            }
        }
        let denominator = slot_values as f64;
        Ok(SetReport {
            fixture: file_record(&fixture.path, fixture.rows.len())?,
            predictions,
            exact_gate: ExactGate {
                rows: rows.len(),
                prompt_sha256_exact: rows.len(),
                input_tokens_exact: rows.len(),
                option_ids_exact: rows.len(),
                answer_token_ids_exact: rows.len(),
                all_slot_boundaries_verified: rows.len(),
                finite_readouts: rows.len(),
                status: "passed",
            },
            numerical_comparison_vs_native_bf16: NumericalComparison {
                rows: rows.len(),
                slot_values,
                argmax_agreement: agreements,
                argmax_agreement_rate: agreements as f64 / rows.len() as f64,
                logit_mae: logit_abs_sum / denominator,
                logit_rmse: (logit_sq_sum / denominator).sqrt(),
                logit_max_abs: logit_max,
                probability_mae: probability_abs_sum / denominator,
                probability_rmse: (probability_sq_sum / denominator).sqrt(),
                probability_max_abs: probability_max,
                mismatch_ids: mismatches,
            },
        })
    }

    fn first_argmax(values: &[f64]) -> Result<usize, String> {
        let Some((&first, rest)) = values.split_first() else {
            return Err("empty probability vector".to_owned());
        };
        if !first.is_finite() {
            return Err("nonfinite probability".to_owned());
        }
        let mut best = first;
        let mut index = 0;
        for (offset, value) in rest.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("nonfinite probability".to_owned());
            }
            if value > best {
                best = value;
                index = offset + 1;
            }
        }
        Ok(index)
    }

    fn top_margin(values: &[f64], choice: usize) -> f64 {
        let runner_up = values
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != choice)
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        values[choice] - runner_up
    }

    fn preflight_outputs<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Result<(), String> {
        for path in paths {
            if fs::symlink_metadata(path).is_ok() {
                return Err(format!(
                    "create-only output already exists: {}",
                    path.display()
                ));
            }
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
                && !parent.is_dir()
            {
                return Err(format!(
                    "output parent does not exist: {}",
                    parent.display()
                ));
            }
        }
        Ok(())
    }

    fn write_jsonl_create_new(path: &Path, rows: &[Readout]) -> Result<FileRecord, String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create {}: {error}", path.display()))?;
        for row in rows {
            serde_json::to_writer(&mut file, row).map_err(|error| error.to_string())?;
            file.write_all(b"\n").map_err(|error| error.to_string())?;
        }
        file.sync_all().map_err(|error| error.to_string())?;
        file_record(path, rows.len())
    }

    fn write_json_create_new<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create {}: {error}", path.display()))?;
        serde_json::to_writer_pretty(&mut file, value).map_err(|error| error.to_string())?;
        file.write_all(b"\n").map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())
    }

    fn file_record(path: &Path, rows: usize) -> Result<FileRecord, String> {
        let (bytes, sha256) = hash_file(path).map_err(|error| error.to_string())?;
        Ok(FileRecord {
            path: path.display().to_string(),
            bytes,
            sha256,
            rows,
        })
    }

    fn read_nonempty_lines(path: &Path) -> Result<Vec<String>, String> {
        let file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        BufReader::new(file)
            .lines()
            .enumerate()
            .filter_map(|(index, line)| match line {
                Ok(line) if line.trim().is_empty() => None,
                Ok(line) => Some(Ok(line)),
                Err(error) => Some(Err(format!(
                    "{} line {}: {error}",
                    path.display(),
                    index + 1
                ))),
            })
            .collect()
    }

    fn parse_args() -> Result<Args, String> {
        let mut cache_dir = None;
        let mut device = Device::Metal;
        let mut threads = None;
        let mut n_ctx = None;
        let mut authored_output = None;
        let mut perturbations_output = None;
        let mut report = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--cache-dir" => cache_dir = Some(PathBuf::from(next(&mut arguments, &argument)?)),
                "--device" => {
                    device = match next(&mut arguments, &argument)?.as_str() {
                        "cpu" => Device::Cpu,
                        "metal" => Device::Metal,
                        "cuda" => Device::Cuda,
                        value => return Err(format!("invalid --device {value:?}")),
                    };
                }
                "--threads" => threads = Some(parse_u32(next(&mut arguments, &argument)?, "threads")?),
                "--n-ctx" => n_ctx = Some(parse_u32(next(&mut arguments, &argument)?, "n-ctx")?),
                "--authored-output" => authored_output = Some(PathBuf::from(next(&mut arguments, &argument)?)),
                "--perturbations-output" => perturbations_output = Some(PathBuf::from(next(&mut arguments, &argument)?)),
                "--report" => report = Some(PathBuf::from(next(&mut arguments, &argument)?)),
                "--help" => return Err("usage: m3_parity --authored-output NEW.jsonl --perturbations-output NEW.jsonl --report NEW.json [--cache-dir PATH] [--device metal|cpu|cuda] [--threads N] [--n-ctx N]".to_owned()),
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }
        Ok(Args {
            cache_dir,
            device,
            threads,
            n_ctx,
            authored_output: authored_output.ok_or("--authored-output is required")?,
            perturbations_output: perturbations_output
                .ok_or("--perturbations-output is required")?,
            report: report.ok_or("--report is required")?,
        })
    }

    fn next(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))
    }

    fn parse_u32(value: String, name: &str) -> Result<u32, String> {
        let value = u32::from_str(&value).map_err(|error| format!("invalid --{name}: {error}"))?;
        if value == 0 {
            return Err(format!("--{name} must be positive"));
        }
        Ok(value)
    }

    fn write_stdout<T: Serialize>(value: &T) -> Result<(), String> {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        serde_json::to_writer(&mut lock, value).map_err(|error| error.to_string())?;
        lock.write_all(b"\n").map_err(|error| error.to_string())
    }

    fn emit_error(error: &str, code: i32) -> i32 {
        let row = ErrorOutput {
            schema: "openjev-m3-parity-error-v1",
            status: "failed",
            error: error.to_owned(),
        };
        let _ = write_stdout(&row);
        code
    }
}

#[cfg(feature = "integration")]
fn main() {
    std::process::exit(integration::run());
}

#[cfg(not(feature = "integration"))]
fn main() {
    println!(
        "{{\"schema\":\"openjev-m3-parity-error-v1\",\"status\":\"failed\",\"error\":\"m3_parity requires --features integration plus a native device feature\"}}"
    );
    std::process::exit(2);
}
