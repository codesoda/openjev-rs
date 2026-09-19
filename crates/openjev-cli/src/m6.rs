use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use openjev_core::{
    EvalFixture, EvalReport, ExecutionMode, GoldRow, PerturbationStabilityReport, Prediction,
    browser_ladder_predictions, evaluate, evaluate_perturbation_stability, fixture_decisions,
    fixture_gold,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    CliError,
    args::{ComparisonArg, EvalArgs, FixtureArg, GlobalArgs},
    commands::{self, Adapter},
    load_scorer,
    output::{self, WriteSummary},
};

#[derive(Serialize)]
struct PublishedAggregate {
    model: &'static str,
    authored_mean_family_balanced_accuracy: f64,
    perturbations_mean_family_balanced_accuracy: f64,
    row_predictions_available: bool,
}

#[derive(Serialize)]
struct ComparisonReport {
    source: &'static str,
    row_prediction_model: &'static str,
    row_prediction_sha256: &'static str,
    selected_fixture: EvalReport,
    stability: PerturbationStabilityReport,
    published_bfloat16_aggregates: Vec<PublishedAggregate>,
    limitations: Vec<&'static str>,
}

#[derive(Serialize)]
struct EvalCommandReport {
    schema: &'static str,
    fixture: &'static str,
    fixture_sha256: &'static str,
    prediction_source: String,
    prediction_sha256: String,
    model: Option<Value>,
    execution: Option<Value>,
    quality: EvalReport,
    stability: Option<PerturbationStabilityReport>,
    comparison: Option<ComparisonReport>,
    limitations: Vec<&'static str>,
}

pub fn execute_eval<W: Write, E: Write>(
    global: &GlobalArgs,
    args: EvalArgs,
    pretty: bool,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    commands::reject_unimplemented_postprocessing(global)?;
    if global.require_shared {
        return Err(CliError::validation(
            "--require-shared is not valid for eval; M6 inference is direct",
        ));
    }
    if args.predictions.is_some() && args.predictions_output.is_some() {
        return Err(CliError::validation(
            "--predictions-output is only valid when eval performs inference",
        ));
    }
    if args.predictions.is_none() && args.predictions_output.is_none() {
        return Err(CliError::validation(
            "inference-backed eval requires create-only --predictions-output raw evidence",
        ));
    }
    preflight_eval_outputs(&args)?;

    let fixture = match args.fixture {
        FixtureArg::Authored144 => EvalFixture::Authored144,
        FixtureArg::Perturbations108 => EvalFixture::Perturbations108,
    };
    let gold = fixture_gold(fixture).map_err(CliError::from_core_validation)?;
    let authored =
        fixture_gold(EvalFixture::Authored144).map_err(CliError::from_core_validation)?;
    let perturbations =
        fixture_gold(EvalFixture::Perturbations108).map_err(CliError::from_core_validation)?;
    let original_ids = authored_original_ids(&authored);
    let selected_ids: HashSet<_> = gold.iter().map(|row| row.id.as_str()).collect();
    let imported = args
        .predictions
        .as_deref()
        .map(read_predictions)
        .transpose()?;
    if let Some(imported) = &imported {
        validate_prediction_scope(&imported.predictions, &selected_ids, &original_ids, fixture)?;
        if fixture == EvalFixture::Perturbations108 {
            ensure_all_baselines(&imported.predictions, &original_ids)?;
        }
    }
    let inference_config = if imported.is_none() {
        Some(commands::scoring_config(global)?)
    } else {
        None
    };

    let mut report_file = args
        .output
        .as_deref()
        .map(output::create_jsonl_new)
        .transpose()?;
    let mut raw_file = args
        .predictions_output
        .as_deref()
        .map(output::create_jsonl_new)
        .transpose()?;

    let (all_predictions, prediction_sha256, model, execution, source) =
        if let (Some(path), Some(imported)) = (args.predictions.as_deref(), imported) {
            (
                imported.predictions,
                sha256(&imported.bytes),
                imported.model,
                imported.execution,
                format!("file:{}", path.display()),
            )
        } else {
            let path = args
                .predictions_output
                .as_deref()
                .expect("validated raw output");
            let generated = generate_predictions(
                inference_config
                    .as_ref()
                    .expect("inference config validated"),
                fixture,
                &authored,
                raw_file.as_mut().expect("raw file reserved"),
                path,
                stderr,
            )?;
            (
                generated.predictions,
                generated.sha256,
                generated.model,
                generated.execution,
                "openjev-direct-inference".to_owned(),
            )
        };

    let selected_predictions = filter_predictions(&all_predictions, &selected_ids);
    let mut quality =
        evaluate(&gold, &selected_predictions).map_err(CliError::from_core_validation)?;
    quality.fixture_sha256 = Some(fixture.sha256().to_owned());
    quality.model_sha256 = model
        .as_ref()
        .and_then(|value| value.get("artifact_sha256"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    quality.config_sha256 = execution
        .as_ref()
        .map(|value| sha256(&serde_json::to_vec(value).expect("value serializes")));
    quality.limitations.push(
        "M6 reports exact point estimates only; bootstrap and calibration belong to later milestones."
            .to_owned(),
    );

    let stability = if fixture == EvalFixture::Perturbations108 {
        ensure_all_baselines(&all_predictions, &original_ids)?;
        Some(
            evaluate_perturbation_stability(&authored, &perturbations, &all_predictions)
                .map_err(CliError::from_core_validation)?,
        )
    } else {
        None
    };
    let comparison = args
        .compare_to
        .map(|comparison| comparison_report(comparison, &gold, &authored, &perturbations))
        .transpose()?;

    let report = EvalCommandReport {
        schema: "openjev-eval-command-v1",
        fixture: fixture.name(),
        fixture_sha256: fixture.sha256(),
        prediction_source: source,
        prediction_sha256,
        model,
        execution,
        quality,
        stability,
        comparison,
        limitations: vec![
            "Probabilities are conditional over the supplied finite option set.",
            "A quantized-vs-BF16 numerical gap can include backend and floating-point differences; it is not attributable solely to quantization.",
            "Published BF16 row logs are embedded only for qwen3-0.6b; MiniCPM and Qwen3.5 comparison is aggregate-only.",
        ],
    };

    if let (Some(file), Some(path)) = (raw_file.as_ref(), args.predictions_output.as_deref()) {
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
                written: report.quality.scored,
                failed: report.quality.invalid + report.quality.missing,
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

struct ImportedPredictions {
    predictions: Vec<Prediction>,
    bytes: Vec<u8>,
    model: Option<Value>,
    execution: Option<Value>,
}

fn read_predictions(path: &Path) -> Result<ImportedPredictions, CliError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CliError::validation(format!(
            "cannot read predictions {}: {error}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(bytes.as_slice());
    let mut predictions = Vec::new();
    let mut model = None;
    let mut execution = None;
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| CliError::validation(error.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            CliError::validation(format!(
                "predictions line {} is invalid JSON: {error}",
                index + 1
            ))
        })?;
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                CliError::validation(format!(
                    "predictions line {} requires a nonempty string id",
                    index + 1
                ))
            })?
            .to_owned();
        let prediction =
            serde_json::from_value::<Prediction>(value.clone()).unwrap_or_else(|error| {
                Prediction {
                    id,
                    probabilities: None,
                    option_ids: None,
                    prediction_id: None,
                    parse_status: Some("unparsed".to_owned()),
                    error: Some(json!({"code":"invalid_prediction","message":error.to_string()})),
                }
            });
        if model.is_none() {
            model = value.get("model").cloned();
            execution = value.get("execution").cloned();
        }
        predictions.push(prediction);
    }
    Ok(ImportedPredictions {
        predictions,
        bytes,
        model,
        execution,
    })
}

struct GeneratedPredictions {
    predictions: Vec<Prediction>,
    sha256: String,
    model: Option<Value>,
    execution: Option<Value>,
}

fn generate_predictions<E: Write>(
    config: &commands::ScoringConfig,
    fixture: EvalFixture,
    authored: &[GoldRow],
    raw: &mut File,
    raw_path: &Path,
    stderr: &mut E,
) -> Result<GeneratedPredictions, CliError> {
    let mut decisions = fixture_decisions(fixture).map_err(CliError::from_core_validation)?;
    if fixture == EvalFixture::Perturbations108 {
        let authored_decisions =
            fixture_decisions(EvalFixture::Authored144).map_err(CliError::from_core_validation)?;
        let originals = authored_original_ids(authored);
        decisions.extend(
            authored_decisions
                .into_iter()
                .filter(|decision| originals.contains(decision.id.as_str())),
        );
    }
    let mut scorer = load_scorer(config)?;
    let mut predictions = Vec::with_capacity(decisions.len());
    let mut hasher = Sha256::new();
    let mut model = None;
    let mut execution = None;
    for decision in decisions {
        let id = decision.id.clone();
        let row = match commands::score_item_with_reason(
            scorer.as_mut(),
            &Adapter::Choice(decision),
            ExecutionMode::Direct,
            config.confidence,
            None,
            None,
        ) {
            Ok(readout) => {
                let value = serde_json::to_value(&readout).expect("readout serializes");
                if model.is_none() {
                    model = value.get("model").cloned();
                    execution = value.get("execution").cloned();
                }
                predictions.push(Prediction {
                    id: readout.id.clone(),
                    probabilities: Some(openjev_core::ProbabilityInput::Array(
                        readout.probabilities.clone(),
                    )),
                    option_ids: Some(readout.option_ids.clone()),
                    prediction_id: Some(readout.choice.clone()),
                    parse_status: Some("parsed".to_owned()),
                    error: None,
                });
                value
            }
            Err(error) => {
                let record = error.with_id(id.clone()).unparsed().record();
                predictions.push(Prediction {
                    id,
                    probabilities: None,
                    option_ids: None,
                    prediction_id: None,
                    parse_status: Some("unparsed".to_owned()),
                    error: Some(serde_json::to_value(&record).expect("error serializes")),
                });
                serde_json::to_value(record).expect("error serializes")
            }
        };
        let mut bytes = serde_json::to_vec(&row).expect("row serializes");
        bytes.push(b'\n');
        raw.write_all(&bytes).map_err(|error| {
            CliError::runtime(
                "output_io",
                format!("write raw predictions {}: {error}", raw_path.display()),
            )
        })?;
        raw.flush()
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
        hasher.update(bytes);
    }
    scorer.shutdown()?;
    let _ = writeln!(
        stderr,
        "eval: wrote {} raw prediction rows",
        predictions.len()
    );
    let digest = hasher.finalize();
    Ok(GeneratedPredictions {
        predictions,
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        model,
        execution,
    })
}

fn comparison_report(
    comparison: ComparisonArg,
    gold: &[GoldRow],
    authored: &[GoldRow],
    perturbations: &[GoldRow],
) -> Result<ComparisonReport, CliError> {
    match comparison {
        ComparisonArg::BrowserLadder => {
            let predictions =
                browser_ladder_predictions().map_err(CliError::from_core_validation)?;
            let ids: HashSet<_> = gold.iter().map(|row| row.id.as_str()).collect();
            let selected = filter_predictions(&predictions, &ids);
            let mut selected_fixture =
                evaluate(gold, &selected).map_err(CliError::from_core_validation)?;
            selected_fixture.fixture_sha256 = Some(
                match gold.len() {
                    144 => openjev_core::AUTHORED144_SHA256,
                    108 => openjev_core::PERTURBATIONS108_SHA256,
                    _ => unreachable!("comparison uses one embedded fixture"),
                }
                .to_owned(),
            );
            let original_ids = authored_original_ids(authored);
            let perturbation_ids: HashSet<_> =
                perturbations.iter().map(|row| row.id.as_str()).collect();
            let stability_predictions: Vec<_> = predictions
                .iter()
                .filter(|prediction| {
                    original_ids.contains(prediction.id.as_str())
                        || perturbation_ids.contains(prediction.id.as_str())
                })
                .cloned()
                .collect();
            let stability =
                evaluate_perturbation_stability(authored, perturbations, &stability_predictions)
                    .map_err(CliError::from_core_validation)?;
            Ok(ComparisonReport {
                source: "SemIf browser ladder frozen reference",
                row_prediction_model: "qwen3-0.6b",
                row_prediction_sha256: openjev_core::BROWSER_LADDER_QWEN3_SHA256,
                selected_fixture,
                stability,
                published_bfloat16_aggregates: published_aggregates(),
                limitations: vec![
                    "The 252 embedded row predictions belong only to qwen3-0.6b.",
                    "Other BF16 models are represented only by published aggregate balanced accuracies.",
                ],
            })
        }
    }
}

fn published_aggregates() -> Vec<PublishedAggregate> {
    vec![
        PublishedAggregate {
            model: "qwen3-0.6b",
            authored_mean_family_balanced_accuracy: 0.44035251527511593,
            perturbations_mean_family_balanced_accuracy: 0.5276895943562611,
            row_predictions_available: true,
        },
        PublishedAggregate {
            model: "minicpm5-2b",
            authored_mean_family_balanced_accuracy: 0.6862540337772537,
            perturbations_mean_family_balanced_accuracy: 0.6925925925925925,
            row_predictions_available: false,
        },
        PublishedAggregate {
            model: "qwen3.5-4b",
            authored_mean_family_balanced_accuracy: 0.8132381607613807,
            perturbations_mean_family_balanced_accuracy: 0.7657848324514992,
            row_predictions_available: false,
        },
    ]
}

fn authored_original_ids(rows: &[GoldRow]) -> HashSet<&str> {
    rows.iter()
        .filter(|row| {
            row.metadata
                .get("provenance")
                .and_then(Value::as_object)
                .and_then(|value| value.get("variant"))
                .and_then(Value::as_str)
                == Some("original")
        })
        .map(|row| row.id.as_str())
        .collect()
}

fn filter_predictions(predictions: &[Prediction], ids: &HashSet<&str>) -> Vec<Prediction> {
    predictions
        .iter()
        .filter(|prediction| ids.contains(prediction.id.as_str()))
        .cloned()
        .collect()
}

fn validate_prediction_scope(
    predictions: &[Prediction],
    fixture_ids: &HashSet<&str>,
    originals: &HashSet<&str>,
    fixture: EvalFixture,
) -> Result<(), CliError> {
    let mut seen = HashSet::new();
    for prediction in predictions {
        let allowed = fixture_ids.contains(prediction.id.as_str())
            || (fixture == EvalFixture::Perturbations108
                && originals.contains(prediction.id.as_str()));
        if !allowed {
            return Err(CliError::validation(format!(
                "unknown prediction ID {:?} for fixture {}",
                prediction.id,
                fixture.name()
            )));
        }
        if !seen.insert(prediction.id.as_str()) {
            return Err(CliError::validation(format!(
                "duplicate prediction ID {:?}",
                prediction.id
            )));
        }
    }
    Ok(())
}

fn ensure_all_baselines(
    predictions: &[Prediction],
    originals: &HashSet<&str>,
) -> Result<(), CliError> {
    let ids: HashSet<_> = predictions
        .iter()
        .map(|prediction| prediction.id.as_str())
        .collect();
    let missing = originals.iter().filter(|id| !ids.contains(**id)).count();
    if missing != 0 {
        return Err(CliError::validation(format!(
            "perturbations108 requires predictions for all {} authored originals; {missing} are missing",
            originals.len()
        )));
    }
    Ok(())
}

fn preflight_eval_outputs(args: &EvalArgs) -> Result<(), CliError> {
    if let Some(path) = args.output.as_deref() {
        commands::preflight_output(path, args.predictions.as_deref())?;
    }
    if let Some(path) = args.predictions_output.as_deref() {
        commands::preflight_output(path, args.predictions.as_deref())?;
    }
    if let (Some(left), Some(right)) = (&args.output, &args.predictions_output)
        && normalized_target(left)? == normalized_target(right)?
    {
        return Err(CliError::validation(
            "--output and --predictions-output must identify different files",
        ));
    }
    Ok(())
}

pub(crate) fn normalized_target(path: &Path) -> Result<PathBuf, CliError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        CliError::validation(format!("cannot resolve {}: {error}", parent.display()))
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| CliError::validation("output path has no filename"))?;
    Ok(parent.join(name))
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
