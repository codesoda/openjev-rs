use std::collections::HashMap;

use openjev_core::{
    Decision, DecisionOption, ExecutionMode, OpenJevError, Readout, StateValue, first_argmax,
    normalized_margin, parse_json_strict, python_json_dumps,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{CliError, commands::Adapter};

const DEFAULT_INSTRUCTIONS: &str = "Select the best option.";
const MAX_QUESTIONS: usize = 64;
const MAX_OPTIONS: usize = 16;
const MAX_EXPANDED_STATE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterError {
    pub error_type: &'static str,
    pub message: String,
}

impl AdapterError {
    fn bad_json(message: impl Into<String>) -> Self {
        Self {
            error_type: "invalid_json",
            message: message.into(),
        }
    }

    fn validation(message: impl Into<String>) -> Self {
        Self {
            error_type: "validation_error",
            message: message.into(),
        }
    }
}

#[derive(Debug)]
pub struct PreparedRequest {
    pub requested_model: Option<String>,
    pub entries: Vec<PreparedEntry>,
    pub inference: Vec<Adapter>,
}

#[derive(Debug)]
pub struct PreparedEntry {
    external_id: String,
    internal_id: Option<String>,
    projection: Projection,
}

#[derive(Debug)]
enum Projection {
    Choice { labels: Vec<String> },
    Noul,
    Score { legend: Vec<Value> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionDisclosure {
    pub requested: String,
    pub effective: String,
    pub fallback: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

pub fn parse_request(bytes: &[u8]) -> Result<PreparedRequest, AdapterError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| AdapterError::bad_json("request body must be valid UTF-8 JSON"))?;
    let value = parse_json_strict(text).map_err(classify_strict_json_error)?;
    let mut request = into_object(value, "request")?;

    let requested_model = match request.shift_remove("model") {
        None => None,
        Some(Value::String(model)) if !model.is_empty() => Some(model),
        Some(_) => {
            return Err(AdapterError::validation(
                "model must be a nonempty string when provided",
            ));
        }
    };
    let state = request
        .shift_remove("state")
        .ok_or_else(|| AdapterError::validation("state is required"))?;
    require_entry_type(&state, "state")?;
    let state = adapt_state(state)?;
    let questions = request
        .shift_remove("questions")
        .ok_or_else(|| AdapterError::validation("questions is required"))?;
    reject_unknown(&request, "request")?;
    let questions = into_object(questions, "questions")?;
    if questions.is_empty() {
        return Err(AdapterError::validation(
            "questions must contain at least one question",
        ));
    }
    if questions.len() > MAX_QUESTIONS {
        return Err(AdapterError::validation(format!(
            "questions must not contain more than {MAX_QUESTIONS} entries"
        )));
    }
    let state_bytes = python_json_dumps(state.as_value())
        .map_err(|error| AdapterError::validation(format!("state cannot be measured: {error}")))?
        .len();
    // Every question currently owns a StateValue clone. Count singleton Choice
    // questions too so the admission budget conservatively bounds replication.
    let expanded_state_bytes = state_bytes.checked_mul(questions.len()).ok_or_else(|| {
        AdapterError::validation("expanded state replication size exceeds the server limit")
    })?;
    if expanded_state_bytes > MAX_EXPANDED_STATE_BYTES {
        return Err(AdapterError::validation(format!(
            "state replicated across questions exceeds the {MAX_EXPANDED_STATE_BYTES}-byte limit"
        )));
    }

    let mut entries = Vec::with_capacity(questions.len());
    let mut inference = Vec::with_capacity(questions.len());
    for (index, (external_id, question)) in questions.into_iter().enumerate() {
        let internal_id = format!("jev-question-{}", index + 1);
        let (projection, adapter) = parse_question(&external_id, &internal_id, &state, question)?;
        let inferred_id = adapter.as_ref().map(|_| internal_id);
        if let Some(adapter) = adapter {
            inference.push(adapter);
        }
        entries.push(PreparedEntry {
            external_id,
            internal_id: inferred_id,
            projection,
        });
    }

    Ok(PreparedRequest {
        requested_model,
        entries,
        inference,
    })
}

fn parse_question(
    external_id: &str,
    internal_id: &str,
    state: &StateValue,
    value: Value,
) -> Result<(Projection, Option<Adapter>), AdapterError> {
    let path = format!("questions.{external_id:?}");
    let mut question = into_object(value, &path)?;
    let kind = match question.shift_remove("type") {
        Some(Value::String(kind)) => kind,
        Some(_) => {
            return Err(AdapterError::validation(format!(
                "{path}.type must be a string"
            )));
        }
        None => return Err(AdapterError::validation(format!("{path}.type is required"))),
    };
    let instructions = match question.shift_remove("instructions") {
        Some(value) => render_instructions(&value, &format!("{path}.instructions"))?,
        None => DEFAULT_INSTRUCTIONS.to_owned(),
    };
    let criteria = question.shift_remove("criteria");
    reject_unknown(&question, &path)?;

    match kind.as_str() {
        "choice" => parse_choice(path, internal_id, state, instructions, criteria),
        "noul" => parse_noul(path, internal_id, state, instructions, criteria),
        "score" => parse_score(path, internal_id, state, instructions, criteria),
        _ => Err(AdapterError::validation(format!(
            "{path}.type must be choice, noul, or score"
        ))),
    }
}

fn parse_choice(
    path: String,
    internal_id: &str,
    state: &StateValue,
    instructions: String,
    criteria: Option<Value>,
) -> Result<(Projection, Option<Adapter>), AdapterError> {
    let criteria = criteria.ok_or_else(|| {
        AdapterError::validation(format!("{path}.criteria is required for choice"))
    })?;
    let criteria = into_object(criteria, &format!("{path}.criteria"))?;
    if criteria.is_empty() || criteria.len() > MAX_OPTIONS {
        return Err(AdapterError::validation(format!(
            "{path}.criteria must contain 1-{MAX_OPTIONS} options"
        )));
    }
    let mut labels = Vec::with_capacity(criteria.len());
    let mut options = Vec::with_capacity(criteria.len());
    for (label, description) in criteria {
        let rendered = render_description(
            &description,
            &format!("{path}.criteria.{label:?}"),
            Some(&label),
        )?;
        labels.push(label.clone());
        options.push(DecisionOption {
            id: label,
            description: rendered,
        });
    }
    let projection = Projection::Choice { labels };
    if options.len() == 1 {
        return Ok((projection, None));
    }
    let decision = Decision::new(internal_id, state.clone(), instructions, options)
        .map_err(core_validation)?;
    Ok((projection, Some(Adapter::Choice(decision))))
}

fn parse_noul(
    path: String,
    internal_id: &str,
    state: &StateValue,
    instructions: String,
    criteria: Option<Value>,
) -> Result<(Projection, Option<Adapter>), AdapterError> {
    let mut yes = "Yes".to_owned();
    let mut no = "No".to_owned();
    if let Some(criteria) = criteria
        && !criteria.is_null()
    {
        let mut criteria = into_object(criteria, &format!("{path}.criteria"))?;
        if let Some(value) = criteria.shift_remove("true") {
            yes = render_description(&value, &format!("{path}.criteria.true"), None)?;
        }
        if let Some(value) = criteria.shift_remove("false") {
            no = render_description(&value, &format!("{path}.criteria.false"), None)?;
        }
        reject_unknown(&criteria, &format!("{path}.criteria"))?;
    }
    let decision = Decision::new(
        internal_id,
        state.clone(),
        instructions,
        vec![
            DecisionOption {
                id: "yes".to_owned(),
                description: yes,
            },
            DecisionOption {
                id: "no".to_owned(),
                description: no,
            },
        ],
    )
    .map_err(core_validation)?;
    Ok((Projection::Noul, Some(Adapter::Choice(decision))))
}

fn parse_score(
    path: String,
    internal_id: &str,
    state: &StateValue,
    instructions: String,
    criteria: Option<Value>,
) -> Result<(Projection, Option<Adapter>), AdapterError> {
    let criteria = match criteria {
        Some(Value::Array(criteria)) => criteria,
        Some(_) => {
            return Err(AdapterError::validation(format!(
                "{path}.criteria must be an array for score"
            )));
        }
        None => {
            return Err(AdapterError::validation(format!(
                "{path}.criteria is required for score"
            )));
        }
    };
    if !(2..=MAX_OPTIONS).contains(&criteria.len()) {
        return Err(AdapterError::validation(format!(
            "{path}.criteria must contain 2-{MAX_OPTIONS} levels"
        )));
    }
    let mut options = Vec::with_capacity(criteria.len());
    for (index, description) in criteria.iter().enumerate() {
        options.push(DecisionOption {
            id: index.to_string(),
            description: render_description(
                description,
                &format!("{path}.criteria[{index}]"),
                None,
            )?,
        });
    }
    let decision = Decision::new(internal_id, state.clone(), instructions, options)
        .map_err(core_validation)?;
    Ok((
        Projection::Score { legend: criteria },
        Some(Adapter::Choice(decision)),
    ))
}

fn render_instructions(value: &Value, path: &str) -> Result<String, AdapterError> {
    require_entry_type(value, path)?;
    if is_empty_entry(value) {
        return Ok(DEFAULT_INSTRUCTIONS.to_owned());
    }
    render_entry(value, path)
}

fn render_description(
    value: &Value,
    path: &str,
    null_fallback: Option<&str>,
) -> Result<String, AdapterError> {
    require_entry_type(value, path)?;
    if value.is_null()
        && let Some(fallback) = null_fallback
    {
        return Ok(fallback.to_owned());
    }
    render_entry(value, path)
}

fn render_entry(value: &Value, path: &str) -> Result<String, AdapterError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Object(_) | Value::Array(_) | Value::Null => {
            python_json_dumps(value).map_err(|error| {
                AdapterError::validation(format!("{path} cannot be rendered: {error}"))
            })
        }
        _ => Err(AdapterError::validation(format!(
            "{path} must be a string, object, array, or null"
        ))),
    }
}

fn adapt_state(value: Value) -> Result<StateValue, AdapterError> {
    if let Ok(state) = StateValue::try_from(value.clone()) {
        return Ok(state);
    }
    // The SDK admits null and empty entries while the existing CLI/core state
    // contract deliberately requires a nonempty top-level container. Preserve
    // the original JSON type under one explicit adapter envelope rather than
    // stringifying structured state.
    let mut envelope = Map::new();
    envelope.insert("value".to_owned(), value);
    StateValue::try_from(Value::Object(envelope)).map_err(core_validation)
}

fn require_entry_type(value: &Value, path: &str) -> Result<(), AdapterError> {
    if matches!(
        value,
        Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null
    ) {
        Ok(())
    } else {
        Err(AdapterError::validation(format!(
            "{path} must be a string, object, array, or null"
        )))
    }
}

fn is_empty_entry(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
        Value::Bool(_) | Value::Number(_) => false,
    }
}

fn into_object(value: Value, path: &str) -> Result<Map<String, Value>, AdapterError> {
    match value {
        Value::Object(value) => Ok(value),
        _ => Err(AdapterError::validation(format!(
            "{path} must be a JSON object"
        ))),
    }
}

fn reject_unknown(fields: &Map<String, Value>, path: &str) -> Result<(), AdapterError> {
    if let Some(field) = fields.keys().next() {
        Err(AdapterError::validation(format!(
            "unsupported field {field:?} in {path}"
        )))
    } else {
        Ok(())
    }
}

fn classify_strict_json_error(error: OpenJevError) -> AdapterError {
    let is_supported_json_with_unsupported_semantics = matches!(
        &error,
        OpenJevError::Serialization { message, .. }
            if message.starts_with("floating-point number ")
                || message.starts_with("integer ")
                || message.starts_with("duplicate JSON key ")
                || message.starts_with("JSON nesting exceeds maximum depth ")
    );
    if is_supported_json_with_unsupported_semantics {
        AdapterError::validation(error.to_string())
    } else {
        AdapterError::bad_json(error.to_string())
    }
}

fn core_validation(error: OpenJevError) -> AdapterError {
    AdapterError::validation(error.to_string())
}

pub fn project_response(
    prepared: PreparedRequest,
    readouts: Vec<Readout>,
    model: String,
) -> Result<(SystemOneResponse, ExecutionDisclosure), CliError> {
    let expected_rows = prepared
        .entries
        .iter()
        .filter(|entry| entry.internal_id.is_some())
        .count();
    if readouts.len() != expected_rows {
        return Err(CliError::runtime(
            "worker",
            "resident scorer returned an unexpected number of rows",
        ));
    }
    let by_id: HashMap<_, _> = readouts
        .into_iter()
        .map(|readout| (readout.id.clone(), readout))
        .collect();
    if by_id.len() != expected_rows {
        return Err(CliError::runtime(
            "worker",
            "resident scorer returned duplicate row identities",
        ));
    }

    let mut input_tokens = 0_u64;
    let mut answers = Map::new();
    let mut modes = Vec::new();
    for entry in prepared.entries {
        let answer = if let Some(internal_id) = entry.internal_id {
            let readout = by_id.get(&internal_id).ok_or_else(|| {
                CliError::runtime("worker", "resident scorer result identity mismatch")
            })?;
            input_tokens = input_tokens.saturating_add(readout.input_tokens);
            modes.push((
                readout.execution.requested_mode,
                readout.execution.effective_mode,
                readout.execution.fallback_reason.is_some(),
            ));
            project_readout(&entry.projection, readout)?
        } else {
            project_singleton(&entry.projection)?
        };
        answers.insert(entry.external_id, answer);
    }
    let disclosure = disclose_execution(&modes);
    Ok((
        SystemOneResponse {
            model,
            answers,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        },
        disclosure,
    ))
}

fn project_singleton(projection: &Projection) -> Result<Value, CliError> {
    let Projection::Choice { labels } = projection else {
        return Err(CliError::runtime(
            "worker",
            "only Choice supports deterministic singleton projection",
        ));
    };
    let label = labels
        .first()
        .ok_or_else(|| CliError::runtime("worker", "singleton Choice has no label"))?;
    let mut probabilities = Map::new();
    probabilities.insert(label.clone(), Value::from(1.0));
    let mut answer = Map::new();
    answer.insert("type".to_owned(), Value::String("choice".to_owned()));
    answer.insert("choice".to_owned(), Value::String(label.clone()));
    answer.insert("confidence".to_owned(), Value::from(1.0));
    answer.insert("probabilities".to_owned(), Value::Object(probabilities));
    Ok(Value::Object(answer))
}

fn project_readout(projection: &Projection, readout: &Readout) -> Result<Value, CliError> {
    match projection {
        Projection::Choice { labels } => {
            ensure_probability_alignment(labels, readout)?;
            let choice_index =
                first_argmax(&readout.probabilities).map_err(CliError::from_runtime_core)?;
            let confidence =
                normalized_margin(&readout.probabilities).map_err(CliError::from_runtime_core)?;
            let mut probabilities = Map::new();
            for (label, probability) in labels.iter().zip(&readout.probabilities) {
                probabilities.insert(label.clone(), Value::from(round_wire(*probability)));
            }
            let mut answer = Map::new();
            answer.insert("type".to_owned(), Value::String("choice".to_owned()));
            answer.insert(
                "choice".to_owned(),
                Value::String(labels[choice_index].clone()),
            );
            answer.insert("confidence".to_owned(), Value::from(round_wire(confidence)));
            answer.insert("probabilities".to_owned(), Value::Object(probabilities));
            Ok(Value::Object(answer))
        }
        Projection::Noul => {
            ensure_probability_alignment(&["yes".to_owned(), "no".to_owned()], readout)?;
            let mut answer = Map::new();
            answer.insert("type".to_owned(), Value::String("noul".to_owned()));
            answer.insert(
                "noul".to_owned(),
                Value::from(round_wire(readout.probabilities[0])),
            );
            Ok(Value::Object(answer))
        }
        Projection::Score { legend } => {
            let expected_ids: Vec<_> = (0..legend.len()).map(|index| index.to_string()).collect();
            ensure_probability_alignment(&expected_ids, readout)?;
            let confidence =
                normalized_margin(&readout.probabilities).map_err(CliError::from_runtime_core)?;
            let score: f64 = readout
                .probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let mut legend_object = Map::new();
            let mut probabilities = Map::new();
            for (index, (description, probability)) in
                legend.iter().zip(&readout.probabilities).enumerate()
            {
                legend_object.insert(index.to_string(), description.clone());
                probabilities.insert(index.to_string(), Value::from(round_wire(*probability)));
            }
            let mut answer = Map::new();
            answer.insert("type".to_owned(), Value::String("score".to_owned()));
            answer.insert("score".to_owned(), Value::from(round_wire(score)));
            answer.insert("confidence".to_owned(), Value::from(round_wire(confidence)));
            answer.insert("legend".to_owned(), Value::Object(legend_object));
            answer.insert("probabilities".to_owned(), Value::Object(probabilities));
            Ok(Value::Object(answer))
        }
    }
}

fn ensure_probability_alignment(
    expected_ids: &[String],
    readout: &Readout,
) -> Result<(), CliError> {
    if readout.probabilities.len() != expected_ids.len() || readout.option_ids != expected_ids {
        return Err(CliError::runtime(
            "worker",
            "resident scorer returned probabilities with mismatched option identities",
        ));
    }
    Ok(())
}

fn round_wire(value: f64) -> f64 {
    let rounded = (value * 100.0).round() / 100.0;
    if rounded == 0.0 { 0.0 } else { rounded }
}

fn disclose_execution(modes: &[(ExecutionMode, ExecutionMode, bool)]) -> ExecutionDisclosure {
    if modes.is_empty() {
        return ExecutionDisclosure {
            requested: "deterministic".to_owned(),
            effective: "deterministic".to_owned(),
            fallback: None,
        };
    }
    let requested = common_mode(modes.iter().map(|mode| mode.0));
    let effective = common_mode(modes.iter().map(|mode| mode.1));
    ExecutionDisclosure {
        requested,
        effective,
        fallback: modes
            .iter()
            .any(|mode| mode.2)
            .then_some("shared unavailable; serial full-prompt fallback"),
    }
}

fn common_mode(mut modes: impl Iterator<Item = ExecutionMode>) -> String {
    let first = modes.next().expect("nonempty execution disclosure");
    if modes.all(|mode| mode == first) {
        mode_name(first).to_owned()
    } else {
        "mixed".to_owned()
    }
}

const fn mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Direct => "direct",
        ExecutionMode::Serial => "serial",
        ExecutionMode::Shared => "shared",
        ExecutionMode::Batch => "batch",
    }
}

#[cfg(test)]
mod tests {
    use openjev_core::{
        Device, ExecutionMetadata, GpuLayersRequested, GpuLayersStatus, Integrity, ModelMetadata,
        Primitive, PromptProfile, TemplateMetadataStatus, standard_limitations,
    };

    use super::*;

    fn readout(id: &str, option_ids: &[&str], probabilities: Vec<f64>) -> Readout {
        let option_ids: Vec<_> = option_ids.iter().map(|id| (*id).to_owned()).collect();
        let choice_index = first_argmax(&probabilities).unwrap();
        Readout {
            schema: "openjev-readout-v1".to_owned(),
            id: id.to_owned(),
            primitive: Primitive::Choice,
            choice: option_ids[choice_index].clone(),
            choice_index,
            option_ids,
            probabilities,
            option_logits: vec![0.0; 3],
            answer_token_ids: vec![1, 2, 3],
            allowed_token_mass: 0.5,
            full_vocab_argmax_id: 1,
            full_vocab_log_normalizer: 1.0,
            input_tokens: 10,
            forward_seconds: None,
            total_seconds: None,
            prompt_sha256: "0".repeat(64),
            prompt_version: "direct-options-v1".to_owned(),
            model: ModelMetadata {
                id: "test".to_owned(),
                source: "test".to_owned(),
                revision: "test".to_owned(),
                file: "test".to_owned(),
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
            limitations: standard_limitations(),
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
                n_seq_max: 4,
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

    #[test]
    fn parses_all_types_and_preserves_external_order_and_values() {
        let request = parse_request(br#"{
          "model":"jev-latest","state":{"z":1,"a":null},"questions":{
            "\u96ea":{"type":"choice","instructions":{"task":"pick"},"criteria":{"b":null,"a":{"why":"A"}}},
            "truth":{"type":"noul","criteria":{"true":{"means":"yes"},"false":"Nope"}},
            "score":{"type":"score","instructions":[],"criteria":[null,{"level":"high"}]}
          }
        }"#).unwrap();
        assert_eq!(request.entries.len(), 3);
        assert_eq!(request.inference.len(), 3);
        assert_eq!(request.entries[0].external_id, "雪");
        assert_eq!(request.inference[0].decision().options[0].id, "b");
        assert_eq!(request.inference[0].decision().options[0].description, "b");
        assert_eq!(
            request.inference[2].decision().question,
            DEFAULT_INSTRUCTIONS
        );
    }

    #[test]
    fn singleton_choice_is_deterministic_and_null_empty_state_are_adapted_without_inference() {
        let request = parse_request(
            br#"{"state":null,"questions":{"only":{"type":"choice","criteria":{"\u03bb":null}}}}"#,
        )
        .unwrap();
        assert!(request.inference.is_empty());
        let (response, disclosure) =
            project_response(request, Vec::new(), "qwen3-0.6b".to_owned()).unwrap();
        assert_eq!(response.answers["only"]["choice"], "λ");
        assert_eq!(response.answers["only"]["confidence"], 1.0);
        assert_eq!(response.usage.input_tokens, 0);
        assert_eq!(disclosure.effective, "deterministic");
    }

    #[test]
    fn projection_uses_unrounded_values_then_rounds_each_wire_number_without_renormalizing() {
        let request = parse_request(
            br#"{
          "state":"s","questions":{
            "c":{"type":"choice","criteria":{"first":"A","second":"B","third":"C"}},
            "n":{"type":"noul"},
            "s":{"type":"score","criteria":["low","mid","high"]}
          }
        }"#,
        )
        .unwrap();
        let rows = vec![
            readout(
                "jev-question-1",
                &["first", "second", "third"],
                vec![1.0 / 3.0; 3],
            ),
            readout("jev-question-2", &["yes", "no"], vec![0.126, 0.874]),
            readout(
                "jev-question-3",
                &["0", "1", "2"],
                vec![0.126, 0.333, 0.541],
            ),
        ];
        let (response, _) = project_response(request, rows, "m".to_owned()).unwrap();
        assert_eq!(response.answers["c"]["choice"], "first");
        assert_eq!(response.answers["c"]["probabilities"]["first"], 0.33);
        assert_eq!(response.answers["c"]["probabilities"]["second"], 0.33);
        assert_eq!(response.answers["c"]["probabilities"]["third"], 0.33);
        assert_eq!(response.answers["n"]["noul"], 0.13);
        assert_eq!(response.answers["s"]["score"], 1.42);
        assert_eq!(response.answers["s"]["legend"]["1"], "mid");
        assert_eq!(response.usage.input_tokens, 30);
    }

    #[test]
    fn empty_external_ids_and_labels_are_mapped_without_entering_internal_ids() {
        let request = parse_request(
            br#"{"state":"s","questions":{"":{"type":"choice","criteria":{"":null,"other":"Other"}}}}"#,
        )
        .unwrap();
        assert_eq!(request.entries[0].external_id, "");
        assert_eq!(request.inference[0].decision().id, "jev-question-1");
        assert_eq!(request.inference[0].decision().options[0].id, "");
        assert_eq!(request.inference[0].decision().options[0].description, "");
    }

    #[test]
    fn rejects_duplicate_float_depth_unknown_and_invalid_counts() {
        for body in [
            br#"{"state":"s","questions":{},"extra":null}"#.as_slice(),
            br#"{"state":"s","questions":{"q":{"type":"choice","criteria":{}}}}"#,
            br#"{"state":"s","questions":{"q":{"type":"score","criteria":["one"]}}}"#,
        ] {
            assert!(parse_request(body).is_err());
        }
        for body in [
            br#"{"state":"s","state":"x","questions":{"q":{"type":"noul"}}}"#.as_slice(),
            br#"{"state":{"x":1.0},"questions":{"q":{"type":"noul"}}}"#,
        ] {
            assert_eq!(
                parse_request(body).unwrap_err().error_type,
                "validation_error"
            );
        }
        assert_eq!(
            parse_request(br#"{"#).unwrap_err().error_type,
            "invalid_json"
        );
        let deep = format!(
            "{{\"state\":{}0{},\"questions\":{{\"q\":{{\"type\":\"noul\"}}}}}}",
            "[".repeat(130),
            "]".repeat(130)
        );
        assert!(parse_request(deep.as_bytes()).is_err());

        let questions = (0..65)
            .map(|index| format!("\"q{index}\":{{\"type\":\"noul\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            parse_request(format!("{{\"state\":\"s\",\"questions\":{{{questions}}}}}").as_bytes())
                .is_err()
        );
        let questions = (0..64)
            .map(|index| format!("\"q{index}\":{{\"type\":\"noul\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            parse_request(format!("{{\"state\":\"s\",\"questions\":{{{questions}}}}}").as_bytes())
                .is_ok()
        );
        let options = (0..17)
            .map(|index| format!("\"o{index}\":null"))
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            parse_request(
                format!("{{\"state\":\"s\",\"questions\":{{\"q\":{{\"type\":\"choice\",\"criteria\":{{{options}}}}}}}}}")
                    .as_bytes(),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_expanded_state_replication_before_per_question_clones() {
        let state = "x".repeat(70_000);
        let questions = (0..64)
            .map(|index| format!("\"q{index}\":{{\"type\":\"noul\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        let body = format!("{{\"state\":\"{state}\",\"questions\":{{{questions}}}}}");
        assert!(body.len() < 1024 * 1024);
        let error = parse_request(body.as_bytes()).unwrap_err();
        assert_eq!(error.error_type, "validation_error");
        assert!(error.message.contains("replicated across questions"));
    }
}
