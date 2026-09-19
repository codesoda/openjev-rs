use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser::Error as _};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::validate::{
    deserialize_strict_json_value, drop_value_iteratively, parse_json_strict, validate_integer_json,
};

pub const DIRECT_READOUT: &str =
    "native full-vocabulary last-position logits restricted to declared answer slots";
pub const PROBABILITY_STATUS: &str =
    "conditional option score; uncalibrated as decision confidence";
pub const FORCED_TYPED_LIMITATION: &str = "A forced typed output can still be semantically wrong.";
pub const CONDITIONAL_PROBABILITY_LIMITATION: &str = "Softmax over allowed tokens is conditional on the supplied alternatives; it is not calibrated operational confidence.";
pub const CONFIDENCE_STATUS: &str = "normalized margin; uncalibrated";

#[must_use]
pub fn standard_limitations() -> Vec<String> {
    vec![
        FORCED_TYPED_LIMITATION.to_owned(),
        CONDITIONAL_PROBABILITY_LIMITATION.to_owned(),
    ]
}

/// Errors exposed by the backend-neutral library.
#[derive(Debug, Error)]
pub enum OpenJevError {
    #[error("validation error at {path}: {message}")]
    Validation { path: String, message: String },
    #[error("serialization error at {path}: {message}")]
    Serialization { path: String, message: String },
    #[error("template error: {0}")]
    Template(String),
    #[error("answer-slot error: {0}")]
    Slot(String),
    #[error("context error: {0}")]
    Context(String),
    #[error("cache error: {0}")]
    Cache(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("parity error: {0}")]
    Parity(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl OpenJevError {
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Validation { .. } => "validation",
            Self::Serialization { .. } => "serialization",
            Self::Template(_) => "template",
            Self::Slot(_) => "slot",
            Self::Context(_) => "context",
            Self::Cache(_) => "cache",
            Self::Model(_) => "model",
            Self::Decode(_) => "decode",
            Self::Parity(_) => "parity",
            Self::Io(_) => "io",
        }
    }
}

pub type Result<T> = std::result::Result<T, OpenJevError>;

/// A validated nonempty top-level string, object, or array.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct StateValue(Value);

impl StateValue {
    pub fn string(value: impl Into<String>) -> Result<Self> {
        Self::try_from(Value::String(value.into()))
    }

    pub fn parse_json(input: &str) -> Result<Self> {
        Self::try_from(parse_json_strict(input)?)
    }

    #[must_use]
    pub const fn as_value(&self) -> &Value {
        &self.0
    }

    #[must_use]
    pub fn into_value(self) -> Value {
        self.0
    }
}

/// Converts an already-built value after bounded recursive validation.
///
/// Duplicate keys must be rejected while parsing; a `Value` whose producer
/// already collapsed duplicates cannot retain or recover that information.
impl TryFrom<Value> for StateValue {
    type Error = OpenJevError;

    fn try_from(value: Value) -> Result<Self> {
        if let Err(error) = validate_integer_json(&value, "$") {
            drop_value_iteratively(value);
            return Err(error);
        }
        match &value {
            Value::String(text) if !text.is_empty() => Ok(Self(value)),
            Value::Array(values) if !values.is_empty() => Ok(Self(value)),
            Value::Object(values) if !values.is_empty() => Ok(Self(value)),
            Value::String(_) | Value::Array(_) | Value::Object(_) => {
                Err(OpenJevError::Validation {
                    path: "$".to_owned(),
                    message: "state must be nonempty".to_owned(),
                })
            }
            _ => Err(OpenJevError::Validation {
                path: "$".to_owned(),
                message: "state must be a string, object, or array".to_owned(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for StateValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = deserialize_strict_json_value(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    pub id: String,
    pub state: StateValue,
    pub question: String,
    pub options: Vec<DecisionOption>,
    #[serde(flatten)]
    metadata: Map<String, Value>,
}

impl Decision {
    pub fn new(
        id: impl Into<String>,
        state: StateValue,
        question: impl Into<String>,
        options: Vec<DecisionOption>,
    ) -> Result<Self> {
        let result = Self {
            id: id.into(),
            state,
            question: question.into(),
            options,
            metadata: Map::new(),
        };
        result.validate()?;
        Ok(result)
    }

    pub fn from_json_str(input: &str) -> Result<Self> {
        Self::from_value(parse_json_strict(input)?)
    }

    fn from_value(value: Value) -> Result<Self> {
        let mut fields = into_object(value, "$")?;
        let id = take_string(&mut fields, "id", "$")?;
        let state = take_required(&mut fields, "state", "$")?;
        let question = take_string(&mut fields, "question", "$")?;
        let options = take_options(&mut fields, "options", "$")?;

        let mut decision = Self::new(id, StateValue::try_from(state)?, question, options)?;
        decision.metadata = fields;
        Ok(decision)
    }

    #[must_use]
    pub const fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(OpenJevError::Validation {
                path: "$.id".to_owned(),
                message: "id must be a nonempty string".to_owned(),
            });
        }
        if self.question.is_empty() {
            return Err(OpenJevError::Validation {
                path: "$.question".to_owned(),
                message: "question must be a nonempty string".to_owned(),
            });
        }
        if !(2..=16).contains(&self.options.len()) {
            return Err(OpenJevError::Validation {
                path: "$.options".to_owned(),
                message: "options must contain 2-16 entries".to_owned(),
            });
        }
        let mut ids = HashSet::with_capacity(self.options.len());
        for (index, option) in self.options.iter().enumerate() {
            if !ids.insert(option.id.as_str()) {
                return Err(OpenJevError::Validation {
                    path: format!("$.options[{index}].id"),
                    message: format!("duplicate option ID {:?}", option.id),
                });
            }
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for Decision {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = deserialize_strict_json_value(deserializer)?;
        Self::from_value(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Question {
    pub id: String,
    pub question: String,
    pub options: Vec<DecisionOption>,
}

impl Question {
    pub fn new(
        id: impl Into<String>,
        question: impl Into<String>,
        options: Vec<DecisionOption>,
    ) -> Result<Self> {
        let id = id.into();
        let question = question.into();
        let placeholder = Decision::new(
            id.clone(),
            StateValue::string("placeholder")?,
            question.clone(),
            options.clone(),
        )?;
        Ok(Self {
            id: placeholder.id,
            question: placeholder.question,
            options,
        })
    }

    fn from_value(value: Value) -> Result<Self> {
        let mut fields = into_object(value, "$")?;
        let id = take_string(&mut fields, "id", "$")?;
        let question = take_string(&mut fields, "question", "$")?;
        let options = take_options(&mut fields, "options", "$")?;
        Self::new(id, question, options)
    }
}

fn into_object(value: Value, path: &str) -> Result<Map<String, Value>> {
    match value {
        Value::Object(fields) => Ok(fields),
        _ => Err(OpenJevError::Validation {
            path: path.to_owned(),
            message: "must be an object".to_owned(),
        }),
    }
}

fn take_required(fields: &mut Map<String, Value>, key: &str, parent: &str) -> Result<Value> {
    fields
        .shift_remove(key)
        .ok_or_else(|| OpenJevError::Validation {
            path: crate::validate::object_path(parent, key),
            message: "missing required field".to_owned(),
        })
}

fn take_string(fields: &mut Map<String, Value>, key: &str, parent: &str) -> Result<String> {
    let path = crate::validate::object_path(parent, key);
    match take_required(fields, key, parent)? {
        Value::String(value) => Ok(value),
        _ => Err(OpenJevError::Validation {
            path,
            message: "must be a string".to_owned(),
        }),
    }
}

fn take_options(
    fields: &mut Map<String, Value>,
    key: &str,
    parent: &str,
) -> Result<Vec<DecisionOption>> {
    let path = crate::validate::object_path(parent, key);
    let values = match take_required(fields, key, parent)? {
        Value::Array(values) => values,
        _ => {
            return Err(OpenJevError::Validation {
                path,
                message: "must be an array".to_owned(),
            });
        }
    };

    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let option_path = format!("{path}[{index}]");
            let mut fields = into_object(value, &option_path)?;
            Ok(DecisionOption {
                id: take_string(&mut fields, "id", &option_path)?,
                description: take_string(&mut fields, "description", &option_path)?,
            })
        })
        .collect()
}

impl<'de> Deserialize<'de> for Question {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = deserialize_strict_json_value(deserializer)?;
        Self::from_value(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Primitive {
    #[default]
    Choice,
    Noul,
    Score,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLayersRequested {
    All,
    Count(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuLayersStatus {
    Reported,
    KnownDisabled,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplateMetadataStatus {
    Exact,
    ReviewedEquivalent,
    OverrideUnverified,
}

impl Serialize for GpuLayersRequested {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::All => serializer.serialize_str("all"),
            Self::Count(count) => serializer.serialize_u32(*count),
        }
    }
}

impl<'de> Deserialize<'de> for GpuLayersRequested {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            String(String),
            Count(u32),
        }
        match Raw::deserialize(deserializer)? {
            Raw::String(value) if value == "all" => Ok(Self::All),
            Raw::String(value) => Err(de::Error::custom(format!(
                "gpu_layers_requested string must be \"all\", got {value:?}"
            ))),
            Raw::Count(count) => Ok(Self::Count(count)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Integrity {
    ManifestSha256,
    CallerSha256,
    LocalUnverified,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeReference {
    pub source: String,
    pub revision: String,
    pub dtype: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub source: String,
    pub revision: String,
    pub file: String,
    pub quant: String,
    pub backend: String,
    pub artifact_sha256: String,
    pub integrity: Integrity,
    pub dtype: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_reference: Option<NativeReference>,
    pub template_profile: crate::prompt::PromptProfile,
    pub template_sha256: Option<String>,
    pub template_override: bool,
    pub template_status: TemplateMetadataStatus,
    pub template_equivalence_evidence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serving_config: Option<String>,
    pub adapter: Option<String>,
    pub adapter_sha256: Option<String>,
    pub adapter_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformers_version: Option<String>,
}

impl ModelMetadata {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, value) in [
            ("id", self.id.as_str()),
            ("source", self.source.as_str()),
            ("revision", self.revision.as_str()),
            ("file", self.file.as_str()),
            ("quant", self.quant.as_str()),
            ("backend", self.backend.as_str()),
            ("dtype", self.dtype.as_str()),
        ] {
            if value.is_empty() {
                return validation(&format!("{path}.{field}"), "value must be nonempty");
            }
        }
        check_hash(&self.artifact_sha256, &format!("{path}.artifact_sha256"))?;
        if let Some(hash) = &self.template_sha256 {
            check_hash(hash, &format!("{path}.template_sha256"))?;
        }
        if let Some(hash) = &self.adapter_sha256 {
            check_hash(hash, &format!("{path}.adapter_sha256"))?;
        }
        match self.template_status {
            TemplateMetadataStatus::Exact => {
                if self.template_sha256.is_none()
                    || self.template_override
                    || self.template_equivalence_evidence.is_some()
                {
                    return validation(
                        &format!("{path}.template_status"),
                        "exact template metadata requires a hash, no override, and no equivalence evidence",
                    );
                }
            }
            TemplateMetadataStatus::ReviewedEquivalent => {
                if self.template_sha256.is_none()
                    || self.template_override
                    || self
                        .template_equivalence_evidence
                        .as_ref()
                        .is_none_or(String::is_empty)
                {
                    return validation(
                        &format!("{path}.template_status"),
                        "reviewed-equivalent metadata requires a hash and nonempty evidence without an override",
                    );
                }
            }
            TemplateMetadataStatus::OverrideUnverified => {
                if !self.template_override || self.template_equivalence_evidence.is_some() {
                    return validation(
                        &format!("{path}.template_status"),
                        "override-unverified metadata requires an explicit override and no equivalence evidence",
                    );
                }
            }
        }
        if (self.adapter_sha256.is_some() || self.adapter_revision.is_some())
            && self.adapter.is_none()
        {
            return validation(path, "adapter metadata requires adapter identity");
        }
        if let Some(reference) = &self.native_reference
            && (reference.source.is_empty()
                || reference.dtype != "bfloat16"
                || !is_lower_hex(&reference.revision, 40))
        {
            return validation(
                &format!("{path}.native_reference"),
                "native reference requires source, 40-hex revision, and bfloat16 dtype",
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub requested_mode: ExecutionMode,
    pub effective_mode: ExecutionMode,
    pub fallback_reason: Option<String>,
    pub device: Device,
    pub device_name: String,
    pub gpu_layers_requested: GpuLayersRequested,
    pub gpu_layers_actual: Option<u32>,
    pub gpu_layers_status: GpuLayersStatus,
    pub threads: u32,
    pub n_ctx_requested: Option<u32>,
    pub n_ctx_actual: u32,
    pub max_tokens: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub kv_unified: bool,
    pub waves: u32,
    pub probe_id: Option<String>,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
}

impl ExecutionMetadata {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, value) in [
            ("threads", self.threads),
            ("n_ctx_actual", self.n_ctx_actual),
            ("max_tokens", self.max_tokens),
            ("n_batch", self.n_batch),
            ("n_ubatch", self.n_ubatch),
            ("n_seq_max", self.n_seq_max),
            ("waves", self.waves),
        ] {
            if value == 0 {
                return validation(&format!("{path}.{field}"), "value must be positive");
            }
        }
        if self.n_ctx_requested == Some(0) {
            return validation(
                &format!("{path}.n_ctx_requested"),
                "value must be null or positive",
            );
        }
        if self.n_ubatch > self.n_batch || self.n_batch > self.n_ctx_actual {
            return validation(path, "require n_ubatch <= n_batch <= n_ctx_actual");
        }
        if self.device_name.is_empty() || self.run_id.is_empty() {
            return validation(path, "device_name and run_id must be nonempty");
        }
        match self.gpu_layers_status {
            GpuLayersStatus::KnownDisabled => {
                if self.device != Device::Cpu
                    || self.gpu_layers_requested != GpuLayersRequested::Count(0)
                    || self.gpu_layers_actual != Some(0)
                {
                    return validation(
                        &format!("{path}.gpu_layers_status"),
                        "known-disabled requires CPU, requested zero, and actual zero",
                    );
                }
            }
            GpuLayersStatus::Unavailable => {
                if self.gpu_layers_actual.is_some() {
                    return validation(
                        &format!("{path}.gpu_layers_actual"),
                        "unavailable actual layer count must be null",
                    );
                }
            }
            GpuLayersStatus::Reported => {
                if self.gpu_layers_actual.is_none() {
                    return validation(
                        &format!("{path}.gpu_layers_actual"),
                        "reported actual layer count must be an integer",
                    );
                }
            }
        }
        if self.group_id.as_ref().is_some_and(String::is_empty) {
            return validation(&format!("{path}.group_id"), "group_id must be nonempty");
        }
        if self.requested_mode == self.effective_mode {
            if self.fallback_reason.is_some() {
                return validation(
                    &format!("{path}.fallback_reason"),
                    "unchanged execution mode cannot have a fallback reason",
                );
            }
        } else if self
            .fallback_reason
            .as_ref()
            .is_none_or(|reason| reason.is_empty())
        {
            return validation(
                &format!("{path}.fallback_reason"),
                "mode fallback requires a nonempty reason",
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    Direct,
    Serial,
    Shared,
    Batch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedTiming {
    #[serde(serialize_with = "serialize_f64")]
    pub total_seconds: f64,
    #[serde(serialize_with = "serialize_f64")]
    pub encode_seconds: f64,
    pub prefix_tokens: u64,
    #[serde(serialize_with = "serialize_f64")]
    pub prefill_seconds: f64,
    #[serde(serialize_with = "serialize_f64")]
    pub replicate_seconds: f64,
    #[serde(serialize_with = "serialize_f64")]
    pub suffix_forward_seconds: f64,
    pub batch_size: u64,
    pub true_suffix_tokens: u64,
    pub padded_suffix_tokens: u64,
}

impl SharedTiming {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, value) in [
            ("total_seconds", self.total_seconds),
            ("encode_seconds", self.encode_seconds),
            ("prefill_seconds", self.prefill_seconds),
            ("replicate_seconds", self.replicate_seconds),
            ("suffix_forward_seconds", self.suffix_forward_seconds),
        ] {
            check_nonnegative(value, &format!("{path}.{field}"))?;
        }
        if self.batch_size == 0 {
            return validation(&format!("{path}.batch_size"), "batch_size must be positive");
        }
        if self.padded_suffix_tokens != self.true_suffix_tokens {
            return validation(
                &format!("{path}.padded_suffix_tokens"),
                "Rust shared batching must report no padding",
            );
        }
        Ok(())
    }
}

/// Provenance for one raw option-order scoring sample.
///
/// M1 defines this schema container only; permutation generation is an M7 concern.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RawSample {
    pub permutation: Vec<usize>,
    pub option_ids: Vec<String>,
    pub answer_token_ids: Vec<u32>,
    #[serde(serialize_with = "serialize_f64_vec")]
    pub option_logits: Vec<f64>,
    #[serde(serialize_with = "serialize_f64_vec")]
    pub probabilities: Vec<f64>,
    pub prompt_sha256: String,
    pub prompt_version: String,
    pub input_tokens: u64,
    #[serde(serialize_with = "serialize_f64")]
    pub allowed_token_mass: f64,
    pub full_vocab_argmax_id: u32,
    #[serde(serialize_with = "serialize_f64")]
    pub full_vocab_log_normalizer: f64,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub forward_seconds: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub total_seconds: Option<f64>,
    pub execution: ExecutionMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_timing: Option<SharedTiming>,
}

impl RawSample {
    fn validate(&self, path: &str) -> Result<()> {
        let count = self.option_ids.len();
        if !(2..=16).contains(&count)
            || self.answer_token_ids.len() != count
            || self.option_logits.len() != count
            || self.probabilities.len() != count
            || self.permutation.len() != count
        {
            return validation(path, "raw sample vectors must have equal 2-16 lengths");
        }
        if self.option_ids.iter().collect::<HashSet<_>>().len() != count
            || self.answer_token_ids.iter().collect::<HashSet<_>>().len() != count
        {
            return validation(path, "raw sample option and token IDs must be unique");
        }
        let permutation: HashSet<_> = self.permutation.iter().copied().collect();
        if permutation.len() != count || !(0..count).all(|index| permutation.contains(&index)) {
            return validation(
                &format!("{path}.permutation"),
                "permutation must contain every displayed index exactly once",
            );
        }
        check_finite_slice(&self.option_logits, &format!("{path}.option_logits"))?;
        check_probabilities(&self.probabilities, &format!("{path}.probabilities"))?;
        if self.input_tokens == 0 {
            return validation(&format!("{path}.input_tokens"), "value must be positive");
        }
        check_hash(&self.prompt_sha256, &format!("{path}.prompt_sha256"))?;
        if self.prompt_version != crate::prompt::PROMPT_VERSION {
            return validation(
                &format!("{path}.prompt_version"),
                "unexpected prompt version",
            );
        }
        check_probability(
            self.allowed_token_mass,
            &format!("{path}.allowed_token_mass"),
        )?;
        check_finite(
            self.full_vocab_log_normalizer,
            &format!("{path}.full_vocab_log_normalizer"),
        )?;
        for (field, value) in [
            ("forward_seconds", self.forward_seconds),
            ("total_seconds", self.total_seconds),
        ] {
            if let Some(value) = value {
                check_nonnegative(value, &format!("{path}.{field}"))?;
            }
        }
        self.execution.validate(&format!("{path}.execution"))?;
        if let Some(timing) = &self.shared_timing {
            timing.validate(&format!("{path}.shared_timing"))?;
        }
        Ok(())
    }
}

/// Provenance container for postprocessing.
///
/// M1 exports the type and schema but intentionally implements no M7 transform.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Postprocess {
    pub version: String,
    #[serde(serialize_with = "serialize_f64")]
    pub temperature: f64,
    pub calibration_id: Option<String>,
    pub permutation_count: u32,
    pub seed: u64,
    pub permutation_algorithm: String,
    pub aggregation: String,
    pub raw_fields_reference: usize,
    #[serde(serialize_with = "serialize_f64_vec")]
    pub base_probabilities: Vec<f64>,
    pub samples: Vec<RawSample>,
}

impl Postprocess {
    fn validate(&self, path: &str, option_ids: &[String]) -> Result<()> {
        let option_count = option_ids.len();
        if self.version != "openjev-postprocess-v1"
            || self.permutation_algorithm != "sha256-factoradic-v1"
            || self.aggregation != "mean-id-aligned-probabilities"
            || self.raw_fields_reference != 0
        {
            return validation(path, "postprocess provenance constants are invalid");
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return validation(
                &format!("{path}.temperature"),
                "temperature must be positive and finite",
            );
        }
        if !(1..=64).contains(&self.permutation_count)
            || usize::try_from(self.permutation_count).ok() != Some(self.samples.len())
            || self.samples.is_empty()
        {
            return validation(path, "permutation_count must equal 1-64 samples");
        }
        if self.base_probabilities.len() != option_count {
            return validation(
                &format!("{path}.base_probabilities"),
                "base probabilities must align with the readout",
            );
        }
        check_probabilities(
            &self.base_probabilities,
            &format!("{path}.base_probabilities"),
        )?;
        for (index, sample) in self.samples.iter().enumerate() {
            sample.validate(&format!("{path}.samples[{index}]"))?;
            if sample.option_ids.iter().collect::<HashSet<_>>()
                != option_ids.iter().collect::<HashSet<_>>()
            {
                return validation(
                    &format!("{path}.samples[{index}].option_ids"),
                    "sample option IDs must align semantically with the readout",
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Readout {
    pub schema: String,
    pub id: String,
    pub primitive: Primitive,
    pub choice: String,
    pub choice_index: usize,
    pub option_ids: Vec<String>,
    #[serde(serialize_with = "serialize_f64_vec")]
    pub probabilities: Vec<f64>,
    #[serde(serialize_with = "serialize_f64_vec")]
    pub option_logits: Vec<f64>,
    pub answer_token_ids: Vec<u32>,
    #[serde(serialize_with = "serialize_f64")]
    pub allowed_token_mass: f64,
    pub full_vocab_argmax_id: u32,
    #[serde(serialize_with = "serialize_f64")]
    pub full_vocab_log_normalizer: f64,
    pub input_tokens: u64,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub forward_seconds: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub total_seconds: Option<f64>,
    pub prompt_sha256: String,
    pub prompt_version: String,
    pub model: ModelMetadata,
    pub readout: String,
    pub probability_status: String,
    pub limitations: Vec<String>,
    pub execution: ExecutionMetadata,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_status: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub p_yes: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64_vec"
    )]
    pub level_values: Option<Vec<f64>>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argmax_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_sha256: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub prefill_seconds: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub copy_seconds: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_opt_f64"
    )]
    pub suffix_forward_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_timing: Option<SharedTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess: Option<Postprocess>,
}

impl Readout {
    pub fn validate(&self) -> Result<()> {
        if self.schema != "openjev-readout-v1" {
            return validation("$.schema", "schema must be openjev-readout-v1");
        }
        if self.id.is_empty() {
            return validation("$.id", "id must be nonempty");
        }
        if self.input_tokens == 0 {
            return validation("$.input_tokens", "input_tokens must be positive");
        }
        if self.prompt_version != crate::prompt::PROMPT_VERSION {
            return validation("$.prompt_version", "unexpected prompt version");
        }
        check_hash(&self.prompt_sha256, "$.prompt_sha256")?;
        if self.probability_status != PROBABILITY_STATUS {
            return validation(
                "$.probability_status",
                "probability_status must use the exact honesty string",
            );
        }
        if !self
            .limitations
            .iter()
            .any(|value| value == FORCED_TYPED_LIMITATION)
            || !self
                .limitations
                .iter()
                .any(|value| value == CONDITIONAL_PROBABILITY_LIMITATION)
        {
            return validation("$.limitations", "required limitations are missing");
        }
        self.model.validate("$.model")?;
        self.execution.validate("$.execution")?;
        let expected_readout = if self.execution.effective_mode == ExecutionMode::Shared {
            "native selected suffix-position logits"
        } else {
            DIRECT_READOUT
        };
        if self.readout != expected_readout {
            return validation("$.readout", "readout does not match the execution path");
        }
        match self.execution.effective_mode {
            ExecutionMode::Shared => {
                if self
                    .execution
                    .probe_id
                    .as_ref()
                    .is_none_or(String::is_empty)
                    || self.shared_timing.is_none()
                    || self.prefix_tokens.is_none()
                    || self.prefix_sha256.is_none()
                    || self.cache_hit != Some(true)
                    || self.forward_seconds.is_some()
                    || self.total_seconds.is_some()
                {
                    return validation(
                        "$",
                        "shared execution requires a probe ID, real prefix reuse metadata, common timing, and no fabricated per-row timing",
                    );
                }
            }
            ExecutionMode::Batch => {
                if self
                    .execution
                    .probe_id
                    .as_ref()
                    .is_none_or(String::is_empty)
                    || self.shared_timing.is_some()
                    || self.prefix_tokens.is_some()
                    || self.prefix_sha256.is_some()
                    || self.cache_hit != Some(false)
                {
                    return validation(
                        "$",
                        "batch execution requires a probe ID and cannot claim shared-prefix metadata",
                    );
                }
            }
            ExecutionMode::Direct | ExecutionMode::Serial => {
                if self.execution.probe_id.is_some() || self.shared_timing.is_some() {
                    return validation(
                        "$",
                        "direct/serial execution cannot claim probe-authorized shared timing",
                    );
                }
            }
        }
        let count = self.option_ids.len();
        if !(2..=16).contains(&count) {
            return validation("$.option_ids", "option count must be 2-16");
        }
        if self.probabilities.len() != count
            || self.option_logits.len() != count
            || self.answer_token_ids.len() != count
        {
            return validation("$", "readout vectors must have equal lengths");
        }
        if self.choice_index >= count || self.choice != self.option_ids[self.choice_index] {
            return validation("$.choice", "choice and choice_index do not align");
        }
        let option_ids: HashSet<_> = self.option_ids.iter().collect();
        let token_ids: HashSet<_> = self.answer_token_ids.iter().collect();
        if option_ids.len() != count || token_ids.len() != count {
            return validation("$", "option IDs and answer token IDs must be unique");
        }
        check_finite_slice(&self.option_logits, "$.option_logits")?;
        check_probabilities(&self.probabilities, "$.probabilities")?;
        if crate::numerics::first_argmax(&self.probabilities)? != self.choice_index {
            return validation(
                "$.choice_index",
                "choice_index must be the first maximum probability",
            );
        }
        check_finite(self.allowed_token_mass, "$.allowed_token_mass")?;
        check_finite(
            self.full_vocab_log_normalizer,
            "$.full_vocab_log_normalizer",
        )?;
        if !(0.0..=1.0).contains(&self.allowed_token_mass) {
            return validation("$.allowed_token_mass", "mass must be in [0,1]");
        }
        if let Some(hash) = &self.prefix_sha256 {
            check_hash(hash, "$.prefix_sha256")?;
        }
        for (path, value) in [
            ("$.forward_seconds", self.forward_seconds),
            ("$.total_seconds", self.total_seconds),
            ("$.confidence", self.confidence),
            ("$.p_yes", self.p_yes),
            ("$.expected_value", self.expected_value),
            ("$.prefill_seconds", self.prefill_seconds),
            ("$.copy_seconds", self.copy_seconds),
            ("$.suffix_forward_seconds", self.suffix_forward_seconds),
        ] {
            if let Some(value) = value {
                check_finite(value, path)?;
            }
        }
        for (path, value) in [
            ("$.forward_seconds", self.forward_seconds),
            ("$.total_seconds", self.total_seconds),
            ("$.prefill_seconds", self.prefill_seconds),
            ("$.copy_seconds", self.copy_seconds),
            ("$.suffix_forward_seconds", self.suffix_forward_seconds),
        ] {
            if value.is_some_and(|value| value < 0.0) {
                return validation(path, "timing must be nonnegative");
            }
        }
        if self.confidence.is_some() != self.confidence_status.is_some() {
            return validation(
                "$.confidence",
                "confidence and confidence_status must appear together",
            );
        }
        if let Some(confidence) = self.confidence {
            check_probability(confidence, "$.confidence")?;
            if self.confidence_status.as_deref() != Some(CONFIDENCE_STATUS) {
                return validation(
                    "$.confidence_status",
                    "confidence_status must use the exact honesty string",
                );
            }
            let expected = crate::numerics::normalized_margin(&self.probabilities)?;
            if (confidence - expected).abs() > 1e-12 {
                return validation(
                    "$.confidence",
                    "confidence does not match normalized margin",
                );
            }
        }
        match self.primitive {
            Primitive::Choice => {
                if self.p_yes.is_some()
                    || self.level_values.is_some()
                    || self.expected_value.is_some()
                    || self.argmax_level.is_some()
                {
                    return validation("$", "choice readout has primitive-only fields");
                }
            }
            Primitive::Noul => {
                if self.option_ids != ["yes", "no"]
                    || self.p_yes != Some(self.probabilities[0])
                    || self.level_values.is_some()
                    || self.expected_value.is_some()
                    || self.argmax_level.is_some()
                {
                    return validation("$", "noul readout fields are inconsistent");
                }
            }
            Primitive::Score => {
                let Some(values) = &self.level_values else {
                    return validation("$.level_values", "score requires level_values");
                };
                let expected: f64 = self
                    .probabilities
                    .iter()
                    .zip(values)
                    .map(|(probability, value)| probability * value)
                    .sum();
                if self.p_yes.is_some()
                    || values.len() != count
                    || self.expected_value != Some(expected)
                    || self.argmax_level.as_deref()
                        != Some(self.option_ids[self.choice_index].as_str())
                {
                    return validation("$", "score readout fields are inconsistent");
                }
            }
        }
        if let Some(values) = &self.level_values {
            check_finite_slice(values, "$.level_values")?;
        }
        if let Some(timing) = &self.shared_timing {
            timing.validate("$.shared_timing")?;
        }
        if let Some(postprocess) = &self.postprocess {
            postprocess.validate("$.postprocess", &self.option_ids)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub schema: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub error: ErrorDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_status: Option<String>,
}

impl ErrorRecord {
    #[must_use]
    pub fn from_error(error: &OpenJevError) -> Self {
        Self {
            schema: "openjev-error-v1".to_owned(),
            id: None,
            error: ErrorDetail {
                code: error.code().to_owned(),
                message: error.to_string(),
                details: Map::new(),
            },
            parse_status: None,
        }
    }

    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            schema: "openjev-error-v1".to_owned(),
            id: None,
            error: ErrorDetail {
                code: code.into(),
                message: message.into(),
                details: Map::new(),
            },
            parse_status: None,
        }
    }
}

fn validation<T>(path: &str, message: &str) -> Result<T> {
    Err(OpenJevError::Validation {
        path: path.to_owned(),
        message: message.to_owned(),
    })
}

fn check_finite(value: f64, path: &str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        validation(path, "value must be finite")
    }
}

fn check_finite_slice(values: &[f64], path: &str) -> Result<()> {
    for (index, value) in values.iter().copied().enumerate() {
        check_finite(value, &format!("{path}[{index}]"))?;
    }
    Ok(())
}

fn check_nonnegative(value: f64, path: &str) -> Result<()> {
    check_finite(value, path)?;
    if value < 0.0 {
        validation(path, "value must be nonnegative")
    } else {
        Ok(())
    }
}

fn check_probability(value: f64, path: &str) -> Result<()> {
    check_finite(value, path)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        validation(path, "probability must be in [0,1]")
    }
}

fn check_probabilities(values: &[f64], path: &str) -> Result<()> {
    if values.len() < 2 {
        return validation(path, "probability vector must contain at least two values");
    }
    for (index, value) in values.iter().copied().enumerate() {
        check_probability(value, &format!("{path}[{index}]"))?;
    }
    if (values.iter().sum::<f64>() - 1.0).abs() > 1e-12 {
        return validation(path, "probabilities must sum to one within 1e-12");
    }
    Ok(())
}

fn check_hash(value: &str, path: &str) -> Result<()> {
    if is_lower_hex(value, 64) {
        Ok(())
    } else {
        validation(
            path,
            "value must be exactly 64 lowercase hexadecimal characters",
        )
    }
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn serialize_f64<S>(value: &f64, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else {
        Err(S::Error::custom("nonfinite output number"))
    }
}

pub(crate) fn serialize_opt_f64<S>(
    value: &Option<f64>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(value) if value.is_finite() => serializer.serialize_some(value),
        Some(_) => Err(S::Error::custom("nonfinite output number")),
        None => serializer.serialize_none(),
    }
}

fn serialize_f64_vec<S>(values: &[f64], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if values.iter().all(|value| value.is_finite()) {
        values.serialize(serializer)
    } else {
        Err(S::Error::custom("nonfinite output number"))
    }
}

fn serialize_opt_f64_vec<S>(
    values: &Option<Vec<f64>>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match values {
        Some(values) if values.iter().all(|value| value.is_finite()) => {
            serializer.serialize_some(values)
        }
        Some(_) => Err(S::Error::custom("nonfinite output number")),
        None => serializer.serialize_none(),
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt::Debug, io::Cursor};

    use serde::de::DeserializeOwned;
    use serde_json::Number;

    use super::*;

    const OPTIONS_JSON: &str = r#"[{"id":"a","description":"A"},{"id":"b","description":"B"}]"#;

    fn serde_json_routes<T>(input: &str) -> Vec<std::result::Result<T, serde_json::Error>>
    where
        T: DeserializeOwned,
    {
        vec![
            serde_json::from_str(input),
            serde_json::from_slice(input.as_bytes()),
            serde_json::from_reader(Cursor::new(input.as_bytes())),
        ]
    }

    fn assert_serde_json_routes_accept<T>(input: &str, expected: &T)
    where
        T: DeserializeOwned + Debug + PartialEq,
    {
        for parsed in serde_json_routes::<T>(input) {
            assert_eq!(&parsed.unwrap(), expected, "input: {input}");
        }
    }

    fn assert_serde_json_routes_reject<T>(input: &str)
    where
        T: DeserializeOwned + Debug,
    {
        for parsed in serde_json_routes::<T>(input) {
            assert!(parsed.is_err(), "route accepted input: {input}");
        }
    }

    fn decision_json(state: &str) -> String {
        format!(r#"{{"id":"d","state":{state},"question":"q","options":{OPTIONS_JSON}}}"#)
    }

    fn question_json(extra_value: &str) -> String {
        format!(r#"{{"id":"q","question":"q","options":{OPTIONS_JSON},"ignored":{extra_value}}}"#)
    }

    fn nested_arrays(depth: usize, leaf: &str) -> String {
        format!("{}{leaf}{}", "[".repeat(depth), "]".repeat(depth))
    }

    #[test]
    fn state_top_level_shape_and_whitespace_match_upstream() {
        assert!(StateValue::string(" ").is_ok());
        assert!(StateValue::string("").is_err());
        assert!(StateValue::try_from(Value::Array(vec![])).is_err());
        assert!(StateValue::try_from(Value::Object(Map::new())).is_err());
        assert!(StateValue::try_from(Value::Null).is_err());
        assert!(StateValue::parse_json(r#"[{"nested": []}, null, true]"#).is_ok());
    }

    #[test]
    fn decision_allows_empty_options_but_not_duplicate_ids() {
        let options = vec![
            DecisionOption {
                id: String::new(),
                description: String::new(),
            },
            DecisionOption {
                id: "second".to_owned(),
                description: String::new(),
            },
        ];
        assert!(Decision::new(" ", StateValue::string(" ").unwrap(), " ", options).is_ok());

        let duplicates = vec![
            DecisionOption {
                id: String::new(),
                description: "a".to_owned(),
            },
            DecisionOption {
                id: String::new(),
                description: "b".to_owned(),
            },
        ];
        assert!(Decision::new("id", StateValue::string("s").unwrap(), "q", duplicates).is_err());
    }

    #[test]
    fn raw_state_routes_preserve_reserved_keys_and_json_looking_strings() {
        for input in [
            r#"{"$serde_json::private::RawValue":"[1,2]"}"#,
            r#"{"outer":[{"$serde_json::private::RawValue":"[1,2]"},{"$serde_json::private::Number":"123"}]}"#,
            r#""[1,2]""#,
            r#""hello""#,
        ] {
            let direct = StateValue::parse_json(input).unwrap();
            assert_eq!(direct.as_value(), &parse_json_strict(input).unwrap());
            assert_serde_json_routes_accept(input, &direct);
        }
    }

    #[test]
    fn raw_decision_routes_move_state_and_metadata_without_reinterpreting_values() {
        let input = concat!(
            r#"{"before":{"$serde_json::private::RawValue":"[1,2]"},"id":"d","$serde_json::private::RawValue":{"payload":"[3,4]"},"middle":[{"$serde_json::private::Number":"123"}],"state":{"$serde_json::private::RawValue":"[1,2]","nested":[{"$serde_json::private::Number":"456"}]},"question":"q","options":["#,
            r#"{"id":"a","description":"A","$serde_json::private::RawValue":"hello"},"#,
            r#"{"id":"b","description":"B","extra":{"$serde_json::private::Number":"789"}}],"$serde_json::private::Number":["789"],"after":"hello"}"#,
        );
        let direct = Decision::from_json_str(input).unwrap();
        assert_eq!(
            direct.state.as_value().to_string(),
            r#"{"$serde_json::private::RawValue":"[1,2]","nested":[{"$serde_json::private::Number":"456"}]}"#
        );
        assert_eq!(
            Value::Object(direct.metadata().clone()).to_string(),
            r#"{"before":{"$serde_json::private::RawValue":"[1,2]"},"$serde_json::private::RawValue":{"payload":"[3,4]"},"middle":[{"$serde_json::private::Number":"123"}],"$serde_json::private::Number":["789"],"after":"hello"}"#
        );
        assert_eq!(direct.options[0].id, "a");
        assert_eq!(direct.options[1].description, "B");
        assert_serde_json_routes_accept(input, &direct);
    }

    #[test]
    fn raw_decision_routes_preserve_string_states() {
        for state in [r#""[1,2]""#, r#""hello""#] {
            let input = decision_json(state);
            let direct = Decision::from_json_str(&input).unwrap();
            assert_eq!(direct.state.as_value(), &parse_json_strict(state).unwrap());
            assert_serde_json_routes_accept(&input, &direct);
        }
    }

    #[test]
    fn raw_question_routes_ignore_extra_fields_without_reinterpreting_them() {
        let input = concat!(
            r#"{"id":"q","question":"question","options":["#,
            r#"{"id":"a","description":"A","$serde_json::private::RawValue":"hello"},"#,
            r#"{"id":"b","description":"B","extra":{"$serde_json::private::Number":"123"}}],"#,
            r#""metadata":{"$serde_json::private::RawValue":"[1,2]"}}"#,
        );
        let expected = Question::new(
            "q",
            "question",
            vec![
                DecisionOption {
                    id: "a".to_owned(),
                    description: "A".to_owned(),
                },
                DecisionOption {
                    id: "b".to_owned(),
                    description: "B".to_owned(),
                },
            ],
        )
        .unwrap();
        assert_serde_json_routes_accept(input, &expected);
    }

    #[test]
    fn typed_extraction_reports_field_paths() {
        let wrong_id = decision_json("\"state\"").replacen(r#""id":"d""#, r#""id":1"#, 1);
        assert!(
            Decision::from_json_str(&wrong_id)
                .unwrap_err()
                .to_string()
                .contains("$.id")
        );

        let missing_description = r#"{"id":"d","state":"state","question":"q","options":[{"id":"a"},{"id":"b","description":"B"}]}"#;
        assert!(
            Decision::from_json_str(missing_description)
                .unwrap_err()
                .to_string()
                .contains("$.options[0].description")
        );
    }

    #[test]
    fn state_numeric_lexemes_match_all_raw_json_routes() {
        for (input, normalized) in [
            ("[-0]", "[0]"),
            (r#"[{"nested":-0}]"#, r#"[{"nested":0}]"#),
            (
                "[-9223372036854775808,18446744073709551615]",
                "[-9223372036854775808,18446744073709551615]",
            ),
        ] {
            let direct = StateValue::parse_json(input).unwrap();
            assert_eq!(direct.as_value().to_string(), normalized);
            assert_serde_json_routes_accept(input, &direct);
        }

        for input in [
            "[-0.0]",
            "[-0e0]",
            "[1.0]",
            "[1e0]",
            "[-9223372036854775809]",
            "[18446744073709551616]",
        ] {
            assert!(StateValue::parse_json(input).is_err(), "input: {input}");
            assert_serde_json_routes_reject::<StateValue>(input);
        }
    }

    #[test]
    fn decision_numeric_lexemes_match_all_raw_json_routes() {
        for (state, normalized) in [
            ("[-0]", "[0]"),
            (r#"[{"nested":-0}]"#, r#"[{"nested":0}]"#),
            (
                "[-9223372036854775808,18446744073709551615]",
                "[-9223372036854775808,18446744073709551615]",
            ),
        ] {
            let input = decision_json(state);
            let direct = Decision::from_json_str(&input).unwrap();
            assert_eq!(direct.state.as_value().to_string(), normalized);
            assert_serde_json_routes_accept(&input, &direct);
        }

        for state in [
            "[-0.0]",
            "[-0e0]",
            "[1.0]",
            "[1e0]",
            "[-9223372036854775809]",
            "[18446744073709551616]",
        ] {
            let input = decision_json(state);
            assert!(Decision::from_json_str(&input).is_err(), "state: {state}");
            assert_serde_json_routes_reject::<Decision>(&input);
        }
    }

    #[test]
    fn raw_json_routes_reject_duplicates_for_all_public_input_types() {
        let state = r#"{"outer":{"x":1,"x":2}}"#;
        let state_error = StateValue::parse_json(state).unwrap_err().to_string();
        assert!(state_error.contains("$.outer.x"));
        assert!(state_error.contains("duplicate"));
        assert_serde_json_routes_reject::<StateValue>(state);

        let decision = decision_json(state);
        let decision_error = Decision::from_json_str(&decision).unwrap_err().to_string();
        assert!(decision_error.contains("$.state.outer.x"));
        assert!(decision_error.contains("duplicate"));
        assert_serde_json_routes_reject::<Decision>(&decision);

        let question = format!(
            r#"{{"id":"q","question":"first","question":"second","options":{OPTIONS_JSON}}}"#
        );
        assert_serde_json_routes_reject::<Question>(&question);
    }

    #[test]
    fn raw_json_routes_share_explicit_depth_boundaries() {
        for depth in [127, 128] {
            let input = nested_arrays(depth, "0");
            let direct = StateValue::parse_json(&input).unwrap();
            assert_serde_json_routes_accept(&input, &direct);
        }
        let too_deep = nested_arrays(129, "0");
        assert!(StateValue::parse_json(&too_deep).is_err());
        assert_serde_json_routes_reject::<StateValue>(&too_deep);

        // The Decision/Question envelope is itself one container, so 127 state
        // or ignored-value arrays reach the same whole-document limit of 128.
        let decision_at_limit = decision_json(&nested_arrays(127, "0"));
        let direct = Decision::from_json_str(&decision_at_limit).unwrap();
        assert_serde_json_routes_accept(&decision_at_limit, &direct);
        let decision_too_deep = decision_json(&nested_arrays(128, "0"));
        assert!(Decision::from_json_str(&decision_too_deep).is_err());
        assert_serde_json_routes_reject::<Decision>(&decision_too_deep);

        let question_at_limit = question_json(&nested_arrays(127, "0"));
        let expected: Question = serde_json::from_str(&question_at_limit).unwrap();
        assert_serde_json_routes_accept(&question_at_limit, &expected);
        assert_serde_json_routes_reject::<Question>(&question_json(&nested_arrays(128, "0")));
    }

    #[test]
    fn raw_json_routes_bound_five_thousand_nested_containers() {
        let nested = nested_arrays(5_000, "0");
        let state_error = StateValue::parse_json(&nested).unwrap_err().to_string();
        assert!(state_error.contains("maximum depth 128"));
        assert_serde_json_routes_reject::<StateValue>(&nested);

        let decision = decision_json(&nested);
        let decision_error = Decision::from_json_str(&decision).unwrap_err().to_string();
        assert!(decision_error.contains("maximum depth 128"));
        assert_serde_json_routes_reject::<Decision>(&decision);

        assert_serde_json_routes_reject::<Question>(&question_json(&nested));
    }

    #[test]
    fn owned_value_conversion_remains_validated_but_cannot_recover_lexemes() {
        let mut object = Map::new();
        object.insert("z".to_owned(), Value::Number(Number::from(1)));
        object.insert("a".to_owned(), Value::Number(Number::from(u64::MAX)));
        let value = Value::Object(object);
        let state_from_try = StateValue::try_from(value.clone()).unwrap();
        let state_from_deserialize = serde_json::from_value::<StateValue>(value).unwrap();
        assert_eq!(state_from_deserialize, state_from_try);
        assert_eq!(
            state_from_deserialize.as_value().to_string(),
            r#"{"z":1,"a":18446744073709551615}"#
        );

        let lost_lexeme = serde_json::from_str::<Value>("[-0]").unwrap();
        assert_eq!(lost_lexeme.to_string(), "[-0.0]");
        assert!(StateValue::try_from(lost_lexeme.clone()).is_err());
        assert!(serde_json::from_value::<StateValue>(lost_lexeme).is_err());
    }
}
