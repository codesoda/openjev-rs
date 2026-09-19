use std::{collections::HashSet, path::Path};

#[cfg(feature = "native")]
use openjev_core::Question;
use openjev_core::{
    CONFIDENCE_STATUS, Decision, DecisionOption, Device, ExecutionMode, GpuLayersRequested, Noul,
    Readout, Score, ScoreLevel, StateValue, normalized_margin, python_json_dumps,
};
#[cfg(feature = "native")]
use openjev_llama::{
    CacheOptions, MAX_ABS_SLOT_LOGIT, MAX_PROBABILITY_DELTA, NATIVE_PIN, PROBE_SUITE_VERSION,
    ProbeCaseResult, ProbeCaseStatus, ProbeConfiguration, ProbeEligibility, ProbeMode,
    ProbePublication, ProbeReceipt, begin_probe_publication, load_passing_receipt, probe_id,
};
use openjev_llama::{ModelCache, ModelRegistry, ModelSpec, validate_hub_identity, validate_sha256};
use serde::Serialize;

use crate::{
    CliError,
    args::{DecideArgs, DeviceArg, GlobalArgs, ModeArg, NoulArgs, ScoreArgs},
};

pub const SHARED_FALLBACK_REASON: &str = "no passing exact-configuration shared probe receipt";
pub const BATCH_FALLBACK_REASON: &str =
    "no passing exact-configuration independent-batch probe receipt";

#[derive(Clone, Debug)]
pub struct ScoringConfig {
    pub model: ModelSpec,
    pub cache_dir: Option<std::path::PathBuf>,
    pub offline: bool,
    pub device: Device,
    pub gpu_layers: GpuLayersRequested,
    pub threads: u32,
    pub n_ctx: Option<u32>,
    pub max_tokens: u32,
    pub max_context_tokens: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub max_sequences: u32,
    pub confidence: bool,
}

#[derive(Clone, Debug)]
pub enum Adapter {
    Choice(Decision),
    Noul(Noul),
    Score(Score),
}

impl Adapter {
    pub fn decision(&self) -> &Decision {
        match self {
            Self::Choice(decision) => decision,
            Self::Noul(noul) => noul.decision(),
            Self::Score(score) => score.decision(),
        }
    }

    pub fn adapt(&self, readout: Readout) -> Result<Readout, CliError> {
        match self {
            Self::Choice(_) => Ok(readout),
            Self::Noul(noul) => noul.adapt(readout).map_err(CliError::from_runtime_core),
            Self::Score(score) => score.adapt(readout).map_err(CliError::from_runtime_core),
        }
    }
}

pub trait DecisionScorer {
    fn score_direct(&mut self, decision: Decision) -> Result<Readout, CliError>;

    fn probe_id(&self, _mode: ExecutionMode) -> Result<String, String> {
        Err("native shared/batch eligibility is unavailable".to_owned())
    }

    fn score_shared(
        &mut self,
        _decisions: Vec<Decision>,
        _probe_id: String,
    ) -> Result<Vec<Readout>, CliError> {
        Err(CliError::runtime(
            "shared_unavailable",
            "scorer does not implement shared execution",
        ))
    }

    fn score_batch(
        &mut self,
        _decisions: Vec<Decision>,
        _probe_id: String,
    ) -> Result<Vec<Readout>, CliError> {
        Err(CliError::runtime(
            "batch_unavailable",
            "scorer does not implement batch execution",
        ))
    }

    fn shutdown(&mut self) -> Result<(), CliError> {
        Ok(())
    }
}

pub fn scoring_config(global: &GlobalArgs) -> Result<ScoringConfig, CliError> {
    reject_unimplemented_postprocessing(global)?;
    if global.max_sequences == Some(0) {
        return Err(CliError::validation("--max-sequences must be positive"));
    }
    if global.max_sequences.is_some_and(|value| value > 64) {
        return Err(CliError::validation(
            "--max-sequences must not exceed the validated v1 limit of 64",
        ));
    }
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_validation)?;
    let model = parse_model_spec(global, &registry)?;
    let device = match global.device {
        Some(DeviceArg::Cpu) => Device::Cpu,
        Some(DeviceArg::Metal) => Device::Metal,
        Some(DeviceArg::Cuda) => Device::Cuda,
        None if cfg!(all(target_os = "macos", feature = "metal")) => Device::Metal,
        None => Device::Cpu,
    };
    if device == Device::Metal && !cfg!(feature = "metal") {
        return Err(CliError::validation(
            "Metal execution requires building openjev-cli with --features metal",
        ));
    }
    if device == Device::Cuda && !cfg!(feature = "cuda") {
        return Err(CliError::validation(
            "CUDA execution requires building openjev-cli with --features cuda",
        ));
    }
    let gpu_layers = parse_gpu_layers(global.gpu_layers.as_deref(), device)?;
    let threads = global.threads.unwrap_or_else(default_threads);
    let max_tokens = global.max_tokens.unwrap_or(4096);
    let max_context_tokens = global.max_context_tokens.unwrap_or(32_768);
    let n_batch = global.n_batch.unwrap_or(512);
    let n_ubatch = global.n_ubatch.unwrap_or(512);
    let max_sequences = global.max_sequences.unwrap_or(32);
    for (name, value) in [
        ("threads", threads),
        ("max-tokens", max_tokens),
        ("max-context-tokens", max_context_tokens),
        ("n-batch", n_batch),
        ("n-ubatch", n_ubatch),
    ] {
        if value == 0 {
            return Err(CliError::validation(format!("--{name} must be positive")));
        }
    }
    if global.n_ctx == Some(0) {
        return Err(CliError::validation("--n-ctx must be positive"));
    }
    if global.n_ctx.is_some_and(|n_ctx| n_ctx > max_context_tokens) {
        return Err(CliError::validation(
            "--n-ctx must not exceed --max-context-tokens",
        ));
    }
    if global.n_ctx.is_none() && max_context_tokens < 4096 {
        return Err(CliError::validation(
            "automatic context allocation requires --max-context-tokens >= 4096",
        ));
    }
    if n_ubatch > n_batch {
        return Err(CliError::validation("--n-ubatch must not exceed --n-batch"));
    }
    if threads > i32::MAX as u32 {
        return Err(CliError::validation("--threads exceeds i32::MAX"));
    }
    Ok(ScoringConfig {
        model,
        cache_dir: global.cache_dir.clone(),
        offline: global.offline,
        device,
        gpu_layers,
        threads,
        n_ctx: global.n_ctx,
        max_tokens,
        max_context_tokens,
        n_batch,
        n_ubatch,
        max_sequences,
        confidence: global.confidence,
    })
}

pub(crate) fn reject_unimplemented_postprocessing(global: &GlobalArgs) -> Result<(), CliError> {
    if global.permute.is_some_and(|count| count != 1) {
        return Err(CliError::unsupported(
            "nonidentity --permute is not implemented until M7",
        ));
    }
    if global.seed.is_some_and(|seed| seed != 0) {
        return Err(CliError::unsupported(
            "nonzero --seed is not implemented until M7 permutation support",
        ));
    }
    if let Some(temperature) = global.temperature
        && (!temperature.is_finite() || temperature <= 0.0)
    {
        return Err(CliError::validation(
            "--temperature must be finite and positive",
        ));
    }
    if global
        .temperature
        .is_some_and(|temperature| temperature != 1.0)
    {
        return Err(CliError::unsupported(
            "nondefault --temperature is not implemented until M7",
        ));
    }
    if global.calibration.is_some() {
        return Err(CliError::unsupported(
            "--calibration is not implemented until M7",
        ));
    }
    Ok(())
}

fn parse_model_spec(global: &GlobalArgs, registry: &ModelRegistry) -> Result<ModelSpec, CliError> {
    let value = global.model.as_deref().unwrap_or(registry.default_model());
    if registry.resolve(value).is_ok() {
        if global.model_sha256.is_some() || global.template_profile.is_some() {
            return Err(CliError::validation(
                "--model-sha256 and --template-profile are only valid for custom artifacts",
            ));
        }
        return Ok(ModelSpec::RegistryId(value.to_owned()));
    }
    let profile = global.template_profile.ok_or_else(|| {
        CliError::validation(
            "custom model artifacts require explicit --template-profile qwen3|qwen3.5|minicpm5",
        )
    })?;
    if let Some(rest) = value.strip_prefix("hf:") {
        let (repo, suffix) = rest.split_once('@').ok_or_else(|| {
            CliError::validation("custom Hub model must be hf:OWNER/REPO@40HEX:FILENAME")
        })?;
        let (revision, file) = suffix.split_once(':').ok_or_else(|| {
            CliError::validation("custom Hub model must be hf:OWNER/REPO@40HEX:FILENAME")
        })?;
        validate_hub_identity(repo, revision, file).map_err(CliError::from_backend_validation)?;
        if !file.to_ascii_lowercase().ends_with(".gguf") {
            return Err(CliError::validation(
                "custom Hub model filename must end in .gguf",
            ));
        }
        let expected_sha256 = global
            .model_sha256
            .clone()
            .ok_or_else(|| CliError::validation("custom Hub models require --model-sha256"))?;
        validate_sha256(&expected_sha256, "--model-sha256")
            .map_err(CliError::from_backend_validation)?;
        return Ok(ModelSpec::Hub {
            repo: repo.to_owned(),
            revision: revision.to_owned(),
            file: file.to_owned(),
            expected_sha256,
            profile,
        });
    }
    let path = std::path::PathBuf::from(value);
    let looks_local = path.is_absolute()
        || value.contains('/')
        || value.starts_with('.')
        || value.to_ascii_lowercase().ends_with(".gguf");
    if !looks_local || !value.to_ascii_lowercase().ends_with(".gguf") {
        return Err(CliError::validation(format!(
            "unknown registered model {value:?}; local models must be a path ending in .gguf"
        )));
    }
    if let Some(expected) = &global.model_sha256 {
        validate_sha256(expected, "--model-sha256").map_err(CliError::from_backend_validation)?;
    }
    Ok(ModelSpec::Local {
        path,
        expected_sha256: global.model_sha256.clone(),
        profile,
    })
}

fn parse_gpu_layers(value: Option<&str>, device: Device) -> Result<GpuLayersRequested, CliError> {
    let requested = match value {
        None if device == Device::Cpu => GpuLayersRequested::Count(0),
        None => GpuLayersRequested::All,
        Some("all") => GpuLayersRequested::All,
        Some(value) => GpuLayersRequested::Count(value.parse::<u32>().map_err(|error| {
            CliError::validation(format!("invalid --gpu-layers {value:?}: {error}"))
        })?),
    };
    if device == Device::Cpu && requested != GpuLayersRequested::Count(0) {
        return Err(CliError::validation(
            "CPU execution requires --gpu-layers 0; KQV/op offload is disabled",
        ));
    }
    Ok(requested)
}

fn default_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|value| u32::try_from(value.get()).unwrap_or(u32::MAX))
        .unwrap_or(1)
}

pub fn decide_items(args: DecideArgs, state: StateValue) -> Result<Vec<Adapter>, CliError> {
    let option_ids = aligned_ids(&args.option_id, args.option.len(), "option")?;
    let options: Vec<_> = option_ids
        .into_iter()
        .zip(args.option)
        .map(|(id, description)| DecisionOption { id, description })
        .collect();
    let count = args.question.len();
    args.question
        .into_iter()
        .enumerate()
        .map(|(index, question)| {
            let id = generated_id(args.id.as_deref(), index, count);
            Decision::new(id, state.clone(), question, options.clone())
                .map(Adapter::Choice)
                .map_err(CliError::from_core_validation)
        })
        .collect()
}

pub fn noul_item(args: NoulArgs, state: StateValue) -> Result<Adapter, CliError> {
    Noul::new(
        args.id.unwrap_or_else(|| "decision-1".to_owned()),
        state,
        args.question,
    )
    .map(Adapter::Noul)
    .map_err(CliError::from_core_validation)
}

pub fn score_item(args: ScoreArgs, state: StateValue) -> Result<Adapter, CliError> {
    let ids = aligned_ids(&args.level_id, args.level.len(), "level")?;
    if !args.level_value.is_empty() && args.level_value.len() != args.level.len() {
        return Err(CliError::validation(format!(
            "--level-value count {} does not match --level count {}",
            args.level_value.len(),
            args.level.len()
        )));
    }
    let values = if args.level_value.is_empty() {
        (0..args.level.len()).map(|index| index as f64).collect()
    } else {
        args.level_value
    };
    let levels = ids
        .into_iter()
        .zip(args.level)
        .zip(values)
        .map(|((id, description), value)| ScoreLevel {
            id,
            description,
            value,
        })
        .collect();
    Score::new(
        args.id.unwrap_or_else(|| "decision-1".to_owned()),
        state,
        args.question,
        levels,
    )
    .map(Adapter::Score)
    .map_err(CliError::from_core_validation)
}

fn aligned_ids(explicit: &[String], count: usize, prefix: &str) -> Result<Vec<String>, CliError> {
    if explicit.is_empty() {
        return Ok((1..=count)
            .map(|index| format!("{prefix}-{index}"))
            .collect());
    }
    if explicit.len() != count {
        return Err(CliError::validation(format!(
            "--{prefix}-id count {} does not match --{prefix} count {count}",
            explicit.len()
        )));
    }
    Ok(explicit.to_vec())
}

fn generated_id(base: Option<&str>, index: usize, count: usize) -> String {
    match (base, count) {
        (Some(base), 1) => base.to_owned(),
        (Some(base), _) => format!("{base}/{}", index + 1),
        (None, _) => format!("decision-{}", index + 1),
    }
}

pub fn validate_unique_ids(items: &[Adapter]) -> Result<(), CliError> {
    let mut ids = HashSet::with_capacity(items.len());
    for item in items {
        if !ids.insert(item.decision().id.as_str()) {
            return Err(CliError::validation(format!(
                "duplicate decision ID {:?}",
                item.decision().id
            )));
        }
    }
    Ok(())
}

pub fn validate_shared_states(rows: &[Decision]) -> Result<(), CliError> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    let expected =
        python_json_dumps(first.state.as_value()).map_err(CliError::from_core_validation)?;
    for row in &rows[1..] {
        let actual =
            python_json_dumps(row.state.as_value()).map_err(CliError::from_core_validation)?;
        if actual.as_bytes() != expected.as_bytes() {
            return Err(CliError::validation(
                "shared run requires every state to have identical serialized bytes and object order",
            ));
        }
    }
    Ok(())
}

pub fn mode_for_decide(question_count: usize) -> ExecutionMode {
    if question_count > 1 {
        ExecutionMode::Shared
    } else {
        ExecutionMode::Direct
    }
}

pub fn mode_from_arg(mode: ModeArg) -> ExecutionMode {
    match mode {
        ModeArg::Direct => ExecutionMode::Direct,
        ModeArg::Serial => ExecutionMode::Serial,
        ModeArg::Shared => ExecutionMode::Shared,
        ModeArg::Batch => ExecutionMode::Batch,
    }
}

pub fn fallback_reason(mode: ExecutionMode) -> Option<&'static str> {
    match mode {
        ExecutionMode::Shared => Some(SHARED_FALLBACK_REASON),
        ExecutionMode::Batch => Some(BATCH_FALLBACK_REASON),
        ExecutionMode::Direct | ExecutionMode::Serial => None,
    }
}

pub fn score_item_with(
    scorer: &mut dyn DecisionScorer,
    item: &Adapter,
    requested_mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
) -> Result<Readout, CliError> {
    score_item_with_reason(
        scorer,
        item,
        requested_mode,
        confidence,
        group_id,
        fallback_reason(requested_mode),
    )
}

pub fn score_item_with_reason(
    scorer: &mut dyn DecisionScorer,
    item: &Adapter,
    requested_mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
    fallback: Option<&str>,
) -> Result<Readout, CliError> {
    let mut readout = scorer.score_direct(item.decision().clone())?;
    readout.execution.requested_mode = requested_mode;
    readout.execution.effective_mode = match requested_mode {
        ExecutionMode::Direct => ExecutionMode::Direct,
        ExecutionMode::Serial | ExecutionMode::Shared | ExecutionMode::Batch => {
            ExecutionMode::Serial
        }
    };
    readout.execution.fallback_reason =
        if readout.execution.requested_mode == readout.execution.effective_mode {
            None
        } else {
            fallback.map(str::to_owned).or_else(|| {
                Some("requested execution mode used serial full-prompt fallback".to_owned())
            })
        };
    readout.execution.group_id = group_id.map(str::to_owned);
    readout.model.serving_config = Some(
        match readout.execution.effective_mode {
            ExecutionMode::Direct => "llama-direct-v1",
            ExecutionMode::Serial => "llama-serial-full-prompt-v1",
            ExecutionMode::Shared => "llama-state-prefix-parallel-v1",
            ExecutionMode::Batch => "llama-independent-batch-v1",
        }
        .to_owned(),
    );
    adapt_group_readout(item, readout, confidence, group_id)
}

pub fn adapt_group_readout(
    item: &Adapter,
    mut readout: Readout,
    confidence: bool,
    group_id: Option<&str>,
) -> Result<Readout, CliError> {
    readout.execution.group_id = group_id.map(str::to_owned);
    let mut readout = item.adapt(readout)?;
    if confidence {
        readout.confidence =
            Some(normalized_margin(&readout.probabilities).map_err(CliError::from_runtime_core)?);
        readout.confidence_status = Some(CONFIDENCE_STATUS.to_owned());
    }
    readout.validate().map_err(CliError::from_runtime_core)?;
    Ok(readout)
}

pub fn check_require_shared(global: &GlobalArgs, mode: ExecutionMode) -> Result<(), CliError> {
    if global.require_shared {
        if mode != ExecutionMode::Shared {
            return Err(CliError::validation(
                "--require-shared is only valid when shared mode is requested",
            ));
        }
        if !cfg!(feature = "native") {
            return Err(CliError::unsupported(
                "--require-shared cannot be satisfied by a backend-disabled build",
            ));
        }
        // Exact receipt eligibility is checked after artifact resolution but
        // before the first inference call.
        return Ok(());
    }
    Ok(())
}

pub fn preflight_output(output: &Path, input: Option<&Path>) -> Result<(), CliError> {
    if std::fs::symlink_metadata(output).is_ok() {
        return Err(CliError::validation(format!(
            "create-only output already exists: {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = std::fs::canonicalize(parent).map_err(|error| {
        CliError::validation(format!(
            "output parent {} cannot be resolved: {error}",
            parent.display()
        ))
    })?;
    if !std::fs::metadata(&canonical_parent)
        .map_err(|error| CliError::validation(error.to_string()))?
        .is_dir()
    {
        return Err(CliError::validation("output parent must be a directory"));
    }
    let name = output.file_name().ok_or_else(|| {
        CliError::validation(format!("output path {} has no filename", output.display()))
    })?;
    let canonical_target = canonical_parent.join(name);
    if let Some(input) = input {
        let canonical_input = std::fs::canonicalize(input).map_err(|error| {
            CliError::validation(format!(
                "input {} cannot be resolved: {error}",
                input.display()
            ))
        })?;
        if canonical_input == canonical_target {
            return Err(CliError::validation(
                "input and output must not identify the same file",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct ModelsOutput {
    pub schema: &'static str,
    pub default_model: String,
    pub cache_root: String,
    pub models: Vec<ModelStatus>,
}

#[derive(Debug, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub source: String,
    pub revision: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub quant: String,
    pub template_profile: openjev_core::PromptProfile,
    pub cached: bool,
    pub verified: bool,
    pub cache_status: String,
    pub path: Option<String>,
    pub support_status: &'static str,
    pub shared_probe_status: &'static str,
    pub batch_probe_status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelPathOutput {
    pub schema: &'static str,
    pub id: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub integrity: String,
    pub cache_hit: bool,
}

pub fn models_list(global: &GlobalArgs) -> Result<ModelsOutput, CliError> {
    reject_models_irrelevant(global, "list")?;
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let cache = ModelCache::from_precedence(global.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    let mut models = Vec::with_capacity(registry.list().len());
    for entry in registry.list() {
        let (cached, verified, cache_status, path) = match cache.inspect(entry) {
            Ok(Some(artifact)) => (
                true,
                true,
                "verified-manifest-sha256".to_owned(),
                Some(artifact.path.display().to_string()),
            ),
            Ok(None) => (false, false, "missing".to_owned(), None),
            Err(error) => (true, false, format!("verification-failed: {error}"), None),
        };
        models.push(ModelStatus {
            id: entry.id.clone(),
            source: entry.repo.clone(),
            revision: entry.revision.clone(),
            file: entry.file.clone(),
            bytes: entry.bytes,
            sha256: entry.sha256.clone(),
            quant: entry.quant.clone(),
            template_profile: entry.profile,
            cached,
            verified,
            cache_status,
            path,
            support_status: "registered-runtime-load-not-attempted-by-list",
            shared_probe_status: "configuration-specific-local-receipt-required",
            batch_probe_status: "configuration-specific-local-receipt-required",
        });
    }
    Ok(ModelsOutput {
        schema: "openjev-models-v1",
        default_model: registry.default_model().to_owned(),
        cache_root: cache.root().display().to_string(),
        models,
    })
}

pub(crate) fn reject_models_irrelevant(
    global: &GlobalArgs,
    operation: &str,
) -> Result<(), CliError> {
    let scoring_flag = global.device.is_some()
        || global.gpu_layers.is_some()
        || global.threads.is_some()
        || global.n_ctx.is_some()
        || global.max_tokens.is_some()
        || global.max_context_tokens.is_some()
        || global.n_batch.is_some()
        || global.n_ubatch.is_some()
        || global.max_sequences.is_some()
        || global.require_shared
        || global.permute.is_some()
        || global.seed.is_some()
        || global.temperature.is_some()
        || global.calibration.is_some()
        || global.confidence;
    if scoring_flag {
        return Err(CliError::validation(format!(
            "scoring options are irrelevant to models {operation}"
        )));
    }
    if operation == "list"
        && (global.model.is_some()
            || global.model_sha256.is_some()
            || global.template_profile.is_some()
            || global.offline)
    {
        return Err(CliError::validation(
            "model selection, template, hash, and offline flags are irrelevant to models list",
        ));
    }
    Ok(())
}

#[cfg(feature = "native")]
pub fn probe_configuration(
    config: &ScoringConfig,
    artifact_sha256: &str,
    profile: openjev_core::PromptProfile,
) -> ProbeConfiguration {
    ProbeConfiguration {
        artifact_sha256: artifact_sha256.to_owned(),
        native_pin: NATIVE_PIN.to_owned(),
        probe_suite_version: PROBE_SUITE_VERSION.to_owned(),
        device: config.device,
        gpu_layers_requested: config.gpu_layers,
        offload_kqv: config.device != Device::Cpu,
        op_offload: config.device != Device::Cpu,
        threads: config.threads,
        n_ctx: config.n_ctx,
        max_tokens: config.max_tokens,
        max_context_tokens: config.max_context_tokens,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_seq_max: config.max_sequences,
        kv_unified: true,
        profile,
    }
}

#[cfg(feature = "native")]
pub fn require_probe_eligibility(
    config: &ScoringConfig,
    mode: ProbeMode,
) -> Result<String, CliError> {
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let cache = ModelCache::from_precedence(config.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    let resolved = openjev_llama::resolve_model_spec(
        &registry,
        &cache,
        &config.model,
        CacheOptions {
            offline: config.offline,
            repair: false,
        },
    )
    .map_err(CliError::from_backend_runtime)?;
    let configuration = probe_configuration(
        config,
        resolved.model().artifact_sha256(),
        resolved.model().profile(),
    );
    load_passing_receipt(&cache, resolved.model().id(), mode, &configuration)
        .map(|eligibility| eligibility.probe_id().to_owned())
        .map_err(|error| {
            CliError::unsupported(format!(
                "--require-shared requires a passing exact-configuration local receipt: {error}"
            ))
        })
}

#[cfg(feature = "native")]
pub struct NativeScorer {
    engine: Option<openjev_llama::EngineHandle>,
    shared_probe: Result<ProbeEligibility, String>,
    batch_probe: Result<ProbeEligibility, String>,
}

#[cfg(feature = "native")]
impl NativeScorer {
    pub fn load(config: &ScoringConfig) -> Result<Self, CliError> {
        let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
        let cache = ModelCache::from_precedence(config.cache_dir.as_deref())
            .map_err(CliError::from_backend_runtime)?;
        let resolved = openjev_llama::resolve_model_spec(
            &registry,
            &cache,
            &config.model,
            CacheOptions {
                offline: config.offline,
                repair: false,
            },
        )
        .map_err(CliError::from_backend_runtime)?;
        let probe_configuration = probe_configuration(
            config,
            resolved.model().artifact_sha256(),
            resolved.model().profile(),
        );
        let model_id = resolved.model().id().to_owned();
        let eligibility = |mode| {
            load_passing_receipt(&cache, &model_id, mode, &probe_configuration)
                .map_err(|error| error.to_string())
        };
        let shared_probe = eligibility(ProbeMode::Shared);
        let batch_probe = eligibility(ProbeMode::Batch);
        let options = openjev_llama::EngineOptions {
            device: config.device,
            gpu_layers: config.gpu_layers,
            threads: config.threads,
            n_ctx: config.n_ctx,
            max_tokens: config.max_tokens,
            max_context_tokens: config.max_context_tokens,
            n_batch: config.n_batch,
            n_ubatch: config.n_ubatch,
            max_sequences: config.max_sequences,
        };
        options
            .validate()
            .map_err(CliError::from_backend_validation)?;
        let (model, artifact) = resolved.into_parts();
        let engine = openjev_llama::EngineHandle::spawn_resolved(model, artifact, options)
            .map_err(CliError::from_backend_runtime)?;
        Ok(Self {
            engine: Some(engine),
            shared_probe,
            batch_probe,
        })
    }
}

#[cfg(feature = "native")]
impl DecisionScorer for NativeScorer {
    fn score_direct(&mut self, decision: Decision) -> Result<Readout, CliError> {
        self.engine
            .as_ref()
            .ok_or_else(|| CliError::runtime("worker", "owner worker is shut down"))?
            .score_direct(decision)
            .map_err(CliError::from_backend_runtime)
    }

    fn probe_id(&self, mode: ExecutionMode) -> Result<String, String> {
        match mode {
            ExecutionMode::Shared => self
                .shared_probe
                .as_ref()
                .map(|eligibility| eligibility.probe_id().to_owned())
                .map_err(Clone::clone),
            ExecutionMode::Batch => self
                .batch_probe
                .as_ref()
                .map(|eligibility| eligibility.probe_id().to_owned())
                .map_err(Clone::clone),
            ExecutionMode::Direct | ExecutionMode::Serial => {
                Err("direct/serial execution does not use a probe receipt".to_owned())
            }
        }
    }

    fn score_shared(
        &mut self,
        decisions: Vec<Decision>,
        probe_id: String,
    ) -> Result<Vec<Readout>, CliError> {
        let state = decisions
            .first()
            .ok_or_else(|| CliError::validation("shared group must not be empty"))?
            .state
            .clone();
        let questions = decisions
            .into_iter()
            .map(|decision| Question::new(decision.id, decision.question, decision.options))
            .collect::<openjev_core::types::Result<Vec<_>>>()
            .map_err(CliError::from_core_validation)?;
        let eligibility = self
            .shared_probe
            .as_ref()
            .map_err(|reason| CliError::runtime("shared_unavailable", reason.clone()))?;
        if eligibility.probe_id() != probe_id {
            return Err(CliError::runtime(
                "shared_unavailable",
                "selected shared receipt ID changed before dispatch",
            ));
        }
        self.engine
            .as_ref()
            .ok_or_else(|| CliError::runtime("worker", "owner worker is shut down"))?
            .score_shared(state, questions, eligibility.clone())
            .map_err(CliError::from_backend_runtime)
    }

    fn score_batch(
        &mut self,
        decisions: Vec<Decision>,
        probe_id: String,
    ) -> Result<Vec<Readout>, CliError> {
        let eligibility = self
            .batch_probe
            .as_ref()
            .map_err(|reason| CliError::runtime("batch_unavailable", reason.clone()))?;
        if eligibility.probe_id() != probe_id {
            return Err(CliError::runtime(
                "batch_unavailable",
                "selected batch receipt ID changed before dispatch",
            ));
        }
        self.engine
            .as_ref()
            .ok_or_else(|| CliError::runtime("worker", "owner worker is shut down"))?
            .score_batch(decisions, eligibility.clone())
            .map_err(CliError::from_backend_runtime)
    }

    fn shutdown(&mut self) -> Result<(), CliError> {
        if let Some(engine) = self.engine.take() {
            engine.shutdown().map_err(CliError::from_backend_runtime)?;
        }
        Ok(())
    }
}

#[cfg(feature = "native")]
pub fn models_resolve(
    global: &GlobalArgs,
    id: &str,
    pull: bool,
    repair: bool,
) -> Result<ModelPathOutput, CliError> {
    reject_models_irrelevant(global, if pull { "pull" } else { "path" })?;
    if global.model.is_some() {
        return Err(CliError::validation(
            "use the models pull/path positional ID instead of global --model",
        ));
    }
    let mut selection = global.clone();
    selection.model = Some(id.to_owned());
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let spec = parse_model_spec(&selection, &registry)?;
    if repair && !matches!(&spec, ModelSpec::RegistryId(_)) {
        return Err(CliError::unsupported(
            "--repair is currently limited to registered cache-owned artifacts",
        ));
    }
    let cache = ModelCache::from_precedence(global.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    let resolved = openjev_llama::resolve_model_spec(
        &registry,
        &cache,
        &spec,
        CacheOptions {
            offline: if pull { global.offline } else { true },
            repair: pull && repair,
        },
    )
    .map_err(CliError::from_backend_runtime)?;
    let id = resolved.model().id().to_owned();
    let integrity = resolved.model().integrity();
    let (_, artifact) = resolved.into_parts();
    Ok(ModelPathOutput {
        schema: "openjev-model-path-v1",
        id,
        path: artifact.path.display().to_string(),
        bytes: artifact.bytes,
        sha256: artifact.sha256,
        integrity: match integrity {
            openjev_core::Integrity::ManifestSha256 => "manifest-sha256",
            openjev_core::Integrity::CallerSha256 => "caller-sha256",
            openjev_core::Integrity::LocalUnverified => "local-unverified",
        }
        .to_owned(),
        cache_hit: artifact.cache_hit,
    })
}

#[cfg(feature = "native")]
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ProbeCommandReport {
    pub schema: String,
    pub process_status: String,
    pub receipt: Option<ProbeReceipt>,
    pub receipt_path: Option<String>,
    pub enabled: bool,
    pub failure_reason: Option<String>,
}

#[cfg(feature = "native")]
struct OwnedProbeCase {
    id: &'static str,
    decisions: Vec<Decision>,
    repeats: usize,
    require_over_512: bool,
}

#[cfg(feature = "native")]
pub fn run_probe_child(
    global: &GlobalArgs,
    id: &str,
    mode: ProbeMode,
) -> Result<ProbeReceipt, CliError> {
    if std::env::var("OPENJEV_INTERNAL_PROBE_CRASH").as_deref() == Ok("1") {
        std::process::abort();
    }
    let mut selected = global.clone();
    selected.model = Some(id.to_owned());
    let config = scoring_config(&selected)?;
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let cache = ModelCache::from_precedence(config.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    let resolved = openjev_llama::resolve_model_spec(
        &registry,
        &cache,
        &config.model,
        CacheOptions {
            offline: config.offline,
            repair: false,
        },
    )
    .map_err(CliError::from_backend_runtime)?;
    let model_id = resolved.model().id().to_owned();
    let configuration = probe_configuration(
        &config,
        resolved.model().artifact_sha256(),
        resolved.model().profile(),
    );
    let expected_probe_id =
        probe_id(mode, &configuration).map_err(CliError::from_backend_runtime)?;
    let options = openjev_llama::EngineOptions {
        device: config.device,
        gpu_layers: config.gpu_layers,
        threads: config.threads,
        n_ctx: config.n_ctx,
        max_tokens: config.max_tokens,
        max_context_tokens: config.max_context_tokens,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        max_sequences: config.max_sequences,
    };
    let (model, artifact) = resolved.into_parts();
    let engine = openjev_llama::EngineHandle::spawn_resolved(model, artifact, options)
        .map_err(CliError::from_backend_runtime)?;
    let cases = probe_cases(&config, mode)?;
    let mut results = Vec::with_capacity(cases.len());
    let mut decisive_failure = None;
    for case in cases {
        if let Some(reason) = &decisive_failure {
            results.push(ProbeCaseResult {
                id: case.id.to_owned(),
                status: ProbeCaseStatus::UnrunAfterDecisiveFailure,
                rows: 0,
                max_abs_slot_logit: None,
                max_probability_delta: None,
                same_first_argmax: None,
                detail: Some(format!("not run after decisive failure: {reason}")),
            });
            continue;
        }
        match run_one_probe_case(&engine, mode, &expected_probe_id, &case) {
            Ok(result) if result.status == ProbeCaseStatus::Passed => results.push(result),
            Ok(result) => {
                decisive_failure = result.detail.clone();
                results.push(result);
            }
            Err(error) => {
                let reason = error.to_string();
                decisive_failure = Some(reason.clone());
                results.push(ProbeCaseResult {
                    id: case.id.to_owned(),
                    status: ProbeCaseStatus::Failed,
                    rows: 0,
                    max_abs_slot_logit: None,
                    max_probability_delta: None,
                    same_first_argmax: None,
                    detail: Some(reason),
                });
            }
        }
    }
    let shutdown_error = engine.shutdown().err().map(|error| error.to_string());
    let failure_reason = shutdown_error.or(decisive_failure);
    ProbeReceipt::new(model_id, mode, configuration, results, failure_reason)
        .map_err(CliError::from_backend_runtime)
}

#[cfg(feature = "native")]
pub fn prepare_probe_publication(
    global: &GlobalArgs,
    id: &str,
    mode: ProbeMode,
) -> Result<ProbePublication, CliError> {
    let mut selected = global.clone();
    selected.model = Some(id.to_owned());
    let config = scoring_config(&selected)?;
    let registry = ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let cache = ModelCache::from_precedence(config.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    // This resolves and verifies only model identity/artifact bytes. It does not
    // initialize llama.cpp, load a model, or create a context in the parent.
    let resolved = openjev_llama::resolve_model_spec(
        &registry,
        &cache,
        &config.model,
        CacheOptions {
            offline: config.offline,
            repair: false,
        },
    )
    .map_err(CliError::from_backend_runtime)?;
    let expected = probe_configuration(
        &config,
        resolved.model().artifact_sha256(),
        resolved.model().profile(),
    );
    begin_probe_publication(&cache, resolved.model().id(), mode, &expected)
        .map_err(CliError::from_backend_runtime)
}

#[cfg(feature = "native")]
fn run_one_probe_case(
    engine: &openjev_llama::EngineHandle,
    mode: ProbeMode,
    probe_id: &str,
    case: &OwnedProbeCase,
) -> Result<ProbeCaseResult, CliError> {
    let mut baseline = Vec::with_capacity(case.decisions.len());
    for decision in &case.decisions {
        baseline.push(
            engine
                .score_direct(decision.clone())
                .map_err(CliError::from_backend_runtime)?,
        );
    }
    if case.require_over_512 && baseline.iter().all(|row| row.input_tokens <= 512) {
        return Ok(ProbeCaseResult {
            id: case.id.to_owned(),
            status: ProbeCaseStatus::Failed,
            rows: u32::try_from(case.decisions.len()).unwrap_or(u32::MAX),
            max_abs_slot_logit: None,
            max_probability_delta: None,
            same_first_argmax: None,
            detail: Some(
                "owned long case did not produce a full prompt over 512 tokens".to_owned(),
            ),
        });
    }
    let mut max_logit = 0.0_f64;
    let mut max_probability = 0.0_f64;
    let mut same_argmax = true;
    for _ in 0..case.repeats {
        let candidate = match mode {
            ProbeMode::Shared => {
                let state = case.decisions[0].state.clone();
                if case.decisions.iter().any(|row| row.state != state) {
                    return Err(CliError::validation(
                        "owned shared probe case contains nonidentical states",
                    ));
                }
                let questions = case
                    .decisions
                    .iter()
                    .map(|row| {
                        Question::new(row.id.clone(), row.question.clone(), row.options.clone())
                    })
                    .collect::<openjev_core::types::Result<Vec<_>>>()
                    .map_err(CliError::from_core_validation)?;
                engine
                    .probe_shared_candidate(state, questions, probe_id.to_owned())
                    .map_err(CliError::from_backend_runtime)?
            }
            ProbeMode::Batch => engine
                .probe_batch_candidate(case.decisions.clone(), probe_id.to_owned())
                .map_err(CliError::from_backend_runtime)?,
        };
        if candidate.len() != baseline.len()
            || candidate
                .iter()
                .zip(&baseline)
                .any(|(actual, direct)| actual.id != direct.id)
        {
            return Ok(ProbeCaseResult {
                id: case.id.to_owned(),
                status: ProbeCaseStatus::Failed,
                rows: u32::try_from(case.decisions.len()).unwrap_or(u32::MAX),
                max_abs_slot_logit: None,
                max_probability_delta: None,
                same_first_argmax: Some(false),
                detail: Some("output order or identity differs from direct baseline".to_owned()),
            });
        }
        for (actual, direct) in candidate.iter().zip(&baseline) {
            if actual.option_logits.len() != direct.option_logits.len()
                || actual.probabilities.len() != direct.probabilities.len()
            {
                return Err(CliError::runtime(
                    "probe",
                    "candidate/direct vector lengths differ",
                ));
            }
            for (left, right) in actual.option_logits.iter().zip(&direct.option_logits) {
                max_logit = max_logit.max((left - right).abs());
            }
            for (left, right) in actual.probabilities.iter().zip(&direct.probabilities) {
                max_probability = max_probability.max((left - right).abs());
            }
            same_argmax &= actual.choice_index == direct.choice_index;
        }
    }
    let passed =
        max_logit <= MAX_ABS_SLOT_LOGIT && max_probability <= MAX_PROBABILITY_DELTA && same_argmax;
    Ok(ProbeCaseResult {
        id: case.id.to_owned(),
        status: if passed {
            ProbeCaseStatus::Passed
        } else {
            ProbeCaseStatus::Failed
        },
        rows: u32::try_from(case.decisions.len()).unwrap_or(u32::MAX),
        max_abs_slot_logit: Some(max_logit),
        max_probability_delta: Some(max_probability),
        same_first_argmax: Some(same_argmax),
        detail: (!passed).then(|| {
            format!(
                "frozen tolerance failure: max_abs_slot_logit={max_logit}, max_probability_delta={max_probability}, same_first_argmax={same_argmax}"
            )
        }),
    })
}

#[cfg(feature = "native")]
fn probe_cases(config: &ScoringConfig, mode: ProbeMode) -> Result<Vec<OwnedProbeCase>, CliError> {
    let binary = probe_options(2);
    let ternary = probe_options(3);
    let sixteen = probe_options(16);
    let short = StateValue::string("probe short state").map_err(CliError::from_core_validation)?;
    let changed = StateValue::string("probe changed state with independent contents")
        .map_err(CliError::from_core_validation)?;
    let long_state = StateValue::string(format!(
        "long owned evidence {}",
        "evidence-segment ".repeat(700)
    ))
    .map_err(CliError::from_core_validation)?;
    let suffix_words = usize::try_from(config.n_batch)
        .unwrap_or(512)
        .saturating_add(160)
        .min(2_000);
    let long_question = format!(
        "Evaluate this deliberately long ragged criterion: {}",
        "criterion-segment ".repeat(suffix_words)
    );
    let one = vec![probe_decision(
        "binary-1",
        short.clone(),
        "short binary",
        binary.clone(),
    )?];
    let ragged = vec![
        probe_decision(
            "ragged-1",
            short.clone(),
            "short three-way",
            ternary.clone(),
        )?,
        probe_decision("ragged-2", short.clone(), &long_question, ternary.clone())?,
    ];
    let many = (1..=21)
        .map(|index| {
            let state = if mode == ProbeMode::Batch {
                StateValue::string(format!(
                    "{} independent-batch-state-{index}",
                    long_state.as_value().as_str().unwrap_or("owned long state")
                ))
                .map_err(CliError::from_core_validation)?
            } else {
                long_state.clone()
            };
            probe_decision(
                &format!("wide-{index:02}"),
                state,
                &format!(
                    "sixteen-way criterion {index} {}",
                    "ragged ".repeat(index % 7)
                ),
                sixteen.clone(),
            )
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let changed_case = vec![
        probe_decision(
            "changed-1",
            changed.clone(),
            "changed state first",
            binary.clone(),
        )?,
        probe_decision(
            "changed-2",
            if mode == ProbeMode::Batch {
                StateValue::string("a second distinct independent batch state")
                    .map_err(CliError::from_core_validation)?
            } else {
                changed
            },
            "changed state second",
            binary.clone(),
        )?,
    ];
    let cycles = (1..=config.max_sequences.saturating_add(1))
        .map(|index| {
            let state = if mode == ProbeMode::Batch {
                StateValue::string(format!("independent cycle state {index}"))
                    .map_err(CliError::from_core_validation)?
            } else {
                short.clone()
            };
            probe_decision(
                &format!("cycle-{index:02}"),
                state,
                &format!("copy clear cycle {index}"),
                binary.clone(),
            )
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    Ok(vec![
        OwnedProbeCase {
            id: "binary-short-1-branch",
            decisions: one,
            repeats: 1,
            require_over_512: false,
        },
        OwnedProbeCase {
            id: "three-way-ragged-2-branches-multichunk",
            decisions: ragged,
            repeats: 1,
            require_over_512: true,
        },
        OwnedProbeCase {
            id: "sixteen-way-long-state-21-branches",
            decisions: many,
            repeats: 1,
            require_over_512: true,
        },
        OwnedProbeCase {
            id: "changed-state-isolation",
            decisions: changed_case,
            repeats: 1,
            require_over_512: false,
        },
        OwnedProbeCase {
            id: "repeated-copy-clear-cycles",
            decisions: cycles,
            repeats: 3,
            require_over_512: false,
        },
    ])
}

#[cfg(feature = "native")]
fn probe_options(count: usize) -> Vec<DecisionOption> {
    (1..=count)
        .map(|index| DecisionOption {
            id: format!("option-{index}"),
            description: format!("Owned deterministic option {index}"),
        })
        .collect()
}

#[cfg(feature = "native")]
fn probe_decision(
    id: &str,
    state: StateValue,
    question: &str,
    options: Vec<DecisionOption>,
) -> Result<Decision, CliError> {
    Decision::new(id, state, question, options).map_err(CliError::from_core_validation)
}

#[cfg(test)]
mod tests {
    use openjev_core::{
        ExecutionMetadata, GpuLayersStatus, Integrity, ModelMetadata, Primitive, PromptProfile,
        TemplateMetadataStatus, standard_limitations,
    };

    use super::*;

    struct DeterministicScorer;

    impl DecisionScorer for DeterministicScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<Readout, CliError> {
            let option_ids: Vec<_> = decision
                .options
                .iter()
                .map(|option| option.id.clone())
                .collect();
            let count = option_ids.len();
            let probabilities = if count == 2 {
                vec![0.25, 0.75]
            } else {
                vec![1.0 / count as f64; count]
            };
            let choice_index = openjev_core::first_argmax(&probabilities).unwrap();
            let readout = Readout {
                schema: "openjev-readout-v1".to_owned(),
                id: decision.id,
                primitive: Primitive::Choice,
                choice: option_ids[choice_index].clone(),
                choice_index,
                option_ids,
                probabilities,
                option_logits: (0..count).map(|index| index as f64).collect(),
                answer_token_ids: (1..=u32::try_from(count).unwrap()).collect(),
                allowed_token_mass: 0.5,
                full_vocab_argmax_id: 1,
                full_vocab_log_normalizer: 2.0,
                input_tokens: 10,
                forward_seconds: Some(0.01),
                total_seconds: Some(0.02),
                prompt_sha256: "0".repeat(64),
                prompt_version: "direct-options-v1".to_owned(),
                model: ModelMetadata {
                    id: "test".to_owned(),
                    source: "local".to_owned(),
                    revision: "sha256:test".to_owned(),
                    file: "test.gguf".to_owned(),
                    quant: "test".to_owned(),
                    backend: "deterministic-test-scorer".to_owned(),
                    artifact_sha256: "1".repeat(64),
                    integrity: Integrity::LocalUnverified,
                    dtype: "test".to_owned(),
                    native_reference: None,
                    template_profile: PromptProfile::Qwen3,
                    template_sha256: None,
                    template_override: true,
                    template_status: TemplateMetadataStatus::OverrideUnverified,
                    template_equivalence_evidence: None,
                    serving_config: Some("llama-direct-v1".to_owned()),
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
                    device_name: "deterministic-test-device".to_owned(),
                    gpu_layers_requested: GpuLayersRequested::Count(0),
                    gpu_layers_actual: Some(0),
                    gpu_layers_status: GpuLayersStatus::KnownDisabled,
                    threads: 1,
                    n_ctx_requested: None,
                    n_ctx_actual: 128,
                    max_tokens: 128,
                    n_batch: 128,
                    n_ubatch: 128,
                    n_seq_max: 1,
                    kv_unified: true,
                    waves: 1,
                    probe_id: None,
                    run_id: "test-run".to_owned(),
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
            };
            readout.validate().unwrap();
            Ok(readout)
        }
    }

    #[test]
    fn deterministic_injection_covers_noul_score_confidence_and_shared_fallback() {
        let state = StateValue::string(" exact state ").unwrap();
        let noul = Adapter::Noul(Noul::new("n", state.clone(), "q").unwrap());
        let mut scorer = DeterministicScorer;
        let noul = score_item_with(
            &mut scorer,
            &noul,
            ExecutionMode::Shared,
            true,
            Some("group"),
        )
        .unwrap();
        assert_eq!(noul.primitive, Primitive::Noul);
        assert_eq!(noul.p_yes, Some(0.25));
        assert_eq!(noul.execution.requested_mode, ExecutionMode::Shared);
        assert_eq!(noul.execution.effective_mode, ExecutionMode::Serial);
        assert_eq!(
            noul.execution.fallback_reason.as_deref(),
            Some(SHARED_FALLBACK_REASON)
        );
        assert_eq!(noul.confidence_status.as_deref(), Some(CONFIDENCE_STATUS));

        let score = Adapter::Score(
            Score::new(
                "s",
                state,
                "q",
                vec![
                    ScoreLevel {
                        id: "low".to_owned(),
                        description: "Low".to_owned(),
                        value: -2.0,
                    },
                    ScoreLevel {
                        id: "high".to_owned(),
                        description: "High".to_owned(),
                        value: 6.0,
                    },
                ],
            )
            .unwrap(),
        );
        let score =
            score_item_with(&mut scorer, &score, ExecutionMode::Serial, false, None).unwrap();
        assert_eq!(score.primitive, Primitive::Score);
        assert_eq!(score.expected_value, Some(4.0));
        assert_eq!(score.argmax_level.as_deref(), Some("high"));
        assert_eq!(score.execution.requested_mode, ExecutionMode::Serial);
        assert_eq!(score.execution.effective_mode, ExecutionMode::Serial);
        assert!(score.execution.fallback_reason.is_none());
    }
}
