use std::{
    collections::HashSet,
    fmt::Write as _,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::JoinHandle,
    time::Instant,
};

use llama_cpp_2::{
    LogOptions, list_llama_ggml_backend_devices,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    send_logs_to_tracing,
    token::LlamaToken,
};
use openjev_core::{
    DIRECT_READOUT, Decision, Device, ExecutionMetadata, ExecutionMode, GpuLayersRequested,
    GpuLayersStatus, ModelMetadata, NativeReference, NumericReadout, PROBABILITY_STATUS, Primitive,
    PromptProfile, Question, Readout, SharedTiming, SlotTokenizer, StateValue,
    TemplateMetadataStatus, prepare_prompt, read_logits, standard_limitations, state_prefix_text,
    verify_slots,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    BackendError, ModelEntry, NATIVE_PIN, PROBE_SUITE_VERSION, ProbeConfiguration,
    ProbeEligibility, ProbeMode, Result, RuntimeModelSpec, VerifiedArtifact, cache::hash_file,
    probe_id,
};

const LLAMA_CPP_COMMIT: &str = "e79e4bf660e19f2ad851e06c6913f7a8c5852621";
static RUN_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub device: Device,
    pub gpu_layers: GpuLayersRequested,
    pub threads: u32,
    pub n_ctx: Option<u32>,
    pub max_tokens: u32,
    pub max_context_tokens: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Maximum context sequence slots. Shared mode reserves slot 0 for the
    /// immutable prefix, so at most `max_sequences - 1` branches are active.
    pub max_sequences: u32,
}

impl EngineOptions {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("threads", self.threads),
            ("max_tokens", self.max_tokens),
            ("max_context_tokens", self.max_context_tokens),
            ("n_batch", self.n_batch),
            ("n_ubatch", self.n_ubatch),
            ("max_sequences", self.max_sequences),
        ] {
            if value == 0 {
                return Err(BackendError::Configuration(format!(
                    "{name} must be positive"
                )));
            }
        }
        if self.max_sequences > 64 {
            return Err(BackendError::Configuration(
                "max_sequences must not exceed the validated v1 limit of 64".to_owned(),
            ));
        }
        if self.n_ctx == Some(0) {
            return Err(BackendError::Configuration(
                "n_ctx must be positive when specified".to_owned(),
            ));
        }
        if self.n_ubatch > self.n_batch {
            return Err(BackendError::Configuration(
                "n_ubatch must not exceed n_batch".to_owned(),
            ));
        }
        match self.device {
            Device::Cpu => {
                if self.gpu_layers != GpuLayersRequested::Count(0) {
                    return Err(BackendError::Configuration(
                        "CPU execution requires gpu_layers=0".to_owned(),
                    ));
                }
            }
            Device::Metal if !cfg!(feature = "metal") => {
                return Err(BackendError::Configuration(
                    "Metal execution requires the metal Cargo feature".to_owned(),
                ));
            }
            Device::Cuda if !cfg!(feature = "cuda") => {
                return Err(BackendError::Configuration(
                    "CUDA execution requires the cuda Cargo feature".to_owned(),
                ));
            }
            Device::Metal | Device::Cuda => {}
        }
        Ok(())
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|value| u32::try_from(value.get()).unwrap_or(u32::MAX))
            .unwrap_or(1);
        Self {
            device: Device::Cpu,
            gpu_layers: GpuLayersRequested::Count(0),
            threads,
            n_ctx: None,
            max_tokens: 4096,
            max_context_tokens: 32_768,
            n_batch: 512,
            n_ubatch: 512,
            max_sequences: 32,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeDevice {
    pub name: String,
    pub description: String,
    pub backend: String,
    pub device_type: String,
    pub memory_total: usize,
    pub memory_free: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplateStatus {
    Exact,
    ReviewedEquivalent,
    OverrideUnverified,
    Mismatch,
    Missing,
}

#[derive(Clone, Debug, Serialize)]
pub struct LoadedModelInfo {
    pub id: String,
    pub artifact_sha256: String,
    pub architecture: String,
    pub n_ctx_train: u32,
    pub vocabulary_size: u32,
    pub model_bytes: u64,
    pub profile: PromptProfile,
    pub gguf_template_sha256: Option<String>,
    pub expected_native_template_sha256: Option<String>,
    pub template_status: TemplateStatus,
    pub template_diagnostic: Option<String>,
    pub load_seconds: f64,
    pub backend: String,
    pub device_requested: Device,
    pub gpu_layers_requested: GpuLayersRequested,
    pub gpu_layers_actual: Option<u32>,
    pub gpu_layers_status: GpuLayersStatus,
    pub runtime_devices: Vec<RuntimeDevice>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EncodedPrompt {
    pub id: String,
    pub prompt_sha256: String,
    pub prompt_version: String,
    pub option_ids: Vec<String>,
    pub prompt_token_ids: Vec<u32>,
    pub answer_token_ids: Vec<u32>,
}

impl EncodedPrompt {
    /// Fail closed on any strict reference-field mismatch.
    pub fn validate_reference(
        &self,
        expected_id: &str,
        expected_option_ids: &[String],
        expected_prompt_sha256: &str,
        expected_input_tokens: usize,
        expected_answer_token_ids: &[u32],
    ) -> Result<()> {
        let mismatch = if self.id != expected_id {
            Some(format!("id {:?} != {expected_id:?}", self.id))
        } else if self.option_ids != expected_option_ids {
            Some("option_ids differ".to_owned())
        } else if self.prompt_sha256 != expected_prompt_sha256 {
            Some(format!(
                "prompt_sha256 {} != {expected_prompt_sha256}",
                self.prompt_sha256
            ))
        } else if self.prompt_token_ids.len() != expected_input_tokens {
            Some(format!(
                "input_tokens {} != {expected_input_tokens}",
                self.prompt_token_ids.len()
            ))
        } else if self.answer_token_ids != expected_answer_token_ids {
            Some("answer_token_ids differ".to_owned())
        } else {
            None
        };
        if let Some(message) = mismatch {
            Err(BackendError::Core(format!(
                "strict parity mismatch for {expected_id:?}: {message}"
            )))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DirectSmokeReport {
    pub schema: &'static str,
    pub model: LoadedModelInfo,
    pub id: String,
    pub prompt_sha256: String,
    pub input_tokens: usize,
    pub answer_token_ids: Vec<u32>,
    pub option_logits: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub choice_index: usize,
    pub allowed_token_mass: f64,
    pub full_vocab_argmax_id: u32,
    pub full_vocab_log_normalizer: f64,
    pub warmup_forward_seconds: f64,
    pub forward_seconds: f64,
    pub n_ctx_actual: u32,
    pub n_batch_actual: u32,
    pub n_ubatch_actual: u32,
    pub finite_readout: bool,
}

struct DirectPass {
    numeric: NumericReadout,
    forward_seconds: f64,
    n_ctx_actual: u32,
    n_batch_actual: u32,
    n_ubatch_actual: u32,
}

enum Request {
    Encode {
        decision: Box<Decision>,
        response: SyncSender<Result<EncodedPrompt>>,
    },
    Score {
        decision: Box<Decision>,
        response: SyncSender<Result<Readout>>,
    },
    ScoreShared {
        state: Box<StateValue>,
        questions: Vec<Question>,
        probe_id: String,
        response: SyncSender<Result<Vec<Readout>>>,
    },
    ScoreBatch {
        decisions: Vec<Decision>,
        probe_id: String,
        response: SyncSender<Result<Vec<Readout>>>,
    },
    Smoke {
        decision: Box<Decision>,
        response: SyncSender<Result<DirectSmokeReport>>,
    },
    Shutdown,
}

pub struct EngineHandle {
    sender: SyncSender<Request>,
    join: Option<JoinHandle<()>>,
    info: LoadedModelInfo,
}

impl EngineHandle {
    pub fn spawn(
        model: ModelEntry,
        artifact: VerifiedArtifact,
        options: EngineOptions,
    ) -> Result<Self> {
        Self::spawn_resolved(model.into(), artifact, options)
    }

    pub fn spawn_resolved(
        model: RuntimeModelSpec,
        artifact: VerifiedArtifact,
        options: EngineOptions,
    ) -> Result<Self> {
        options.validate()?;
        model.validate()?;
        if artifact.model_id != model.id
            || artifact.bytes != model.bytes
            || artifact.sha256 != model.sha256
        {
            return Err(BackendError::Configuration(
                "resolved artifact identity does not match the engine model specification"
                    .to_owned(),
            ));
        }
        let (sender, receiver) = sync_channel(1);
        let (startup_sender, startup_receiver) = sync_channel(1);
        let join = std::thread::Builder::new()
            .name(format!("openjev-{}-owner", model.id))
            .spawn(move || worker_main(model, artifact, options, receiver, startup_sender))
            .map_err(BackendError::Io)?;
        let info = match startup_receiver.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(error)) => {
                let _ = join.join();
                return Err(error);
            }
            Err(_) => {
                let _ = join.join();
                return Err(BackendError::Worker(
                    "owner thread exited before startup completed".to_owned(),
                ));
            }
        };
        Ok(Self {
            sender,
            join: Some(join),
            info,
        })
    }

    #[must_use]
    pub fn model_info(&self) -> &LoadedModelInfo {
        &self.info
    }

    /// Render and verify the complete no-BOS prompt and every declared answer slot.
    ///
    /// This performs no decode and is exposed for strict integration gates. Production
    /// scoring performs the same checks again as part of its one-prefill decision path.
    pub fn encode_direct(&self, decision: Decision) -> Result<EncodedPrompt> {
        let (sender, receiver) = sync_channel(1);
        self.sender
            .send(Request::Encode {
                decision: Box::new(decision),
                response: sender,
            })
            .map_err(|_| BackendError::Worker("owner thread is unavailable".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| BackendError::Worker("owner thread dropped its response".to_owned()))?
    }

    /// Score one validated decision with exactly one prompt prefill (possibly chunked).
    pub fn score_direct(&self, decision: Decision) -> Result<Readout> {
        let (sender, receiver) = sync_channel(1);
        self.sender
            .send(Request::Score {
                decision: Box::new(decision),
                response: sender,
            })
            .map_err(|_| BackendError::Worker("owner thread is unavailable".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| BackendError::Worker("owner thread dropped its response".to_owned()))?
    }

    /// Score questions against one immutable state prefix. The complete local
    /// receipt must be passing; the owner thread also verifies its ID against
    /// this engine's exact runtime configuration.
    pub fn score_shared(
        &self,
        state: StateValue,
        questions: Vec<Question>,
        eligibility: ProbeEligibility,
    ) -> Result<Vec<Readout>> {
        if eligibility.mode() != ProbeMode::Shared {
            return Err(BackendError::Configuration(
                "shared execution requires a passing shared probe receipt".to_owned(),
            ));
        }
        self.send_shared(state, questions, eligibility.probe_id().to_owned())
    }

    /// Execute one untrusted shared candidate only inside the isolated probe
    /// child. This cannot authorize standard production scoring.
    pub fn probe_shared_candidate(
        &self,
        state: StateValue,
        questions: Vec<Question>,
        expected_probe_id: String,
    ) -> Result<Vec<Readout>> {
        require_probe_child()?;
        self.send_shared(state, questions, expected_probe_id)
    }

    fn send_shared(
        &self,
        state: StateValue,
        questions: Vec<Question>,
        probe_id: String,
    ) -> Result<Vec<Readout>> {
        let (sender, receiver) = sync_channel(1);
        self.sender
            .send(Request::ScoreShared {
                state: Box::new(state),
                questions,
                probe_id,
                response: sender,
            })
            .map_err(|_| BackendError::Worker("owner thread is unavailable".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| BackendError::Worker("owner thread dropped its response".to_owned()))?
    }

    /// Score unrelated full prompts in distinct native sequence IDs without
    /// prefix sharing. The complete local receipt must be passing and exact.
    pub fn score_batch(
        &self,
        decisions: Vec<Decision>,
        eligibility: ProbeEligibility,
    ) -> Result<Vec<Readout>> {
        if eligibility.mode() != ProbeMode::Batch {
            return Err(BackendError::Configuration(
                "batch execution requires a passing batch probe receipt".to_owned(),
            ));
        }
        self.send_batch(decisions, eligibility.probe_id().to_owned())
    }

    /// Execute one untrusted packed-batch candidate only inside the isolated
    /// probe child. This cannot authorize standard production scoring.
    pub fn probe_batch_candidate(
        &self,
        decisions: Vec<Decision>,
        expected_probe_id: String,
    ) -> Result<Vec<Readout>> {
        require_probe_child()?;
        self.send_batch(decisions, expected_probe_id)
    }

    fn send_batch(&self, decisions: Vec<Decision>, probe_id: String) -> Result<Vec<Readout>> {
        let (sender, receiver) = sync_channel(1);
        self.sender
            .send(Request::ScoreBatch {
                decisions,
                probe_id,
                response: sender,
            })
            .map_err(|_| BackendError::Worker("owner thread is unavailable".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| BackendError::Worker("owner thread dropped its response".to_owned()))?
    }

    /// M2 diagnostic path: one unreported warmup pass followed by one measured pass.
    pub fn smoke_direct(&self, decision: Decision) -> Result<DirectSmokeReport> {
        let (sender, receiver) = sync_channel(1);
        self.sender
            .send(Request::Smoke {
                decision: Box::new(decision),
                response: sender,
            })
            .map_err(|_| BackendError::Worker("owner thread is unavailable".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| BackendError::Worker("owner thread dropped its response".to_owned()))?
    }

    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.sender.send(Request::Shutdown);
        self.join_worker()
    }

    fn join_worker(&mut self) -> Result<()> {
        if let Some(join) = self.join.take() {
            join_owner_thread(join)?;
        }
        Ok(())
    }
}

fn join_owner_thread(join: JoinHandle<()>) -> Result<()> {
    join.join()
        .map_err(|_| BackendError::Worker("owner thread panicked".to_owned()))
}

fn require_probe_child() -> Result<()> {
    if std::env::var("OPENJEV_PROBE_CHILD").as_deref() == Ok("1") {
        Ok(())
    } else {
        Err(BackendError::Configuration(
            "unverified native candidates are restricted to an isolated probe child".to_owned(),
        ))
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        let _ = self.sender.send(Request::Shutdown);
        let _ = self.join_worker();
    }
}

fn worker_main(
    spec: RuntimeModelSpec,
    artifact: VerifiedArtifact,
    options: EngineOptions,
    receiver: Receiver<Request>,
    startup: SyncSender<Result<LoadedModelInfo>>,
) {
    let initialization = initialize_worker(&spec, &artifact, &options);
    let (backend, model, info) = match initialization {
        Ok(values) => values,
        Err(error) => {
            let _ = startup.send(Err(error));
            return;
        }
    };
    if startup.send(Ok(info.clone())).is_err() {
        return;
    }

    while let Ok(request) = receiver.recv() {
        match request {
            Request::Encode { decision, response } => {
                let result = encode_direct_prompt(&model, &spec, &info, &decision);
                let _ = response.send(result);
            }
            Request::Score { decision, response } => {
                let result = score_direct(&backend, &model, &spec, &options, &info, &decision);
                let _ = response.send(result);
            }
            Request::ScoreShared {
                state,
                questions,
                probe_id,
                response,
            } => {
                let result = score_shared_native(
                    &backend, &model, &spec, &options, &info, &state, &questions, &probe_id,
                );
                let _ = response.send(result);
            }
            Request::ScoreBatch {
                decisions,
                probe_id,
                response,
            } => {
                let result = score_batch_native(
                    &backend, &model, &spec, &options, &info, &decisions, &probe_id,
                );
                let _ = response.send(result);
            }
            Request::Smoke { decision, response } => {
                let result = direct_smoke(&backend, &model, &spec, &options, &info, &decision);
                let _ = response.send(result);
            }
            Request::Shutdown => break,
        }
    }
    drop(model);
    drop(backend);
}

fn initialize_worker(
    spec: &RuntimeModelSpec,
    artifact: &VerifiedArtifact,
    options: &EngineOptions,
) -> Result<(LlamaBackend, LlamaModel, LoadedModelInfo)> {
    let (actual_bytes, actual_sha256) = hash_file(&artifact.path)?;
    if actual_bytes != spec.bytes || actual_sha256 != spec.sha256 {
        return Err(BackendError::Integrity {
            path: artifact.path.clone(),
            expected_bytes: spec.bytes,
            actual_bytes,
            expected_sha256: spec.sha256.clone(),
            actual_sha256,
        });
    }

    send_logs_to_tracing(LogOptions::default());
    let backend = LlamaBackend::init().map_err(|error| BackendError::Native(error.to_string()))?;
    if options.device != Device::Cpu && !backend.supports_gpu_offload() {
        return Err(BackendError::Native(format!(
            "requested {:?}, but this native build/runtime reports no GPU offload support",
            options.device
        )));
    }

    let model_params = match options.device {
        Device::Cpu => LlamaModelParams::default().with_n_gpu_layers(0),
        Device::Metal | Device::Cuda => match options.gpu_layers {
            GpuLayersRequested::All => LlamaModelParams::default().with_n_gpu_layers(u32::MAX),
            GpuLayersRequested::Count(count) => {
                LlamaModelParams::default().with_n_gpu_layers(count)
            }
        },
    };
    let load_started = Instant::now();
    let model = LlamaModel::load_from_file(&backend, &artifact.path, &model_params)
        .map_err(|error| BackendError::ModelLoad(error.to_string()))?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let architecture = model
        .meta_val_str("general.architecture")
        .map_err(|error| BackendError::Metadata(format!("general.architecture: {error}")))?;
    let gguf_template = model.meta_val_str("tokenizer.chat_template").ok();
    let gguf_template_sha256 = gguf_template
        .as_ref()
        .map(|template| digest_hex(&Sha256::digest(template.as_bytes())));
    let expected = spec
        .native_reference
        .as_ref()
        .map(|reference| reference.template_sha256.clone());
    let (template_status, template_diagnostic) =
        adjudicate_template(spec, &actual_sha256, gguf_template_sha256.as_deref());
    let vocabulary_size = u32::try_from(model.n_vocab()).map_err(|_| {
        BackendError::Metadata("model vocabulary size is negative or exceeds u32".to_owned())
    })?;
    let runtime_devices = list_llama_ggml_backend_devices()
        .into_iter()
        .map(|device| RuntimeDevice {
            name: device.name,
            description: device.description,
            backend: device.backend,
            device_type: format!("{:?}", device.device_type),
            memory_total: device.memory_total,
            memory_free: device.memory_free,
        })
        .collect();
    let (gpu_layers_actual, gpu_layers_status) = if options.device == Device::Cpu {
        (Some(0), GpuLayersStatus::KnownDisabled)
    } else {
        (None, GpuLayersStatus::Unavailable)
    };
    let info = LoadedModelInfo {
        id: spec.id.clone(),
        artifact_sha256: actual_sha256,
        architecture,
        n_ctx_train: model.n_ctx_train(),
        vocabulary_size,
        model_bytes: model.size(),
        profile: spec.profile,
        gguf_template_sha256,
        expected_native_template_sha256: expected,
        template_status,
        template_diagnostic,
        load_seconds,
        backend: format!("llama-cpp-2/0.1.156 llama.cpp/{LLAMA_CPP_COMMIT}"),
        device_requested: options.device,
        gpu_layers_requested: options.gpu_layers,
        gpu_layers_actual,
        gpu_layers_status,
        runtime_devices,
    };
    tracing::info!(
        model = %info.id,
        architecture = %info.architecture,
        n_ctx_train = info.n_ctx_train,
        template_sha256 = ?info.gguf_template_sha256,
        template_status = ?info.template_status,
        load_seconds = info.load_seconds,
        "loaded pinned GGUF"
    );
    Ok((backend, model, info))
}

fn adjudicate_template(
    spec: &RuntimeModelSpec,
    artifact_sha256: &str,
    gguf_template_sha256: Option<&str>,
) -> (TemplateStatus, Option<String>) {
    if spec.template_override {
        return (
            TemplateStatus::OverrideUnverified,
            Some(
                "explicit template-profile override for a custom artifact; no registered golden or native-template equivalence is claimed"
                    .to_owned(),
            ),
        );
    }
    let Some(expected) = spec
        .native_reference
        .as_ref()
        .map(|reference| &reference.template_sha256)
    else {
        return (
            TemplateStatus::Mismatch,
            Some("registered model is missing its native template identity".to_owned()),
        );
    };
    match gguf_template_sha256 {
        Some(actual) if actual == expected => (TemplateStatus::Exact, None),
        Some(actual)
            if spec.template_equivalence.as_ref().is_some_and(|record| {
                record.artifact_sha256 == artifact_sha256
                    && record.gguf_template_sha256 == actual
                    && record.native_profile_sha256 == *expected
            }) => {
                let record = spec
                    .template_equivalence
                    .as_ref()
                    .expect("matching equivalence record exists");
                (
                    TemplateStatus::ReviewedEquivalent,
                    Some(format!(
                        "GGUF template {actual} is not identical to native template {expected}; reviewed equivalent only for {}. Evidence: {}. Tool, multimodal, assistant-reasoning, and arbitrary multi-turn behavior is outside this approval",
                        record.scope, record.evidence
                    )),
                )
            }
        Some(actual) => (
            TemplateStatus::Mismatch,
            Some(format!(
                "GGUF tokenizer.chat_template hash {actual} differs from pinned native template hash {expected} and has no matching artifact-keyed reviewed equivalence record"
            )),
        ),
        None => (
            TemplateStatus::Missing,
            Some(
                "GGUF has no readable tokenizer.chat_template metadata; profile rendering requires adjudication"
                    .to_owned(),
            ),
        ),
    }
}

fn encode_direct_prompt(
    model: &LlamaModel,
    spec: &RuntimeModelSpec,
    info: &LoadedModelInfo,
    decision: &Decision,
) -> Result<EncodedPrompt> {
    // Decision/prompt validation deliberately precedes the production template
    // eligibility check so malformed per-decision input never reaches decode.
    let prompt = prepare_prompt(decision, spec.profile)
        .map_err(|error| BackendError::Core(error.to_string()))?;
    ensure_production_template(info)?;
    let tokenizer = LlamaTokenizer { model };
    let slots = verify_slots(&tokenizer, &prompt.text, decision.options.len())
        .map_err(|error| BackendError::Core(error.to_string()))?;
    Ok(EncodedPrompt {
        id: decision.id.clone(),
        prompt_sha256: prompt.prompt_sha256,
        prompt_version: prompt.prompt_version,
        option_ids: decision
            .options
            .iter()
            .map(|option| option.id.clone())
            .collect(),
        prompt_token_ids: slots.prompt_token_ids,
        answer_token_ids: slots.answer_token_ids,
    })
}

fn ensure_production_template(info: &LoadedModelInfo) -> Result<()> {
    match info.template_status {
        TemplateStatus::Exact
        | TemplateStatus::ReviewedEquivalent
        | TemplateStatus::OverrideUnverified => Ok(()),
        TemplateStatus::Mismatch | TemplateStatus::Missing => Err(BackendError::Metadata(
            info.template_diagnostic.clone().unwrap_or_else(|| {
                "registered artifact has no production-approved prompt template".to_owned()
            }),
        )),
    }
}

fn score_direct(
    backend: &LlamaBackend,
    model: &LlamaModel,
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    info: &LoadedModelInfo,
    decision: &Decision,
) -> Result<Readout> {
    let total_started = Instant::now();
    let encoded = encode_direct_prompt(model, spec, info, decision)?;
    let pass = run_direct_pass(
        backend,
        model,
        options,
        &encoded.prompt_token_ids,
        &encoded.answer_token_ids,
    )?;
    let choice_index = pass.numeric.choice_index;
    let template_status = match info.template_status {
        TemplateStatus::Exact => TemplateMetadataStatus::Exact,
        TemplateStatus::ReviewedEquivalent => TemplateMetadataStatus::ReviewedEquivalent,
        TemplateStatus::OverrideUnverified => TemplateMetadataStatus::OverrideUnverified,
        TemplateStatus::Mismatch | TemplateStatus::Missing => {
            return Err(BackendError::Metadata(
                "unapproved template reached production readout".to_owned(),
            ));
        }
    };
    let template_equivalence_evidence = spec.template_equivalence.as_ref().and_then(|record| {
        (template_status == TemplateMetadataStatus::ReviewedEquivalent).then(|| {
            format!(
                "{}; scope={}; artifact_sha256={}; gguf_template_sha256={}; native_profile_sha256={}",
                record.evidence,
                record.scope,
                record.artifact_sha256,
                record.gguf_template_sha256,
                record.native_profile_sha256
            )
        })
    });
    let run_id = format!(
        "openjev-{}-{}",
        std::process::id(),
        RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let mut readout = Readout {
        schema: "openjev-readout-v1".to_owned(),
        id: encoded.id,
        primitive: Primitive::Choice,
        choice: encoded.option_ids[choice_index].clone(),
        choice_index,
        option_ids: encoded.option_ids,
        probabilities: pass.numeric.probabilities,
        option_logits: pass.numeric.option_logits,
        answer_token_ids: encoded.answer_token_ids,
        allowed_token_mass: pass.numeric.allowed_token_mass,
        full_vocab_argmax_id: pass.numeric.full_vocab_argmax_id,
        full_vocab_log_normalizer: pass.numeric.full_vocab_log_normalizer,
        input_tokens: u64::try_from(encoded.prompt_token_ids.len())
            .map_err(|_| BackendError::Context("prompt token count exceeds u64".to_owned()))?,
        forward_seconds: Some(pass.forward_seconds),
        total_seconds: None,
        prompt_sha256: encoded.prompt_sha256,
        prompt_version: encoded.prompt_version,
        model: ModelMetadata {
            id: spec.id.clone(),
            source: spec.source.clone(),
            revision: spec.revision.clone(),
            file: spec.file.clone(),
            quant: spec.quant.clone(),
            backend: info.backend.clone(),
            artifact_sha256: info.artifact_sha256.clone(),
            integrity: spec.integrity,
            dtype: format!("GGUF quantized/mixed {}", spec.quant),
            native_reference: spec
                .native_reference
                .as_ref()
                .map(|reference| NativeReference {
                    source: reference.source.clone(),
                    revision: reference.revision.clone(),
                    dtype: reference.dtype.clone(),
                }),
            template_profile: spec.profile,
            template_sha256: info.gguf_template_sha256.clone(),
            template_override: spec.template_override,
            template_status,
            template_equivalence_evidence,
            serving_config: Some("llama-direct-v1".to_owned()),
            adapter: None,
            adapter_sha256: None,
            adapter_revision: None,
            torch_version: None,
            transformers_version: None,
        },
        readout: DIRECT_READOUT.to_owned(),
        probability_status: PROBABILITY_STATUS.to_owned(),
        limitations: standard_limitations(),
        execution: ExecutionMetadata {
            requested_mode: ExecutionMode::Direct,
            effective_mode: ExecutionMode::Direct,
            fallback_reason: None,
            device: options.device,
            device_name: selected_device_name(info),
            gpu_layers_requested: options.gpu_layers,
            gpu_layers_actual: info.gpu_layers_actual,
            gpu_layers_status: info.gpu_layers_status,
            threads: options.threads,
            n_ctx_requested: options.n_ctx,
            n_ctx_actual: pass.n_ctx_actual,
            max_tokens: options.max_tokens,
            n_batch: pass.n_batch_actual,
            n_ubatch: pass.n_ubatch_actual,
            n_seq_max: 1,
            kv_unified: true,
            waves: 1,
            probe_id: None,
            run_id,
            group_id: None,
        },
        confidence: None,
        confidence_status: None,
        p_yes: None,
        level_values: None,
        expected_value: None,
        argmax_level: None,
        cache_hit: crate::direct_inference_cache_hit(),
        prefix_tokens: None,
        prefix_sha256: None,
        prefill_seconds: None,
        copy_seconds: None,
        suffix_forward_seconds: None,
        shared_timing: None,
        postprocess: None,
    };
    readout.total_seconds = Some(total_started.elapsed().as_secs_f64());
    readout
        .validate()
        .map_err(|error| BackendError::Core(error.to_string()))?;
    Ok(readout)
}

fn selected_device_name(info: &LoadedModelInfo) -> String {
    let selected = match info.device_requested {
        Device::Cpu => info
            .runtime_devices
            .iter()
            .find(|device| device.device_type == "Cpu"),
        Device::Metal => info
            .runtime_devices
            .iter()
            .find(|device| device.backend == "MTL"),
        Device::Cuda => info
            .runtime_devices
            .iter()
            .find(|device| device.backend.to_ascii_uppercase().contains("CUDA")),
    };
    selected.map_or_else(
        || {
            format!(
                "{:?} (runtime device name unavailable)",
                info.device_requested
            )
        },
        |device| {
            format!(
                "{}: {} ({})",
                device.name, device.description, device.backend
            )
        },
    )
}

struct GroupEncoding {
    decision: Decision,
    encoded: EncodedPrompt,
}

struct GroupExecution {
    numeric: Vec<NumericReadout>,
    n_ctx_actual: u32,
    n_batch_actual: u32,
    n_ubatch_actual: u32,
    waves: u32,
    prefill_seconds: f64,
    replicate_seconds: f64,
    suffix_seconds: f64,
}

fn require_exact_probe_id(
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    mode: ProbeMode,
    supplied: &str,
) -> Result<()> {
    let configuration = ProbeConfiguration {
        artifact_sha256: spec.sha256.clone(),
        native_pin: NATIVE_PIN.to_owned(),
        probe_suite_version: PROBE_SUITE_VERSION.to_owned(),
        device: options.device,
        gpu_layers_requested: options.gpu_layers,
        offload_kqv: options.device != Device::Cpu,
        op_offload: options.device != Device::Cpu,
        threads: options.threads,
        n_ctx: options.n_ctx,
        max_tokens: options.max_tokens,
        max_context_tokens: options.max_context_tokens,
        n_batch: options.n_batch,
        n_ubatch: options.n_ubatch,
        n_seq_max: options.max_sequences,
        kv_unified: true,
        profile: spec.profile,
    };
    let expected = probe_id(mode, &configuration)?;
    if supplied != expected {
        return Err(BackendError::Configuration(format!(
            "supplied probe ID does not authorize this exact {} configuration",
            mode.as_str()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_shared_native(
    backend: &LlamaBackend,
    model: &LlamaModel,
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    info: &LoadedModelInfo,
    state: &StateValue,
    questions: &[Question],
    probe_id: &str,
) -> Result<Vec<Readout>> {
    require_exact_probe_id(spec, options, ProbeMode::Shared, probe_id)?;
    let total_started = Instant::now();
    if questions.is_empty() {
        return Err(BackendError::Configuration(
            "shared scoring requires at least one question".to_owned(),
        ));
    }
    if options.max_sequences < 2 {
        return Err(BackendError::Configuration(
            "shared scoring requires max_sequences >= 2 for prefix plus branch".to_owned(),
        ));
    }
    let mut ids = HashSet::with_capacity(questions.len());
    let encode_started = Instant::now();
    let mut rows = Vec::with_capacity(questions.len());
    for question in questions {
        if !ids.insert(question.id.as_str()) {
            return Err(BackendError::Configuration(format!(
                "duplicate shared question ID {:?}",
                question.id
            )));
        }
        let decision = Decision::new(
            question.id.clone(),
            state.clone(),
            question.question.clone(),
            question.options.clone(),
        )
        .map_err(|error| BackendError::Core(error.to_string()))?;
        let encoded = encode_direct_prompt(model, spec, info, &decision)?;
        rows.push(GroupEncoding { decision, encoded });
    }
    let tokenizer = LlamaTokenizer { model };
    let prefix = encode_state_prefix(&tokenizer, state, spec.profile)?;
    let suffixes: Vec<Vec<u32>> = rows
        .iter()
        .map(|row| {
            if !row.encoded.prompt_token_ids.starts_with(&prefix)
                || row.encoded.prompt_token_ids.len() <= prefix.len()
            {
                return Err(BackendError::Core(format!(
                    "full prompt for {:?} does not start with the exact nonempty state prefix",
                    row.decision.id
                )));
            }
            Ok(row.encoded.prompt_token_ids[prefix.len()..].to_vec())
        })
        .collect::<Result<_>>()?;
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    let execution = run_shared_group(backend, model, options, &prefix, &suffixes, &rows)?;
    let total_seconds = total_started.elapsed().as_secs_f64();
    let prefix_sha256 = prefix_token_hash(&prefix);
    let true_suffix_tokens =
        suffixes.iter().try_fold(0_u64, |total, suffix| {
            total
                .checked_add(u64::try_from(suffix.len()).map_err(|_| {
                    BackendError::Context("suffix token count exceeds u64".to_owned())
                })?)
                .ok_or_else(|| BackendError::Context("suffix token total overflow".to_owned()))
        })?;
    let timing = SharedTiming {
        total_seconds,
        encode_seconds,
        prefix_tokens: u64::try_from(prefix.len())
            .map_err(|_| BackendError::Context("prefix token count exceeds u64".to_owned()))?,
        prefill_seconds: execution.prefill_seconds,
        replicate_seconds: execution.replicate_seconds,
        suffix_forward_seconds: execution.suffix_seconds,
        batch_size: u64::try_from(rows.len())
            .map_err(|_| BackendError::Context("shared batch size exceeds u64".to_owned()))?,
        true_suffix_tokens,
        padded_suffix_tokens: true_suffix_tokens,
    };
    rows.into_iter()
        .zip(execution.numeric)
        .map(|(row, numeric)| {
            build_group_readout(
                spec,
                options,
                info,
                row,
                numeric,
                ExecutionMode::Shared,
                probe_id,
                execution.n_ctx_actual,
                execution.n_batch_actual,
                execution.n_ubatch_actual,
                execution.waves,
                Some((&prefix, &prefix_sha256, &timing)),
            )
        })
        .collect()
}

fn score_batch_native(
    backend: &LlamaBackend,
    model: &LlamaModel,
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    info: &LoadedModelInfo,
    decisions: &[Decision],
    probe_id: &str,
) -> Result<Vec<Readout>> {
    require_exact_probe_id(spec, options, ProbeMode::Batch, probe_id)?;
    if decisions.is_empty() {
        return Err(BackendError::Configuration(
            "batch scoring requires at least one decision".to_owned(),
        ));
    }
    let mut ids = HashSet::with_capacity(decisions.len());
    let rows: Vec<_> = decisions
        .iter()
        .map(|decision| {
            if !ids.insert(decision.id.as_str()) {
                return Err(BackendError::Configuration(format!(
                    "duplicate batch decision ID {:?}",
                    decision.id
                )));
            }
            Ok(GroupEncoding {
                decision: decision.clone(),
                encoded: encode_direct_prompt(model, spec, info, decision)?,
            })
        })
        .collect::<Result<_>>()?;
    let sequences: Vec<Vec<u32>> = rows
        .iter()
        .map(|row| row.encoded.prompt_token_ids.clone())
        .collect();
    let execution = run_independent_batch(backend, model, options, &sequences, &rows)?;
    rows.into_iter()
        .zip(execution.numeric)
        .map(|(row, numeric)| {
            build_group_readout(
                spec,
                options,
                info,
                row,
                numeric,
                ExecutionMode::Batch,
                probe_id,
                execution.n_ctx_actual,
                execution.n_batch_actual,
                execution.n_ubatch_actual,
                execution.waves,
                None,
            )
        })
        .collect()
}

fn run_shared_group(
    backend: &LlamaBackend,
    model: &LlamaModel,
    options: &EngineOptions,
    prefix: &[u32],
    suffixes: &[Vec<u32>],
    rows: &[GroupEncoding],
) -> Result<GroupExecution> {
    validate_group_lengths(model, options, rows)?;
    let cap = context_cap(model, options)?;
    let branch_limit = options
        .max_sequences
        .saturating_sub(1)
        .min(options.n_batch)
        .max(1) as usize;
    let waves = plan_waves(prefix.len(), suffixes, cap, branch_limit)?;
    let required = waves
        .iter()
        .map(|wave| {
            prefix.len()
                + wave
                    .iter()
                    .map(|index| suffixes[*index].len())
                    .sum::<usize>()
        })
        .max()
        .unwrap_or(prefix.len());
    let n_ctx = choose_context(model, options, required)?;
    let (mut context, n_batch, n_ubatch) =
        new_group_context(backend, model, options, n_ctx, options.max_sequences)?;
    let n_ctx_actual = context.n_ctx();
    let mut batch = LlamaBatch::new(n_batch as usize, 1);
    let prefill_started = Instant::now();
    decode_single_sequence(&mut context, &mut batch, prefix, 0, 0, n_batch, false)?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut numeric: Vec<Option<NumericReadout>> = std::iter::repeat_with(|| None)
        .take(suffixes.len())
        .collect();
    let mut replicate_seconds = 0.0;
    let mut suffix_seconds = 0.0;
    for wave in &waves {
        let copy_started = Instant::now();
        for (slot, _) in wave.iter().enumerate() {
            context
                .copy_kv_cache_seq(
                    0,
                    i32::try_from(slot + 1).map_err(|_| {
                        BackendError::Context("branch sequence ID exceeds i32".to_owned())
                    })?,
                    None,
                    None,
                )
                .map_err(|error| BackendError::Decode(error.to_string()))?;
        }
        replicate_seconds += copy_started.elapsed().as_secs_f64();
        let work: Vec<_> = wave
            .iter()
            .enumerate()
            .map(|(slot, index)| SequenceWork {
                output_index: *index,
                seq_id: i32::try_from(slot + 1).expect("bounded max_sequences fits i32"),
                base_position: prefix.len(),
                tokens: &suffixes[*index],
                answer_token_ids: &rows[*index].encoded.answer_token_ids,
            })
            .collect();
        let suffix_started = Instant::now();
        decode_ragged_sequences(&mut context, &mut batch, n_batch, &work, &mut numeric)?;
        suffix_seconds += suffix_started.elapsed().as_secs_f64();
    }
    let numeric = numeric
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                BackendError::Decode(format!("shared branch {index} produced no final logits"))
            })
        })
        .collect::<Result<_>>()?;
    Ok(GroupExecution {
        numeric,
        n_ctx_actual,
        n_batch_actual: n_batch,
        n_ubatch_actual: n_ubatch,
        waves: u32::try_from(waves.len())
            .map_err(|_| BackendError::Context("wave count exceeds u32".to_owned()))?,
        prefill_seconds,
        replicate_seconds,
        suffix_seconds,
    })
}

fn run_independent_batch(
    backend: &LlamaBackend,
    model: &LlamaModel,
    options: &EngineOptions,
    sequences: &[Vec<u32>],
    rows: &[GroupEncoding],
) -> Result<GroupExecution> {
    validate_group_lengths(model, options, rows)?;
    let cap = context_cap(model, options)?;
    let branch_limit = options.max_sequences.min(options.n_batch).max(1) as usize;
    let waves = plan_waves(0, sequences, cap, branch_limit)?;
    let required = waves
        .iter()
        .map(|wave| wave.iter().map(|index| sequences[*index].len()).sum())
        .max()
        .unwrap_or(1);
    let n_ctx = choose_context(model, options, required)?;
    let (mut context, n_batch, n_ubatch) =
        new_group_context(backend, model, options, n_ctx, options.max_sequences)?;
    let n_ctx_actual = context.n_ctx();
    let mut batch = LlamaBatch::new(n_batch as usize, 1);
    let mut numeric: Vec<Option<NumericReadout>> = std::iter::repeat_with(|| None)
        .take(sequences.len())
        .collect();
    let started = Instant::now();
    for wave in &waves {
        let work: Vec<_> = wave
            .iter()
            .enumerate()
            .map(|(slot, index)| SequenceWork {
                output_index: *index,
                seq_id: i32::try_from(slot).expect("bounded max_sequences fits i32"),
                base_position: 0,
                tokens: &sequences[*index],
                answer_token_ids: &rows[*index].encoded.answer_token_ids,
            })
            .collect();
        decode_ragged_sequences(&mut context, &mut batch, n_batch, &work, &mut numeric)?;
    }
    let suffix_seconds = started.elapsed().as_secs_f64();
    let numeric = numeric
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                BackendError::Decode(format!("batch sequence {index} produced no final logits"))
            })
        })
        .collect::<Result<_>>()?;
    Ok(GroupExecution {
        numeric,
        n_ctx_actual,
        n_batch_actual: n_batch,
        n_ubatch_actual: n_ubatch,
        waves: u32::try_from(waves.len())
            .map_err(|_| BackendError::Context("wave count exceeds u32".to_owned()))?,
        prefill_seconds: 0.0,
        replicate_seconds: 0.0,
        suffix_seconds,
    })
}

fn encode_state_prefix<T: SlotTokenizer>(
    tokenizer: &T,
    state: &StateValue,
    profile: PromptProfile,
) -> Result<Vec<u32>> {
    let prefix_text =
        state_prefix_text(state, profile).map_err(|error| BackendError::Core(error.to_string()))?;
    let mut prefix = tokenizer
        .tokenize_no_bos(&prefix_text)
        .map_err(|error| BackendError::Core(error.to_string()))?;
    if prefix.pop().is_none() || prefix.is_empty() {
        return Err(BackendError::Core(
            "shared state prefix must remain nonempty after dropping exactly one token".to_owned(),
        ));
    }
    Ok(prefix)
}

struct SequenceWork<'a> {
    output_index: usize,
    seq_id: i32,
    base_position: usize,
    tokens: &'a [u32],
    answer_token_ids: &'a [u32],
}

fn decode_ragged_sequences(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    n_batch: u32,
    work: &[SequenceWork<'_>],
    output: &mut [Option<NumericReadout>],
) -> Result<()> {
    let mut cursors = vec![0_usize; work.len()];
    let lengths: Vec<_> = work.iter().map(|item| item.tokens.len()).collect();
    while let Some((active, equal_take)) = ragged_step(&cursors, &lengths, n_batch as usize)? {
        batch.clear();
        let mut endings = Vec::new();
        for index in active {
            let item = &work[index];
            for local in 0..equal_take {
                let token_index = cursors[index] + local;
                let position = item
                    .base_position
                    .checked_add(token_index)
                    .ok_or_else(|| BackendError::Context("token position overflow".to_owned()))?;
                let is_final = token_index + 1 == item.tokens.len();
                let batch_offset = batch.n_tokens();
                batch
                    .add(
                        LlamaToken(i32::try_from(item.tokens[token_index]).map_err(|_| {
                            BackendError::Decode("token ID exceeds i32".to_owned())
                        })?),
                        i32::try_from(position).map_err(|_| {
                            BackendError::Context("token position exceeds i32".to_owned())
                        })?,
                        &[item.seq_id],
                        is_final,
                    )
                    .map_err(|error| BackendError::Decode(error.to_string()))?;
                if is_final {
                    endings.push((index, batch_offset));
                }
            }
            cursors[index] += equal_take;
        }
        context
            .decode(batch)
            .map_err(|error| BackendError::Decode(error.to_string()))?;
        for (index, local_offset) in endings {
            let numeric = {
                let vocabulary = context.get_logits_ith(local_offset);
                let selected = select_option_logits(work[index].answer_token_ids, vocabulary)?;
                read_logits(&selected, vocabulary)
                    .map_err(|error| BackendError::Core(error.to_string()))?
            };
            output[work[index].output_index] = Some(numeric);
            let seq_id = u32::try_from(work[index].seq_id)
                .map_err(|_| BackendError::Decode("negative sequence ID".to_owned()))?;
            let cleared = context
                .clear_kv_cache_seq(Some(seq_id), None, None)
                .map_err(|error| BackendError::Decode(error.to_string()))?;
            require_sequence_clear(work[index].seq_id, cleared)?;
        }
    }
    Ok(())
}

fn ragged_step(
    cursors: &[usize],
    lengths: &[usize],
    n_batch: usize,
) -> Result<Option<(Vec<usize>, usize)>> {
    if cursors.len() != lengths.len() || n_batch == 0 {
        return Err(BackendError::Configuration(
            "ragged scheduler requires aligned lengths and a positive batch capacity".to_owned(),
        ));
    }
    if cursors
        .iter()
        .zip(lengths)
        .any(|(cursor, length)| cursor > length)
    {
        return Err(BackendError::Decode(
            "ragged scheduler cursor exceeds its sequence length".to_owned(),
        ));
    }
    let active: Vec<_> = cursors
        .iter()
        .zip(lengths)
        .enumerate()
        .filter_map(|(index, (cursor, length))| (*cursor < *length).then_some(index))
        .collect();
    if active.is_empty() {
        return Ok(None);
    }
    if active.len() > n_batch {
        return Err(BackendError::Context(
            "active sequence count exceeds n_batch".to_owned(),
        ));
    }
    let equal_take = active
        .iter()
        .map(|index| lengths[*index] - cursors[*index])
        .min()
        .expect("active scheduler set is nonempty")
        .min(n_batch / active.len());
    if equal_take == 0 {
        return Err(BackendError::Decode(
            "ragged scheduler made no progress".to_owned(),
        ));
    }
    Ok(Some((active, equal_take)))
}

fn require_sequence_clear(seq_id: i32, cleared: bool) -> Result<()> {
    if cleared {
        Ok(())
    } else {
        Err(BackendError::Decode(format!(
            "native full clear returned false for sequence {seq_id}"
        )))
    }
}

fn decode_single_sequence(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    tokens: &[u32],
    seq_id: i32,
    base_position: usize,
    n_batch: u32,
    final_logits: bool,
) -> Result<Option<Vec<f32>>> {
    let mut result = None;
    for (chunk_index, chunk) in tokens.chunks(n_batch as usize).enumerate() {
        batch.clear();
        let start = chunk_index
            .checked_mul(n_batch as usize)
            .ok_or_else(|| BackendError::Context("token position overflow".to_owned()))?;
        for (local, token) in chunk.iter().copied().enumerate() {
            let absolute = start + local;
            let final_token = final_logits && absolute + 1 == tokens.len();
            batch
                .add(
                    LlamaToken(
                        i32::try_from(token)
                            .map_err(|_| BackendError::Decode("token ID exceeds i32".to_owned()))?,
                    ),
                    i32::try_from(base_position + absolute).map_err(|_| {
                        BackendError::Context("token position exceeds i32".to_owned())
                    })?,
                    &[seq_id],
                    final_token,
                )
                .map_err(|error| BackendError::Decode(error.to_string()))?;
        }
        context
            .decode(batch)
            .map_err(|error| BackendError::Decode(error.to_string()))?;
        if final_logits && start + chunk.len() == tokens.len() {
            result = Some(
                context
                    .get_logits_ith(i32::try_from(chunk.len() - 1).map_err(|_| {
                        BackendError::Decode("chunk-local index exceeds i32".to_owned())
                    })?)
                    .to_vec(),
            );
        }
    }
    Ok(result)
}

fn select_option_logits(answer_token_ids: &[u32], vocabulary: &[f32]) -> Result<Vec<f32>> {
    answer_token_ids
        .iter()
        .map(|token_id| {
            vocabulary.get(*token_id as usize).copied().ok_or_else(|| {
                BackendError::Decode(format!(
                    "answer token ID {token_id} is outside copied vocabulary logits"
                ))
            })
        })
        .collect()
}

fn validate_group_lengths(
    model: &LlamaModel,
    options: &EngineOptions,
    rows: &[GroupEncoding],
) -> Result<()> {
    for row in rows {
        let length = u32::try_from(row.encoded.prompt_token_ids.len())
            .map_err(|_| BackendError::Context("prompt length exceeds u32".to_owned()))?;
        if length == 0 || length > options.max_tokens {
            return Err(BackendError::Context(format!(
                "prompt {:?} has {length} tokens; configured max_tokens is {}",
                row.decision.id, options.max_tokens
            )));
        }
        if length > model.n_ctx_train() {
            return Err(BackendError::Context(format!(
                "prompt {:?} has {length} tokens; model training context is {}",
                row.decision.id,
                model.n_ctx_train()
            )));
        }
    }
    Ok(())
}

fn context_cap(model: &LlamaModel, options: &EngineOptions) -> Result<usize> {
    let cap = options.n_ctx.unwrap_or(options.max_context_tokens);
    if cap > options.max_context_tokens || cap > model.n_ctx_train() {
        return Err(BackendError::Context(format!(
            "context cap {cap} exceeds configured/model limit {}/{}",
            options.max_context_tokens,
            model.n_ctx_train()
        )));
    }
    usize::try_from(cap).map_err(|_| BackendError::Context("context cap overflow".to_owned()))
}

fn plan_waves(
    base_tokens: usize,
    sequences: &[Vec<u32>],
    cap: usize,
    branch_limit: usize,
) -> Result<Vec<Vec<usize>>> {
    if base_tokens > cap {
        return Err(BackendError::Context(format!(
            "shared prefix requires {base_tokens} cells; context cap is {cap}"
        )));
    }
    let mut waves = Vec::new();
    let mut current = Vec::new();
    let mut occupied = base_tokens;
    for (index, sequence) in sequences.iter().enumerate() {
        let required = base_tokens
            .checked_add(sequence.len())
            .ok_or_else(|| BackendError::Context("context occupancy overflow".to_owned()))?;
        if required > cap {
            return Err(BackendError::Context(format!(
                "sequence {index} requires {required} occupied cells; context cap is {cap}"
            )));
        }
        let next = occupied
            .checked_add(sequence.len())
            .ok_or_else(|| BackendError::Context("context occupancy overflow".to_owned()))?;
        if !current.is_empty() && (current.len() == branch_limit || next > cap) {
            waves.push(std::mem::take(&mut current));
            occupied = base_tokens;
        }
        occupied += sequence.len();
        current.push(index);
    }
    if !current.is_empty() {
        waves.push(current);
    }
    if waves.is_empty() {
        return Err(BackendError::Configuration(
            "group planner requires at least one sequence".to_owned(),
        ));
    }
    Ok(waves)
}

fn choose_context(model: &LlamaModel, options: &EngineOptions, required: usize) -> Result<u32> {
    let required = u32::try_from(required)
        .map_err(|_| BackendError::Context("required context exceeds u32".to_owned()))?;
    let selected = if let Some(explicit) = options.n_ctx {
        if required > explicit {
            return Err(BackendError::Context(format!(
                "required occupancy {required} exceeds explicit n_ctx {explicit}"
            )));
        }
        explicit
    } else {
        round_context(required.max(4096))?
    };
    if selected > options.max_context_tokens || selected > model.n_ctx_train() {
        return Err(BackendError::Context(format!(
            "required n_ctx {selected} exceeds configured/model limit {}/{}",
            options.max_context_tokens,
            model.n_ctx_train()
        )));
    }
    Ok(selected)
}

fn new_group_context<'a>(
    backend: &LlamaBackend,
    model: &'a LlamaModel,
    options: &EngineOptions,
    n_ctx: u32,
    n_seq_max: u32,
) -> Result<(llama_cpp_2::context::LlamaContext<'a>, u32, u32)> {
    let n_batch = options.n_batch.min(n_ctx).max(1);
    let n_ubatch = options.n_ubatch.min(n_batch).max(1);
    let threads = i32::try_from(options.threads)
        .map_err(|_| BackendError::Configuration("threads exceeds i32::MAX".to_owned()))?;
    let mut params = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch)
        .with_n_seq_max(n_seq_max)
        .with_kv_unified(true)
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    if options.device == Device::Cpu {
        params = params.with_offload_kqv(false).with_op_offload(false);
    }
    let context = model
        .new_context(backend, params)
        .map_err(|error| BackendError::Context(error.to_string()))?;
    let n_batch_actual = context.n_batch();
    let n_ubatch_actual = context.n_ubatch();
    Ok((context, n_batch_actual, n_ubatch_actual))
}

#[allow(clippy::too_many_arguments)]
fn build_group_readout(
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    info: &LoadedModelInfo,
    row: GroupEncoding,
    numeric: NumericReadout,
    mode: ExecutionMode,
    probe_id: &str,
    n_ctx_actual: u32,
    n_batch_actual: u32,
    n_ubatch_actual: u32,
    waves: u32,
    shared: Option<(&[u32], &str, &SharedTiming)>,
) -> Result<Readout> {
    let template_status = approved_template_status(info)?;
    let choice_index = numeric.choice_index;
    let run_id = format!(
        "openjev-{}-{}",
        std::process::id(),
        RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let (prefix_tokens, prefix_sha256, prefill_seconds, copy_seconds, suffix_seconds, timing) =
        shared.map_or(
            (None, None, None, None, None, None),
            |(prefix, hash, timing)| {
                (
                    Some(prefix.len() as u64),
                    Some(hash.to_owned()),
                    Some(timing.prefill_seconds),
                    Some(timing.replicate_seconds),
                    Some(timing.suffix_forward_seconds),
                    Some(timing.clone()),
                )
            },
        );
    let mut readout = Readout {
        schema: "openjev-readout-v1".to_owned(),
        id: row.encoded.id,
        primitive: Primitive::Choice,
        choice: row.encoded.option_ids[choice_index].clone(),
        choice_index,
        option_ids: row.encoded.option_ids,
        probabilities: numeric.probabilities,
        option_logits: numeric.option_logits,
        answer_token_ids: row.encoded.answer_token_ids,
        allowed_token_mass: numeric.allowed_token_mass,
        full_vocab_argmax_id: numeric.full_vocab_argmax_id,
        full_vocab_log_normalizer: numeric.full_vocab_log_normalizer,
        input_tokens: row.encoded.prompt_token_ids.len() as u64,
        forward_seconds: None,
        total_seconds: None,
        prompt_sha256: row.encoded.prompt_sha256,
        prompt_version: row.encoded.prompt_version,
        model: model_metadata(spec, info, template_status, mode),
        readout: if mode == ExecutionMode::Shared {
            "native selected suffix-position logits".to_owned()
        } else {
            DIRECT_READOUT.to_owned()
        },
        probability_status: PROBABILITY_STATUS.to_owned(),
        limitations: standard_limitations(),
        execution: ExecutionMetadata {
            requested_mode: mode,
            effective_mode: mode,
            fallback_reason: None,
            device: options.device,
            device_name: selected_device_name(info),
            gpu_layers_requested: options.gpu_layers,
            gpu_layers_actual: info.gpu_layers_actual,
            gpu_layers_status: info.gpu_layers_status,
            threads: options.threads,
            n_ctx_requested: options.n_ctx,
            n_ctx_actual,
            max_tokens: options.max_tokens,
            n_batch: n_batch_actual,
            n_ubatch: n_ubatch_actual,
            n_seq_max: options.max_sequences,
            kv_unified: true,
            waves,
            probe_id: Some(probe_id.to_owned()),
            run_id,
            group_id: None,
        },
        confidence: None,
        confidence_status: None,
        p_yes: None,
        level_values: None,
        expected_value: None,
        argmax_level: None,
        cache_hit: Some(mode == ExecutionMode::Shared),
        prefix_tokens,
        prefix_sha256,
        prefill_seconds,
        copy_seconds,
        suffix_forward_seconds: suffix_seconds,
        shared_timing: timing,
        postprocess: None,
    };
    if mode == ExecutionMode::Batch {
        readout.model.serving_config = Some("llama-independent-batch-v1".to_owned());
    }
    readout
        .validate()
        .map_err(|error| BackendError::Core(error.to_string()))?;
    Ok(readout)
}

fn approved_template_status(info: &LoadedModelInfo) -> Result<TemplateMetadataStatus> {
    match info.template_status {
        TemplateStatus::Exact => Ok(TemplateMetadataStatus::Exact),
        TemplateStatus::ReviewedEquivalent => Ok(TemplateMetadataStatus::ReviewedEquivalent),
        TemplateStatus::OverrideUnverified => Ok(TemplateMetadataStatus::OverrideUnverified),
        TemplateStatus::Mismatch | TemplateStatus::Missing => Err(BackendError::Metadata(
            "unapproved template reached production readout".to_owned(),
        )),
    }
}

fn model_metadata(
    spec: &RuntimeModelSpec,
    info: &LoadedModelInfo,
    template_status: TemplateMetadataStatus,
    mode: ExecutionMode,
) -> ModelMetadata {
    let evidence = spec.template_equivalence.as_ref().and_then(|record| {
        (template_status == TemplateMetadataStatus::ReviewedEquivalent).then(|| {
            format!(
                "{}; scope={}; artifact_sha256={}; gguf_template_sha256={}; native_profile_sha256={}",
                record.evidence,
                record.scope,
                record.artifact_sha256,
                record.gguf_template_sha256,
                record.native_profile_sha256
            )
        })
    });
    ModelMetadata {
        id: spec.id.clone(),
        source: spec.source.clone(),
        revision: spec.revision.clone(),
        file: spec.file.clone(),
        quant: spec.quant.clone(),
        backend: info.backend.clone(),
        artifact_sha256: info.artifact_sha256.clone(),
        integrity: spec.integrity,
        dtype: format!("GGUF quantized/mixed {}", spec.quant),
        native_reference: spec
            .native_reference
            .as_ref()
            .map(|reference| NativeReference {
                source: reference.source.clone(),
                revision: reference.revision.clone(),
                dtype: reference.dtype.clone(),
            }),
        template_profile: spec.profile,
        template_sha256: info.gguf_template_sha256.clone(),
        template_override: spec.template_override,
        template_status,
        template_equivalence_evidence: evidence,
        serving_config: Some(
            match mode {
                ExecutionMode::Shared => "llama-state-prefix-parallel-v1",
                ExecutionMode::Batch => "llama-independent-batch-v1",
                ExecutionMode::Direct => "llama-direct-v1",
                ExecutionMode::Serial => "llama-serial-full-prompt-v1",
            }
            .to_owned(),
        ),
        adapter: None,
        adapter_sha256: None,
        adapter_revision: None,
        torch_version: None,
        transformers_version: None,
    }
}

fn prefix_token_hash(tokens: &[u32]) -> String {
    let mut serialized = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            serialized.push_str(", ");
        }
        write!(serialized, "{token}").expect("writing to String cannot fail");
    }
    serialized.push(']');
    digest_hex(&Sha256::digest(serialized.as_bytes()))
}

fn direct_smoke(
    backend: &LlamaBackend,
    model: &LlamaModel,
    spec: &RuntimeModelSpec,
    options: &EngineOptions,
    info: &LoadedModelInfo,
    decision: &Decision,
) -> Result<DirectSmokeReport> {
    let prompt = prepare_prompt(decision, spec.profile)
        .map_err(|error| BackendError::Core(error.to_string()))?;
    let tokenizer = LlamaTokenizer { model };
    let slots = verify_slots(&tokenizer, &prompt.text, decision.options.len())
        .map_err(|error| BackendError::Core(error.to_string()))?;
    let warmup = run_direct_pass(
        backend,
        model,
        options,
        &slots.prompt_token_ids,
        &slots.answer_token_ids,
    )?;
    let measured = run_direct_pass(
        backend,
        model,
        options,
        &slots.prompt_token_ids,
        &slots.answer_token_ids,
    )?;
    let finite_readout = measured
        .numeric
        .option_logits
        .iter()
        .chain(&measured.numeric.probabilities)
        .all(|value| value.is_finite())
        && measured.numeric.allowed_token_mass.is_finite()
        && measured.numeric.full_vocab_log_normalizer.is_finite();
    if !finite_readout {
        return Err(BackendError::Decode(
            "direct smoke produced a nonfinite readout".to_owned(),
        ));
    }
    Ok(DirectSmokeReport {
        schema: "openjev-m2-smoke-v1",
        model: info.clone(),
        id: decision.id.clone(),
        prompt_sha256: prompt.prompt_sha256,
        input_tokens: slots.prompt_token_ids.len(),
        answer_token_ids: slots.answer_token_ids,
        option_logits: measured.numeric.option_logits,
        probabilities: measured.numeric.probabilities,
        choice_index: measured.numeric.choice_index,
        allowed_token_mass: measured.numeric.allowed_token_mass,
        full_vocab_argmax_id: measured.numeric.full_vocab_argmax_id,
        full_vocab_log_normalizer: measured.numeric.full_vocab_log_normalizer,
        warmup_forward_seconds: warmup.forward_seconds,
        forward_seconds: measured.forward_seconds,
        n_ctx_actual: measured.n_ctx_actual,
        n_batch_actual: measured.n_batch_actual,
        n_ubatch_actual: measured.n_ubatch_actual,
        finite_readout,
    })
}

fn run_direct_pass(
    backend: &LlamaBackend,
    model: &LlamaModel,
    options: &EngineOptions,
    prompt_tokens: &[u32],
    answer_token_ids: &[u32],
) -> Result<DirectPass> {
    let prompt_len = u32::try_from(prompt_tokens.len())
        .map_err(|_| BackendError::Context("prompt length exceeds u32".to_owned()))?;
    if prompt_len == 0 || prompt_len > options.max_tokens {
        return Err(BackendError::Context(format!(
            "prompt has {prompt_len} tokens; configured max_tokens is {}",
            options.max_tokens
        )));
    }
    if prompt_len > model.n_ctx_train() {
        return Err(BackendError::Context(format!(
            "prompt has {prompt_len} tokens; model training context is {}",
            model.n_ctx_train()
        )));
    }
    let n_ctx = match options.n_ctx {
        Some(explicit) if prompt_len > explicit => {
            return Err(BackendError::Context(format!(
                "prompt has {prompt_len} tokens; explicit n_ctx is {explicit}"
            )));
        }
        Some(explicit) => explicit,
        None => round_context(prompt_len.max(4096))?,
    };
    if n_ctx > options.max_context_tokens {
        return Err(BackendError::Context(format!(
            "required n_ctx {n_ctx} exceeds max_context_tokens {}",
            options.max_context_tokens
        )));
    }
    if n_ctx > model.n_ctx_train() {
        return Err(BackendError::Context(format!(
            "requested n_ctx {n_ctx} exceeds model training context {}",
            model.n_ctx_train()
        )));
    }
    let n_batch = options.n_batch.min(n_ctx).max(1);
    let n_ubatch = options.n_ubatch.min(n_batch).max(1);
    let thread_count = i32::try_from(options.threads)
        .map_err(|_| BackendError::Configuration("threads exceeds i32::MAX".to_owned()))?;
    let mut context_params = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch)
        .with_n_seq_max(1)
        .with_kv_unified(true)
        .with_n_threads(thread_count)
        .with_n_threads_batch(thread_count);
    if options.device == Device::Cpu {
        context_params = context_params
            .with_offload_kqv(false)
            .with_op_offload(false);
    }
    let mut context = model
        .new_context(backend, context_params)
        .map_err(|error| BackendError::Context(error.to_string()))?;
    let n_ctx_actual = context.n_ctx();
    let n_batch_actual = context.n_batch();
    let n_ubatch_actual = context.n_ubatch();
    let mut batch = LlamaBatch::new(
        usize::try_from(n_batch).expect("u32 fits usize on supported targets"),
        1,
    );
    let forward_started = Instant::now();
    let mut final_logits = None;
    let chunk_size = usize::try_from(n_batch).expect("u32 fits usize on supported targets");
    for (chunk_index, chunk) in prompt_tokens.chunks(chunk_size).enumerate() {
        batch.clear();
        let absolute_start = chunk_index
            .checked_mul(chunk_size)
            .ok_or_else(|| BackendError::Context("prompt position overflow".to_owned()))?;
        for (local_index, token_id) in chunk.iter().copied().enumerate() {
            let absolute = absolute_start
                .checked_add(local_index)
                .ok_or_else(|| BackendError::Context("prompt position overflow".to_owned()))?;
            let position = i32::try_from(absolute)
                .map_err(|_| BackendError::Context("prompt position exceeds i32".to_owned()))?;
            let native_token = i32::try_from(token_id)
                .map_err(|_| BackendError::Decode("token ID exceeds i32".to_owned()))?;
            let is_final = absolute + 1 == prompt_tokens.len();
            batch
                .add(LlamaToken(native_token), position, &[0], is_final)
                .map_err(|error| BackendError::Decode(error.to_string()))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| BackendError::Decode(error.to_string()))?;
        if absolute_start + chunk.len() == prompt_tokens.len() {
            let observed_local = chunk.len() - 1;
            validate_final_chunk_local_index(prompt_tokens.len(), chunk_size, observed_local)?;
            let final_local = i32::try_from(observed_local)
                .map_err(|_| BackendError::Decode("chunk-local index exceeds i32".to_owned()))?;
            final_logits = Some(context.get_logits_ith(final_local).to_vec());
        }
    }
    // get_logits_ith synchronizes native work; copying completes the measured
    // forward interval before f64 postprocessing begins.
    let vocabulary_logits = final_logits.ok_or_else(|| {
        BackendError::Decode("final prompt chunk produced no copied logits".to_owned())
    })?;
    let forward_seconds = forward_started.elapsed().as_secs_f64();
    let option_logits: Vec<f32> = answer_token_ids
        .iter()
        .map(|token_id| {
            usize::try_from(*token_id)
                .ok()
                .and_then(|index| vocabulary_logits.get(index).copied())
                .ok_or_else(|| {
                    BackendError::Decode(format!(
                        "answer token ID {token_id} is outside copied vocabulary logits"
                    ))
                })
        })
        .collect::<Result<_>>()?;
    let numeric = read_logits(&option_logits, &vocabulary_logits)
        .map_err(|error| BackendError::Core(error.to_string()))?;
    Ok(DirectPass {
        numeric,
        forward_seconds,
        n_ctx_actual,
        n_batch_actual,
        n_ubatch_actual,
    })
}

pub fn validate_final_chunk_local_index(
    prompt_tokens: usize,
    chunk_size: usize,
    observed_index: usize,
) -> Result<()> {
    if prompt_tokens == 0 || chunk_size == 0 {
        return Err(BackendError::Decode(
            "prompt length and chunk size must be positive".to_owned(),
        ));
    }
    let expected = (prompt_tokens - 1) % chunk_size;
    if observed_index != expected {
        return Err(BackendError::Decode(format!(
            "final logits index must be chunk-local {expected}, got {observed_index}"
        )));
    }
    Ok(())
}

fn round_context(value: u32) -> Result<u32> {
    value
        .checked_add(255)
        .map(|rounded| rounded / 256 * 256)
        .ok_or_else(|| BackendError::Context("context size overflow".to_owned()))
}

fn digest_hex(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

struct LlamaTokenizer<'a> {
    model: &'a LlamaModel,
}

impl SlotTokenizer for LlamaTokenizer<'_> {
    fn tokenize_no_bos(&self, text: &str) -> openjev_core::types::Result<Vec<u32>> {
        self.model
            .str_to_token(text, AddBos::Never)
            .map_err(|error| openjev_core::OpenJevError::Slot(error.to_string()))?
            .into_iter()
            .map(|token| {
                u32::try_from(token.0).map_err(|_| {
                    openjev_core::OpenJevError::Slot(format!(
                        "tokenizer returned negative token ID {}",
                        token.0
                    ))
                })
            })
            .collect()
    }

    fn token_piece(&self, token_id: u32) -> openjev_core::types::Result<Vec<u8>> {
        let token = i32::try_from(token_id).map_err(|_| {
            openjev_core::OpenJevError::Slot(format!("token ID {token_id} exceeds i32"))
        })?;
        self.model
            .token_to_piece_bytes(LlamaToken(token), 32, false, None)
            .map_err(|error| openjev_core::OpenJevError::Slot(error.to_string()))
    }

    fn vocabulary_size(&self) -> usize {
        usize::try_from(self.model.n_vocab()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::ModelRegistry;

    use super::*;

    #[test]
    fn qwen_template_equivalence_is_narrowly_keyed() {
        let registry = ModelRegistry::bundled().unwrap();
        let qwen_entry = registry.resolve("qwen3-0.6b").unwrap();
        let qwen = RuntimeModelSpec::from(qwen_entry.clone());
        let native = &qwen.native_reference.as_ref().unwrap().template_sha256;
        let gguf = "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361";

        assert_eq!(
            adjudicate_template(&qwen, &qwen.sha256, Some(native)).0,
            TemplateStatus::Exact
        );
        let (status, diagnostic) = adjudicate_template(&qwen, &qwen.sha256, Some(gguf));
        assert_eq!(status, TemplateStatus::ReviewedEquivalent);
        let diagnostic = diagnostic.unwrap();
        assert!(diagnostic.contains("not identical"));
        assert!(diagnostic.contains("two string messages"));

        assert_eq!(
            adjudicate_template(&qwen, &"0".repeat(64), Some(gguf)).0,
            TemplateStatus::Mismatch
        );
        assert_eq!(
            adjudicate_template(&qwen, &qwen.sha256, Some(&"2".repeat(64))).0,
            TemplateStatus::Mismatch
        );
    }

    #[test]
    fn strict_encoding_gate_rejects_spacing_and_bos_mutations() {
        let decision = openjev_core::Decision::new(
            "row",
            openjev_core::StateValue::string("state").unwrap(),
            "question",
            vec![
                openjev_core::DecisionOption {
                    id: "a".to_owned(),
                    description: "A".to_owned(),
                },
                openjev_core::DecisionOption {
                    id: "b".to_owned(),
                    description: "B".to_owned(),
                },
            ],
        )
        .unwrap();
        let prompt = prepare_prompt(&decision, PromptProfile::Qwen3).unwrap();
        let encoded = EncodedPrompt {
            id: "row".to_owned(),
            prompt_sha256: prompt.prompt_sha256.clone(),
            prompt_version: "direct-options-v1".to_owned(),
            option_ids: vec!["a".to_owned(), "b".to_owned()],
            prompt_token_ids: vec![10, 11, 12],
            answer_token_ids: vec![32, 33],
        };
        encoded
            .validate_reference(
                "row",
                &["a".to_owned(), "b".to_owned()],
                &prompt.prompt_sha256,
                3,
                &[32, 33],
            )
            .unwrap();

        // Removing Python's required colon-space changes prompt bytes/hash.
        let altered_spacing = prompt.text.replacen(": ", ":", 1);
        let altered_hash = digest_hex(&Sha256::digest(altered_spacing));
        let mut spacing_mutation = encoded.clone();
        spacing_mutation.prompt_sha256 = altered_hash;
        assert!(
            spacing_mutation
                .validate_reference(
                    "row",
                    &["a".to_owned(), "b".to_owned()],
                    &prompt.prompt_sha256,
                    3,
                    &[32, 33],
                )
                .is_err()
        );

        // An implicit BOS changes the exact no-BOS token sequence/count.
        let mut bos_mutation = encoded;
        bos_mutation.prompt_token_ids.insert(0, 151_643);
        assert!(
            bos_mutation
                .validate_reference(
                    "row",
                    &["a".to_owned(), "b".to_owned()],
                    &prompt.prompt_sha256,
                    3,
                    &[32, 33],
                )
                .is_err()
        );
    }

    #[test]
    fn final_logits_index_must_be_local_to_last_chunk() {
        validate_final_chunk_local_index(513, 512, 0).unwrap();
        assert!(validate_final_chunk_local_index(513, 512, 512).is_err());
        assert!(validate_final_chunk_local_index(0, 512, 0).is_err());
    }

    #[test]
    fn state_prefix_tokenization_drops_exactly_one_token() {
        struct ByteTokenizer;
        impl SlotTokenizer for ByteTokenizer {
            fn tokenize_no_bos(&self, text: &str) -> openjev_core::types::Result<Vec<u32>> {
                Ok(text
                    .as_bytes()
                    .iter()
                    .map(|byte| u32::from(*byte))
                    .collect())
            }
            fn token_piece(&self, token_id: u32) -> openjev_core::types::Result<Vec<u8>> {
                Ok(vec![u8::try_from(token_id).unwrap()])
            }
            fn vocabulary_size(&self) -> usize {
                256
            }
        }
        let state = StateValue::parse_json(r#"{"z": 1, "a": [true]}"#).unwrap();
        let text = state_prefix_text(&state, PromptProfile::Qwen3).unwrap();
        let prefix = encode_state_prefix(&ByteTokenizer, &state, PromptProfile::Qwen3).unwrap();
        assert_eq!(prefix.len() + 1, text.len());
        assert_eq!(
            prefix,
            text.as_bytes()[..text.len() - 1]
                .iter()
                .map(|b| u32::from(*b))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn wave_planner_budgets_prefix_plus_sum_of_active_suffixes() {
        let sequences = vec![vec![0; 4], vec![0; 5], vec![0; 6], vec![0; 2]];
        let waves = plan_waves(10, &sequences, 20, 2).unwrap();
        assert_eq!(waves, vec![vec![0, 1], vec![2, 3]]);
        assert!(plan_waves(10, &[vec![0; 11]], 20, 2).is_err());
        assert!(plan_waves(10, &[], 20, 2).is_err());
        let one_per_wave = plan_waves(1, &sequences, 100, 1).unwrap();
        assert_eq!(one_per_wave.len(), 4);
    }

    #[test]
    fn ragged_scheduler_covers_empty_uneven_and_invalid_bounds() {
        assert_eq!(ragged_step(&[], &[], 4).unwrap(), None);
        assert_eq!(
            ragged_step(&[0, 0], &[2, 5], 4).unwrap(),
            Some((vec![0, 1], 2))
        );
        assert_eq!(
            ragged_step(&[2, 2], &[2, 5], 4).unwrap(),
            Some((vec![1], 3))
        );
        assert!(ragged_step(&[3], &[2], 4).is_err());
        assert!(ragged_step(&[0, 0], &[1, 1], 1).is_err());
        assert!(ragged_step(&[0], &[], 4).is_err());
        assert!(ragged_step(&[0], &[1], 0).is_err());
    }

    #[test]
    fn branch_reuse_requires_successful_full_sequence_clear() {
        require_sequence_clear(1, true).unwrap();
        let error = require_sequence_clear(1, false).unwrap_err();
        assert!(error.to_string().contains("full clear returned false"));
    }

    #[test]
    fn owner_thread_join_reports_panics_and_clean_shutdown() {
        join_owner_thread(std::thread::spawn(|| {})).unwrap();
        let error =
            join_owner_thread(std::thread::spawn(|| panic!("test owner panic"))).unwrap_err();
        assert!(matches!(error, BackendError::Worker(_)));
        assert!(error.to_string().contains("panicked"));
    }
}
