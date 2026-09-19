use std::{
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
    PromptProfile, Readout, SlotTokenizer, TemplateMetadataStatus, prepare_prompt, read_logits,
    standard_limitations, verify_slots,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    BackendError, ModelEntry, Result, RuntimeModelSpec, VerifiedArtifact, cache::hash_file,
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
}

impl EngineOptions {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("threads", self.threads),
            ("max_tokens", self.max_tokens),
            ("max_context_tokens", self.max_context_tokens),
            ("n_batch", self.n_batch),
            ("n_ubatch", self.n_ubatch),
        ] {
            if value == 0 {
                return Err(BackendError::Configuration(format!(
                    "{name} must be positive"
                )));
            }
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
    fn owner_thread_join_reports_panics_and_clean_shutdown() {
        join_owner_thread(std::thread::spawn(|| {})).unwrap();
        let error =
            join_owner_thread(std::thread::spawn(|| panic!("test owner panic"))).unwrap_err();
        assert!(matches!(error, BackendError::Worker(_)));
        assert!(error.to_string().contains("panicked"));
    }
}
