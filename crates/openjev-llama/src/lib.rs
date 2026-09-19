//! Registry, verified model cache, and feature-gated llama.cpp backend.
//!
//! The default build parses and tests the bundled registry and cache integrity
//! logic without compiling llama.cpp or accessing the network. Native loading
//! and the finite direct-smoke worker require the `native`, `metal`, or `cuda`
//! feature.

use std::path::PathBuf;

use openjev_core::{Integrity, PromptProfile};
use thiserror::Error;

pub mod cache;
pub mod model;
pub mod probe;
pub mod registry;

#[cfg(feature = "native")]
pub mod engine;

pub use cache::{CacheOptions, ModelCache, VerifiedArtifact, hash_file};
#[cfg(feature = "native")]
pub use model::resolve_model_spec;
pub use model::{validate_hub_identity, validate_sha256};
pub use probe::{
    MAX_ABS_SLOT_LOGIT, MAX_PROBABILITY_DELTA, NATIVE_PIN, PROBE_SCHEMA, PROBE_SUITE_VERSION,
    ProbeCaseResult, ProbeCaseStatus, ProbeConfiguration, ProbeEligibility, ProbeMode,
    ProbePublication, ProbeReceipt, begin_probe_publication, load_passing_receipt, probe_id,
    receipt_path, write_receipt,
};
pub use registry::{ModelEntry, ModelRegistry, NativeReferenceSpec, TemplateEquivalenceSpec};

#[cfg(feature = "native")]
pub use engine::{
    DirectSmokeReport, EncodedPrompt, EngineHandle, EngineOptions, LoadedModelInfo, RuntimeDevice,
    TemplateStatus, validate_final_chunk_local_index,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendFeature {
    Disabled,
    Native,
    Metal,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSpec {
    RegistryId(String),
    Local {
        path: PathBuf,
        expected_sha256: Option<String>,
        profile: PromptProfile,
    },
    Hub {
        repo: String,
        revision: String,
        file: String,
        expected_sha256: String,
        profile: PromptProfile,
    },
}

/// Fully resolved model identity consumed by the owner-thread engine.
///
/// Registered artifacts retain their reviewed native/template metadata. Custom
/// artifacts always use an explicit profile override and never acquire a
/// fabricated native reference or golden-equivalence claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeModelSpec {
    pub(crate) id: String,
    pub(crate) source: String,
    pub(crate) revision: String,
    pub(crate) file: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) quant: String,
    pub(crate) profile: PromptProfile,
    pub(crate) integrity: Integrity,
    pub(crate) native_reference: Option<NativeReferenceSpec>,
    pub(crate) template_equivalence: Option<TemplateEquivalenceSpec>,
    pub(crate) template_override: bool,
}

impl RuntimeModelSpec {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub const fn integrity(&self) -> Integrity {
        self.integrity
    }

    #[must_use]
    pub const fn profile(&self) -> PromptProfile {
        self.profile
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.sha256
    }

    #[cfg(feature = "native")]
    pub(crate) fn validate(&self) -> Result<()> {
        if self.id.is_empty()
            || self.source.is_empty()
            || self.revision.is_empty()
            || self.file.is_empty()
            || self.bytes == 0
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(BackendError::Configuration(
                "resolved model identity is incomplete or invalid".to_owned(),
            ));
        }
        if self.template_override {
            if self.native_reference.is_some()
                || self.template_equivalence.is_some()
                || self.integrity == Integrity::ManifestSha256
            {
                return Err(BackendError::Configuration(
                    "custom template overrides cannot claim registered native/equivalence metadata"
                        .to_owned(),
                ));
            }
        } else if self.native_reference.is_none() || self.integrity != Integrity::ManifestSha256 {
            return Err(BackendError::Configuration(
                "registered model identity requires manifest integrity and native metadata"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

impl From<ModelEntry> for RuntimeModelSpec {
    fn from(model: ModelEntry) -> Self {
        Self {
            id: model.id,
            source: model.repo,
            revision: model.revision,
            file: model.file,
            bytes: model.bytes,
            sha256: model.sha256,
            quant: model.quant,
            profile: model.profile,
            integrity: Integrity::ManifestSha256,
            native_reference: Some(model.native_reference),
            template_equivalence: model.template_equivalence,
            template_override: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedModel {
    pub(crate) model: RuntimeModelSpec,
    pub(crate) artifact: VerifiedArtifact,
}

impl ResolvedModel {
    #[must_use]
    pub const fn model(&self) -> &RuntimeModelSpec {
        &self.model
    }

    #[must_use]
    pub const fn artifact(&self) -> &VerifiedArtifact {
        &self.artifact
    }

    #[must_use]
    pub fn into_parts(self) -> (RuntimeModelSpec, VerifiedArtifact) {
        (self.model, self.artifact)
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend unavailable: build openjev-llama with native, metal, or cuda")]
    Unavailable,
    #[error("invalid model manifest: {0}")]
    Manifest(String),
    #[error("unknown model ID {0:?}")]
    UnknownModel(String),
    #[error("cannot resolve model cache root: {0}")]
    CacheRoot(String),
    #[error("unsafe cache filesystem state at {path}: {message}", path = path.display())]
    CacheSafety { path: PathBuf, message: String },
    #[error("offline cache miss for model {model_id:?} at {path}", path = path.display())]
    OfflineMiss { model_id: String, path: PathBuf },
    #[error(
        "artifact integrity failure at {path}: expected {expected_bytes} bytes / {expected_sha256}, got {actual_bytes} bytes / {actual_sha256}",
        path = path.display()
    )]
    Integrity {
        path: PathBuf,
        expected_bytes: u64,
        actual_bytes: u64,
        expected_sha256: String,
        actual_sha256: String,
    },
    #[error(
        "caller artifact SHA-256 failure at {path}: expected {expected_sha256}, got {actual_sha256}",
        path = path.display()
    )]
    CallerIntegrity {
        path: PathBuf,
        expected_sha256: String,
        actual_sha256: String,
    },
    #[error(
        "artifact changed while hashing {path}: metadata said {before} bytes, read {after} bytes",
        path = path.display()
    )]
    ArtifactChanged {
        path: PathBuf,
        before: u64,
        after: u64,
    },
    #[error("failed to lock cache entry {path}: {source}", path = path.display())]
    Lock {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "failed to reuse cached artifact {source_path} at {destination}: {source}",
        source_path = source_path.display(),
        destination = destination.display()
    )]
    CacheImport {
        source_path: PathBuf,
        destination: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Hugging Face download failed: {0}")]
    Download(String),
    #[error("cache receipt failed: {0}")]
    Receipt(String),
    #[error("invalid engine configuration: {0}")]
    Configuration(String),
    #[error("native backend failed: {0}")]
    Native(String),
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("model metadata failed: {0}")]
    Metadata(String),
    #[error("context failed: {0}")]
    Context(String),
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("backend-neutral core failed: {0}")]
    Core(String),
    #[error("owner worker failed: {0}")]
    Worker(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, BackendError>;

#[must_use]
pub const fn compiled_backend_feature() -> BackendFeature {
    if cfg!(feature = "cuda") {
        BackendFeature::Cuda
    } else if cfg!(feature = "metal") {
        BackendFeature::Metal
    } else if cfg!(feature = "native") {
        BackendFeature::Native
    } else {
        BackendFeature::Disabled
    }
}

pub fn unavailable() -> Result<()> {
    Err(BackendError::Unavailable)
}

/// Direct scoring creates a fresh context and prefills the complete prompt.
/// Artifact download-cache status is separate runner/cache metadata and must not
/// be reported as inference prefix reuse.
#[cfg(any(feature = "native", test))]
pub(crate) const fn direct_inference_cache_hit() -> Option<bool> {
    Some(false)
}

#[cfg(test)]
mod readout_cache_semantics_tests {
    use super::direct_inference_cache_hit;

    #[test]
    fn cached_artifact_does_not_make_repeated_direct_readouts_cache_hits() {
        let artifact_cache_hit = true;
        assert!(
            artifact_cache_hit,
            "fixture represents a cached GGUF artifact"
        );
        assert_eq!(direct_inference_cache_hit(), Some(false));
        assert_eq!(direct_inference_cache_hit(), Some(false));
    }
}

#[cfg(all(test, not(feature = "native")))]
mod tests {
    use super::*;

    #[test]
    fn default_build_is_backend_disabled_but_registry_is_available() {
        assert_eq!(compiled_backend_feature(), BackendFeature::Disabled);
        assert!(unavailable().is_err());
        assert_eq!(ModelRegistry::bundled().unwrap().list().len(), 3);
    }
}
