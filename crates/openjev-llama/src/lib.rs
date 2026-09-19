//! Registry, verified model cache, and feature-gated llama.cpp backend.
//!
//! The default build parses and tests the bundled registry and cache integrity
//! logic without compiling llama.cpp or accessing the network. Native loading
//! and the finite direct-smoke worker require the `native`, `metal`, or `cuda`
//! feature.

use std::path::PathBuf;

use openjev_core::PromptProfile;
use thiserror::Error;

pub mod cache;
pub mod registry;

#[cfg(feature = "native")]
pub mod engine;

pub use cache::{CacheOptions, ModelCache, VerifiedArtifact, hash_file};
pub use registry::{ModelEntry, ModelRegistry, NativeReferenceSpec, TemplateEquivalenceSpec};

#[cfg(feature = "native")]
pub use engine::{
    DirectSmokeReport, EngineHandle, EngineOptions, LoadedModelInfo, RuntimeDevice, TemplateStatus,
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
