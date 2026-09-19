//! Feature-gated llama.cpp backend boundary.
//!
//! M1 intentionally provides no production inference implementation. Enabling
//! `native`, `metal`, or `cuda` only selects the pinned build dependencies for
//! later milestones.

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendFeature {
    Disabled,
    Native,
    Metal,
    Cuda,
}

#[derive(Debug, Error)]
#[error("llama backend is unavailable in the M1 skeleton")]
pub struct BackendUnavailable;

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

pub fn unavailable() -> Result<(), BackendUnavailable> {
    Err(BackendUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "native"))]
    fn default_build_is_backend_disabled() {
        assert_eq!(compiled_backend_feature(), BackendFeature::Disabled);
        assert!(unavailable().is_err());
    }
}
