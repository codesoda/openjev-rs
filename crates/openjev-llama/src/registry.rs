use std::collections::HashSet;

use openjev_core::PromptProfile;
use serde::{Deserialize, Serialize};

use crate::{BackendError, CacheOptions, ModelCache, Result, VerifiedArtifact};

const MANIFEST_JSON: &str = include_str!("../../../manifests/models.json");
const REGISTRY_SCHEMA: &str = "openjev-model-registry-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeReferenceSpec {
    pub source: String,
    pub revision: String,
    pub dtype: String,
    pub tokenizer_artifact: String,
    pub tokenizer_sha256: String,
    pub template_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateEquivalenceSpec {
    pub artifact_sha256: String,
    pub gguf_template_sha256: String,
    pub native_profile_sha256: String,
    pub scope: String,
    pub evidence: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub quant: String,
    pub profile: PromptProfile,
    pub native_reference: NativeReferenceSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_equivalence: Option<TemplateEquivalenceSpec>,
}

impl ModelEntry {
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("id", self.id.as_str()),
            ("repo", self.repo.as_str()),
            ("file", self.file.as_str()),
            ("quant", self.quant.as_str()),
            (
                "native_reference.source",
                self.native_reference.source.as_str(),
            ),
            (
                "native_reference.tokenizer_artifact",
                self.native_reference.tokenizer_artifact.as_str(),
            ),
        ] {
            if value.is_empty() {
                return Err(BackendError::Manifest(format!(
                    "model {:?} has an empty {field}",
                    self.id
                )));
            }
        }
        validate_repo(&self.repo, "repo", &self.id)?;
        validate_relative_file(&self.file, "file", &self.id)?;
        validate_repo(
            &self.native_reference.source,
            "native_reference.source",
            &self.id,
        )?;
        validate_relative_file(
            &self.native_reference.tokenizer_artifact,
            "native_reference.tokenizer_artifact",
            &self.id,
        )?;
        if self.bytes == 0 {
            return Err(BackendError::Manifest(format!(
                "model {:?} has a zero byte length",
                self.id
            )));
        }
        check_lower_hex(&self.revision, 40, "revision", &self.id)?;
        check_lower_hex(&self.sha256, 64, "sha256", &self.id)?;
        check_lower_hex(
            &self.native_reference.revision,
            40,
            "native_reference.revision",
            &self.id,
        )?;
        check_lower_hex(
            &self.native_reference.tokenizer_sha256,
            64,
            "native_reference.tokenizer_sha256",
            &self.id,
        )?;
        check_lower_hex(
            &self.native_reference.template_sha256,
            64,
            "native_reference.template_sha256",
            &self.id,
        )?;
        if self.native_reference.dtype != "bfloat16" {
            return Err(BackendError::Manifest(format!(
                "model {:?} native reference dtype must be bfloat16",
                self.id
            )));
        }
        if let Some(equivalence) = &self.template_equivalence {
            check_lower_hex(
                &equivalence.artifact_sha256,
                64,
                "template_equivalence.artifact_sha256",
                &self.id,
            )?;
            check_lower_hex(
                &equivalence.gguf_template_sha256,
                64,
                "template_equivalence.gguf_template_sha256",
                &self.id,
            )?;
            check_lower_hex(
                &equivalence.native_profile_sha256,
                64,
                "template_equivalence.native_profile_sha256",
                &self.id,
            )?;
            if equivalence.scope.is_empty() || equivalence.evidence.is_empty() {
                return Err(BackendError::Manifest(format!(
                    "model {:?} template equivalence scope and evidence must be nonempty",
                    self.id
                )));
            }
            if equivalence.artifact_sha256 != self.sha256
                || equivalence.native_profile_sha256 != self.native_reference.template_sha256
            {
                return Err(BackendError::Manifest(format!(
                    "model {:?} template equivalence is not keyed to its artifact and native profile hashes",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegistryManifest {
    schema: String,
    default_model: String,
    models: Vec<ModelEntry>,
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    manifest: RegistryManifest,
}

impl ModelRegistry {
    pub fn bundled() -> Result<Self> {
        let manifest: RegistryManifest = serde_json::from_str(MANIFEST_JSON)
            .map_err(|error| BackendError::Manifest(error.to_string()))?;
        if manifest.schema != REGISTRY_SCHEMA {
            return Err(BackendError::Manifest(format!(
                "unexpected registry schema {:?}",
                manifest.schema
            )));
        }
        if manifest.models.len() != 3 {
            return Err(BackendError::Manifest(format!(
                "bundled registry must contain exactly three models, found {}",
                manifest.models.len()
            )));
        }
        let mut ids = HashSet::with_capacity(manifest.models.len());
        for model in &manifest.models {
            model.validate()?;
            if !ids.insert(model.id.as_str()) {
                return Err(BackendError::Manifest(format!(
                    "duplicate model ID {:?}",
                    model.id
                )));
            }
        }
        if !ids.contains(manifest.default_model.as_str()) {
            return Err(BackendError::Manifest(format!(
                "default model {:?} is absent",
                manifest.default_model
            )));
        }
        Ok(Self { manifest })
    }

    #[must_use]
    pub fn list(&self) -> &[ModelEntry] {
        &self.manifest.models
    }

    #[must_use]
    pub fn default_model(&self) -> &str {
        &self.manifest.default_model
    }

    pub fn resolve(&self, id: &str) -> Result<&ModelEntry> {
        self.manifest
            .models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| BackendError::UnknownModel(id.to_owned()))
    }

    pub fn path(&self, cache: &ModelCache, id: &str) -> Result<VerifiedArtifact> {
        let model = self.resolve(id)?;
        cache.ensure_with(
            model,
            CacheOptions {
                offline: true,
                repair: false,
            },
            |_, _| unreachable!("offline cache resolution cannot fetch"),
        )
    }

    #[cfg(feature = "native")]
    pub fn pull(&self, cache: &ModelCache, id: &str, repair: bool) -> Result<VerifiedArtifact> {
        cache.ensure(
            self.resolve(id)?,
            CacheOptions {
                offline: false,
                repair,
            },
        )
    }
}

fn validate_repo(value: &str, field: &str, id: &str) -> Result<()> {
    let mut components = value.split('/');
    let owner = components.next().unwrap_or_default();
    let name = components.next().unwrap_or_default();
    if components.next().is_some() || !is_portable_component(owner) || !is_portable_component(name)
    {
        return Err(BackendError::Manifest(format!(
            "model {id:?} {field} must be exactly two portable OWNER/NAME components"
        )));
    }
    Ok(())
}

fn validate_relative_file(value: &str, field: &str, id: &str) -> Result<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with("//")
        || value.contains('\\')
        || !value.split('/').all(is_portable_component)
    {
        return Err(BackendError::Manifest(format!(
            "model {id:?} {field} must be a portable safe relative path"
        )));
    }
    Ok(())
}

fn is_portable_component(value: &str) -> bool {
    let stem = value
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    let windows_reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.ends_with('.')
        && !windows_reserved
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn check_lower_hex(value: &str, len: usize, field: &str, id: &str) -> Result<()> {
    if value.len() != len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BackendError::Manifest(format!(
            "model {id:?} {field} must be {len} lowercase hex characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn bundled_registry_has_exact_pins() {
        let registry = ModelRegistry::bundled().unwrap();
        assert_eq!(registry.default_model(), "minicpm5-2b");
        assert_eq!(registry.list().len(), 3);
        let qwen = registry.resolve("qwen3-0.6b").unwrap();
        assert_eq!(qwen.bytes, 639_446_688);
        assert_eq!(
            qwen.sha256,
            "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
        );
        assert!(registry.resolve("not-a-model").is_err());
        let equivalence = qwen.template_equivalence.as_ref().unwrap();
        assert_eq!(equivalence.artifact_sha256, qwen.sha256);
        assert_eq!(
            equivalence.gguf_template_sha256,
            "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361"
        );
    }

    #[test]
    fn qwen_template_fixtures_have_reviewed_identities() {
        let gguf = include_bytes!("../../../fixtures/templates/qwen3-gguf-57f1fd00.jinja");
        let native = include_bytes!("../../../fixtures/templates/qwen3-native-a55ee1b1.jinja");
        assert_eq!(gguf.len(), 4_100);
        assert_eq!(native.len(), 4_168);
        let hex = |bytes: &[u8]| {
            Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hex(gguf),
            "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361"
        );
        assert_eq!(
            hex(native),
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        );
    }

    #[test]
    fn rejects_unsafe_repo_and_file_paths() {
        let registry = ModelRegistry::bundled().unwrap();
        let base = registry.resolve("qwen3-0.6b").unwrap().clone();
        for repo in [
            "owner",
            "owner/name/extra",
            "/owner/name",
            "owner/../name",
            "owner/.",
            "owner//name",
            "owner\\name",
            "C:/name",
            "//server/share",
            "owner/CON",
            "owner/name.",
        ] {
            let mut model = base.clone();
            model.repo = repo.to_owned();
            assert!(model.validate().is_err(), "accepted unsafe repo {repo:?}");
        }
        for file in [
            "/tmp/model.gguf",
            "../model.gguf",
            "nested/../model.gguf",
            "nested/./model.gguf",
            "nested//model.gguf",
            "nested\\model.gguf",
            "C:/model.gguf",
            "//server/model.gguf",
            "nested/CON",
            "nested/name.",
            "",
        ] {
            let mut model = base.clone();
            model.file = file.to_owned();
            assert!(model.validate().is_err(), "accepted unsafe file {file:?}");
        }
        for (source, artifact) in [
            ("owner/../native", "tokenizer_config.json"),
            ("owner/native", "../tokenizer_config.json"),
        ] {
            let mut model = base.clone();
            model.native_reference.source = source.to_owned();
            model.native_reference.tokenizer_artifact = artifact.to_owned();
            assert!(model.validate().is_err());
        }
        let mut nested = base;
        nested.file = "safe/nested/model-Q4_K_M.gguf".to_owned();
        nested.validate().unwrap();
    }
}
