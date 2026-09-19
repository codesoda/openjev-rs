#[cfg(feature = "native")]
use std::{fs, path::Path};

#[cfg(feature = "native")]
use openjev_core::Integrity;

use crate::{BackendError, Result};
#[cfg(feature = "native")]
use crate::{
    CacheOptions, ModelCache, ModelRegistry, ModelSpec, ResolvedModel, RuntimeModelSpec,
    VerifiedArtifact, hash_file,
};

pub fn validate_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(BackendError::Configuration(format!(
            "{field} must be 64 lowercase hexadecimal characters"
        )))
    }
}

pub fn validate_hub_identity(repo: &str, revision: &str, file: &str) -> Result<()> {
    let components: Vec<_> = repo.split('/').collect();
    if components.len() != 2
        || components
            .iter()
            .any(|value| !portable_component_str(value))
    {
        return Err(BackendError::Configuration(
            "custom Hub repository must be exactly two portable OWNER/NAME components".to_owned(),
        ));
    }
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BackendError::Configuration(
            "custom Hub revision must be a 40-character lowercase commit hash".to_owned(),
        ));
    }
    if file.is_empty()
        || file.starts_with('/')
        || file.contains('\\')
        || !file.split('/').all(portable_component_str)
    {
        return Err(BackendError::Configuration(
            "custom Hub filename must be a portable safe relative path".to_owned(),
        ));
    }
    Ok(())
}

fn portable_component_str(value: &str) -> bool {
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

#[cfg(feature = "native")]
pub fn resolve_model_spec(
    registry: &ModelRegistry,
    cache: &ModelCache,
    spec: &ModelSpec,
    options: CacheOptions,
) -> Result<ResolvedModel> {
    match spec {
        ModelSpec::RegistryId(id) => {
            let entry = registry.resolve(id)?.clone();
            let artifact = cache.ensure(&entry, options)?;
            Ok(ResolvedModel {
                model: entry.into(),
                artifact,
            })
        }
        ModelSpec::Local {
            path,
            expected_sha256,
            profile,
        } => resolve_local(path, expected_sha256.as_deref(), *profile),
        ModelSpec::Hub {
            repo,
            revision,
            file,
            expected_sha256,
            profile,
        } => resolve_hub(
            cache,
            repo,
            revision,
            file,
            expected_sha256,
            *profile,
            options,
        ),
    }
}

#[cfg(feature = "native")]
fn resolve_local(
    path: &Path,
    expected_sha256: Option<&str>,
    profile: openjev_core::PromptProfile,
) -> Result<ResolvedModel> {
    if let Some(expected) = expected_sha256 {
        validate_sha256(expected, "--model-sha256")?;
    }
    let canonical = fs::canonicalize(path)?;
    if !fs::metadata(&canonical)?.is_file() {
        return Err(BackendError::Configuration(format!(
            "local model {} is not a regular file",
            path.display()
        )));
    }
    let (bytes, sha256) = hash_file(&canonical)?;
    if let Some(expected) = expected_sha256
        && sha256 != expected
    {
        return Err(BackendError::CallerIntegrity {
            path: canonical,
            expected_sha256: expected.to_owned(),
            actual_sha256: sha256,
        });
    }
    let integrity = if expected_sha256.is_some() {
        Integrity::CallerSha256
    } else {
        Integrity::LocalUnverified
    };
    let display = canonical.display().to_string();
    let id = display.clone();
    Ok(ResolvedModel {
        model: RuntimeModelSpec {
            id: id.clone(),
            source: "local".to_owned(),
            revision: format!("sha256:{sha256}"),
            file: display,
            bytes,
            sha256: sha256.clone(),
            quant: "unknown".to_owned(),
            profile,
            integrity,
            native_reference: None,
            template_equivalence: None,
            template_override: true,
        },
        artifact: VerifiedArtifact {
            model_id: id,
            path: canonical,
            bytes,
            sha256,
            cache_hit: false,
            imported_from_huggingface_cache: false,
        },
    })
}

#[cfg(feature = "native")]
#[allow(clippy::too_many_arguments)]
fn resolve_hub(
    cache: &ModelCache,
    repo: &str,
    revision: &str,
    file: &str,
    expected_sha256: &str,
    profile: openjev_core::PromptProfile,
    options: CacheOptions,
) -> Result<ResolvedModel> {
    validate_hub_identity(repo, revision, file)?;
    validate_sha256(expected_sha256, "--model-sha256")?;
    let artifact = cache.ensure_hub_sha256_with(
        repo,
        revision,
        file,
        expected_sha256,
        options.offline,
        |hub_root| {
            let (owner, name) = repo
                .split_once('/')
                .expect("validated custom repository contains one slash");
            let client = hf_hub::HFClient::builder()
                .cache_dir(hub_root)
                .build_sync()
                .map_err(|error| BackendError::Download(error.to_string()))?;
            client
                .model(owner, name)
                .download_file()
                .filename(file.to_owned())
                .revision(revision.to_owned())
                .force_download(options.repair)
                .local_files_only(false)
                .send()
                .map_err(|error| BackendError::Download(error.to_string()))
        },
    )?;
    let bytes = artifact.bytes;
    let sha256 = artifact.sha256.clone();
    let id = format!("hf:{repo}@{revision}:{file}");
    Ok(ResolvedModel {
        model: RuntimeModelSpec {
            id: id.clone(),
            source: repo.to_owned(),
            revision: revision.to_owned(),
            file: file.to_owned(),
            bytes,
            sha256: sha256.clone(),
            quant: "unknown".to_owned(),
            profile,
            integrity: Integrity::CallerSha256,
            native_reference: None,
            template_equivalence: None,
            template_override: true,
        },
        artifact,
    })
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use openjev_core::{Integrity, PromptProfile};
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn local_custom_artifact_is_hashed_in_place_without_a_golden_claim() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("custom.gguf");
        fs::write(&path, b"tiny local artifact").unwrap();
        let canonical = fs::canonicalize(&path).unwrap();
        let registry = ModelRegistry::bundled().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let resolved = resolve_model_spec(
            &registry,
            &cache,
            &ModelSpec::Local {
                path: path.clone(),
                expected_sha256: None,
                profile: PromptProfile::Qwen3,
            },
            CacheOptions {
                offline: true,
                repair: false,
            },
        )
        .unwrap();
        assert_eq!(resolved.model().integrity(), Integrity::LocalUnverified);
        assert!(resolved.model().template_override);
        assert!(resolved.model().native_reference.is_none());
        assert_eq!(resolved.artifact().path, canonical);
        assert_eq!(fs::read(&path).unwrap(), b"tiny local artifact");
        assert!(!cache.root().exists());
    }
}
