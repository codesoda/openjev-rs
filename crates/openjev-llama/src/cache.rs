use std::{
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{BufReader, ErrorKind, Read, Write},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{BackendError, ModelEntry, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheOptions {
    pub offline: bool,
    pub repair: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerifiedArtifact {
    pub model_id: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub cache_hit: bool,
    pub imported_from_huggingface_cache: bool,
}

#[derive(Clone, Debug)]
pub struct ModelCache {
    root: PathBuf,
    external_hub_root: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Receipt {
    schema: String,
    model_id: String,
    repo: String,
    revision: String,
    file: String,
    bytes: u64,
    sha256: String,
}

impl ModelCache {
    pub fn resolve_root(cache_dir: Option<&Path>) -> Result<PathBuf> {
        if let Some(path) = cache_dir {
            return Ok(path.to_path_buf());
        }
        if let Some(path) = std::env::var_os("OPENJEV_HOME") {
            if path.is_empty() {
                return Err(BackendError::CacheRoot(
                    "OPENJEV_HOME is set but empty".to_owned(),
                ));
            }
            return Ok(PathBuf::from(path));
        }
        let home = std::env::var_os("HOME").ok_or_else(|| {
            BackendError::CacheRoot(
                "cannot resolve ~/.cache/openjev because HOME is not set".to_owned(),
            )
        })?;
        Ok(PathBuf::from(home).join(".cache/openjev"))
    }

    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        let external_hub_root = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".cache/huggingface/hub"));
        Self {
            root,
            external_hub_root,
        }
    }

    #[cfg(test)]
    fn with_external_hub(root: PathBuf, external_hub_root: PathBuf) -> Self {
        Self {
            root,
            external_hub_root: Some(external_hub_root),
        }
    }

    pub fn from_precedence(cache_dir: Option<&Path>) -> Result<Self> {
        Ok(Self::new(Self::resolve_root(cache_dir)?))
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn hub_root(&self) -> PathBuf {
        self.root.join("hub")
    }

    #[must_use]
    pub fn model_path(&self, model: &ModelEntry) -> PathBuf {
        self.hub_root()
            .join(repo_folder(model))
            .join("snapshots")
            .join(&model.revision)
            .join(&model.file)
    }

    #[cfg(feature = "native")]
    pub fn ensure(&self, model: &ModelEntry, options: CacheOptions) -> Result<VerifiedArtifact> {
        self.ensure_with(model, options, |expected, hub_root| {
            let (owner, name) = expected.repo.split_once('/').ok_or_else(|| {
                BackendError::Manifest(format!("repository {:?} is not OWNER/NAME", expected.repo))
            })?;
            let client = hf_hub::HFClient::builder()
                .cache_dir(hub_root)
                .build_sync()
                .map_err(|error| BackendError::Download(error.to_string()))?;
            client
                .model(owner, name)
                .download_file()
                .filename(expected.file.clone())
                .revision(expected.revision.clone())
                .force_download(options.repair)
                .local_files_only(options.offline)
                .send()
                .map_err(|error| BackendError::Download(error.to_string()))
        })
    }

    pub fn verify_path(&self, model: &ModelEntry, path: &Path) -> Result<VerifiedArtifact> {
        let (bytes, sha256) = hash_file(path)?;
        if bytes != model.bytes || sha256 != model.sha256 {
            return Err(BackendError::Integrity {
                path: path.to_path_buf(),
                expected_bytes: model.bytes,
                actual_bytes: bytes,
                expected_sha256: model.sha256.clone(),
                actual_sha256: sha256,
            });
        }
        Ok(VerifiedArtifact {
            model_id: model.id.clone(),
            path: path.to_path_buf(),
            bytes,
            sha256,
            cache_hit: true,
            imported_from_huggingface_cache: false,
        })
    }

    /// Resolve and verify an artifact with an injected fetcher.
    ///
    /// Validation happens before any filesystem operation. The fetcher runs
    /// only after the process-safe per-artifact lock is held, verified cache
    /// reuse is exhausted, and offline mode is rejected.
    pub fn ensure_with<F>(
        &self,
        model: &ModelEntry,
        options: CacheOptions,
        fetch: F,
    ) -> Result<VerifiedArtifact>
    where
        F: FnOnce(&ModelEntry, &Path) -> Result<PathBuf>,
    {
        model.validate()?;
        let _lock = self.lock_model(model)?;
        let snapshot = self.model_path(model);
        let blob = self.blob_path(model);

        if path_present(&snapshot)? {
            match self.verify_owned_path(model, &snapshot) {
                Ok(mut artifact) => {
                    self.write_receipt(model)?;
                    artifact.cache_hit = true;
                    return Ok(artifact);
                }
                Err(error) if !options.repair => return Err(error),
                Err(_) => {
                    self.invalidate_receipt(model)?;
                    if path_present(&blob)? && self.verify_owned_path(model, &blob).is_ok() {
                        self.remove_controlled_entry(&snapshot, &self.hub_root())?;
                        self.import_hard_link(model, &blob, &snapshot)?;
                        let mut artifact = self.verify_owned_path(model, &snapshot)?;
                        self.write_receipt(model)?;
                        artifact.cache_hit = true;
                        return Ok(artifact);
                    }
                    self.quarantine_owned_artifact(model, &snapshot, &blob)?;
                }
            }
        } else if path_present(&blob)? {
            match self.verify_owned_path(model, &blob) {
                Ok(_) => {
                    self.import_hard_link(model, &blob, &snapshot)?;
                    let mut artifact = self.verify_owned_path(model, &snapshot)?;
                    self.write_receipt(model)?;
                    artifact.cache_hit = true;
                    return Ok(artifact);
                }
                Err(error) if !options.repair => return Err(error),
                Err(_) => {
                    self.invalidate_receipt(model)?;
                    self.quarantine_owned_artifact(model, &snapshot, &blob)?;
                }
            }
        } else {
            self.invalidate_receipt(model)?;
        }

        if let Some(source) = self.default_huggingface_path(model)
            && path_present(&source)?
        {
            match self.verify_external_source(model, &source) {
                Ok(_) => {
                    self.import_hard_link(model, &source, &snapshot)?;
                    let mut artifact = self.verify_owned_path(model, &snapshot)?;
                    self.write_receipt(model)?;
                    artifact.cache_hit = true;
                    artifact.imported_from_huggingface_cache = true;
                    return Ok(artifact);
                }
                Err(error) if !options.repair => return Err(error),
                Err(_) => {
                    // An external HF cache is untrusted input. Explicit repair
                    // bypasses it but never changes or quarantines it.
                }
            }
        }

        if options.offline {
            return Err(BackendError::OfflineMiss {
                model_id: model.id.clone(),
                path: snapshot,
            });
        }

        self.create_contained_dir(&self.root, &self.hub_root())?;
        let fetched = fetch(model, &self.hub_root())?;
        let fetched_canonical = self.verify_external_source(model, &fetched)?;
        let snapshot_canonical = fs::canonicalize(&snapshot).ok();
        if snapshot_canonical.as_ref() != Some(&fetched_canonical) {
            self.import_hard_link(model, &fetched_canonical, &snapshot)?;
        }
        let mut artifact = self.verify_owned_path(model, &snapshot)?;
        self.write_receipt(model)?;
        artifact.cache_hit = false;
        Ok(artifact)
    }

    fn lock_model(&self, model: &ModelEntry) -> Result<File> {
        let directory = self.root.join("openjev/locks");
        self.create_contained_dir(&self.root, &directory)?;
        let path = directory.join(format!("{}.lock", model.sha256));
        self.require_mutation_parent(&path, &directory)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        File::lock(&file).map_err(|error| BackendError::Lock {
            path,
            source: error,
        })?;
        Ok(file)
    }

    fn default_huggingface_path(&self, model: &ModelEntry) -> Option<PathBuf> {
        let default_hub = self.external_hub_root.as_ref()?;
        if default_hub == &self.hub_root() {
            return None;
        }
        Some(
            default_hub
                .join(repo_folder(model))
                .join("snapshots")
                .join(&model.revision)
                .join(&model.file),
        )
    }

    fn blob_path(&self, model: &ModelEntry) -> PathBuf {
        self.hub_root()
            .join(repo_folder(model))
            .join("blobs")
            .join(&model.sha256)
    }

    fn verify_owned_path(&self, model: &ModelEntry, path: &Path) -> Result<VerifiedArtifact> {
        let canonical = self.canonical_owned_regular(path, &self.hub_root())?;
        let mut artifact = self.verify_path(model, &canonical)?;
        artifact.path = path.to_path_buf();
        Ok(artifact)
    }

    fn verify_external_source(&self, model: &ModelEntry, source: &Path) -> Result<PathBuf> {
        let canonical = fs::canonicalize(source).map_err(|error| BackendError::CacheImport {
            source_path: source.to_path_buf(),
            destination: self.model_path(model),
            source: error,
        })?;
        let metadata = fs::metadata(&canonical)?;
        if !metadata.is_file() {
            return Err(BackendError::CacheSafety {
                path: source.to_path_buf(),
                message: "artifact source does not resolve to a regular file".to_owned(),
            });
        }
        self.verify_path(model, &canonical)?;
        Ok(canonical)
    }

    fn import_hard_link(&self, model: &ModelEntry, source: &Path, snapshot: &Path) -> Result<()> {
        let source = fs::canonicalize(source).map_err(|error| BackendError::CacheImport {
            source_path: source.to_path_buf(),
            destination: snapshot.to_path_buf(),
            source: error,
        })?;
        if !fs::metadata(&source)?.is_file() {
            return Err(BackendError::CacheSafety {
                path: source,
                message: "hard-link source is not a regular file".to_owned(),
            });
        }
        let hub_root = self.hub_root();
        let blob = self.blob_path(model);
        let blob_parent = blob.parent().ok_or_else(|| BackendError::CacheSafety {
            path: blob.clone(),
            message: "blob path has no parent".to_owned(),
        })?;
        self.create_contained_dir(&hub_root, blob_parent)?;
        self.require_mutation_parent(&blob, &hub_root)?;

        let blob_source = if path_present(&blob)? {
            let existing = self.canonical_owned_regular(&blob, &hub_root)?;
            if existing != source {
                self.verify_path(model, &existing)?;
            }
            existing
        } else {
            fs::hard_link(&source, &blob).map_err(|error| BackendError::CacheImport {
                source_path: source.clone(),
                destination: blob.clone(),
                source: error,
            })?;
            blob.clone()
        };

        let snapshot_parent = snapshot.parent().ok_or_else(|| BackendError::CacheSafety {
            path: snapshot.to_path_buf(),
            message: "snapshot path has no parent".to_owned(),
        })?;
        self.create_contained_dir(&hub_root, snapshot_parent)?;
        self.require_mutation_parent(snapshot, &hub_root)?;
        if path_present(snapshot)? {
            self.remove_controlled_entry(snapshot, &hub_root)?;
        }
        fs::hard_link(&blob_source, snapshot).map_err(|error| BackendError::CacheImport {
            source_path: blob_source,
            destination: snapshot.to_path_buf(),
            source: error,
        })?;
        Ok(())
    }

    fn quarantine_owned_artifact(
        &self,
        model: &ModelEntry,
        snapshot: &Path,
        blob: &Path,
    ) -> Result<()> {
        let hub_root = self.hub_root();
        let blob_present = path_present(blob)?;
        let snapshot_present = path_present(snapshot)?;
        if blob_present {
            self.require_mutation_parent(blob, &hub_root)?;
        }
        if snapshot_present {
            self.require_mutation_parent(snapshot, &hub_root)?;
        }

        let directory = self.root.join("openjev/quarantine");
        self.create_contained_dir(&self.root, &directory)?;
        let quarantine = directory.join(format!("{}-{}", model.sha256, std::process::id()));
        self.require_mutation_parent(&quarantine, &directory)?;
        if path_present(&quarantine)? {
            self.remove_controlled_entry(&quarantine, &directory)?;
        }

        let mut retained_bytes = false;
        if blob_present {
            let metadata = fs::symlink_metadata(blob)?;
            if metadata.file_type().is_symlink() {
                self.remove_controlled_entry(blob, &hub_root)?;
            } else if metadata.is_file() {
                fs::rename(blob, &quarantine)?;
                retained_bytes = true;
            } else {
                return Err(BackendError::CacheSafety {
                    path: blob.to_path_buf(),
                    message: "owned blob is neither a regular file nor a symlink".to_owned(),
                });
            }
        }

        if snapshot_present {
            let metadata = fs::symlink_metadata(snapshot)?;
            if metadata.file_type().is_symlink() || retained_bytes {
                self.remove_controlled_entry(snapshot, &hub_root)?;
            } else if metadata.is_file() {
                fs::rename(snapshot, &quarantine)?;
                retained_bytes = true;
            } else {
                return Err(BackendError::CacheSafety {
                    path: snapshot.to_path_buf(),
                    message: "owned snapshot is neither a regular file nor a symlink".to_owned(),
                });
            }
        }

        if !retained_bytes && path_present(&quarantine)? {
            self.remove_controlled_entry(&quarantine, &directory)?;
        }
        Ok(())
    }

    fn remove_controlled_entry(&self, path: &Path, boundary: &Path) -> Result<()> {
        if !self.mutation_parent_contained(path, boundary)? {
            return Ok(());
        }
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                Err(BackendError::CacheSafety {
                    path: path.to_path_buf(),
                    message: "refusing to remove a directory as a cache artifact".to_owned(),
                })
            }
            Ok(_) => {
                fs::remove_file(path)?;
                Ok(())
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn require_mutation_parent(&self, path: &Path, boundary: &Path) -> Result<()> {
        if self.mutation_parent_contained(path, boundary)? {
            Ok(())
        } else {
            Err(BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "cache mutation parent does not exist".to_owned(),
            })
        }
    }

    fn mutation_parent_contained(&self, path: &Path, boundary: &Path) -> Result<bool> {
        let boundary_relative =
            boundary
                .strip_prefix(&self.root)
                .map_err(|_| BackendError::CacheSafety {
                    path: boundary.to_path_buf(),
                    message: "mutation boundary is outside the cache root".to_owned(),
                })?;
        if boundary_relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(BackendError::CacheSafety {
                path: boundary.to_path_buf(),
                message: "mutation boundary contains a non-normal path component".to_owned(),
            });
        }

        let relative = path
            .strip_prefix(boundary)
            .map_err(|_| BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "mutation path is outside its logical ownership boundary".to_owned(),
            })?;
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "mutation path contains a non-normal path component".to_owned(),
            });
        }
        let parent = path.parent().ok_or_else(|| BackendError::CacheSafety {
            path: path.to_path_buf(),
            message: "mutation path has no parent".to_owned(),
        })?;
        let parent_relative =
            parent
                .strip_prefix(&self.root)
                .map_err(|_| BackendError::CacheSafety {
                    path: parent.to_path_buf(),
                    message: "mutation parent is outside the cache root".to_owned(),
                })?;

        let canonical_root =
            fs::canonicalize(&self.root).map_err(|error| BackendError::CacheSafety {
                path: self.root.clone(),
                message: format!("cache root cannot be resolved before mutation: {error}"),
            })?;
        if !fs::metadata(&canonical_root)?.is_dir() {
            return Err(BackendError::CacheSafety {
                path: self.root.clone(),
                message: "cache root is not a directory".to_owned(),
            });
        }

        let mut current = self.root.clone();
        let mut canonical_boundary = (boundary == self.root).then(|| canonical_root.clone());
        for component in parent_relative.components() {
            let Component::Normal(component) = component else {
                return Err(BackendError::CacheSafety {
                    path: parent.to_path_buf(),
                    message: "mutation parent contains a non-normal path component".to_owned(),
                });
            };
            current.push(component);
            match fs::symlink_metadata(&current) {
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error.into()),
            }
            let resolved =
                fs::canonicalize(&current).map_err(|error| BackendError::CacheSafety {
                    path: current.clone(),
                    message: format!("cache mutation parent cannot be resolved: {error}"),
                })?;
            if !resolved.starts_with(&canonical_root) || !fs::metadata(&resolved)?.is_dir() {
                return Err(BackendError::CacheSafety {
                    path: current.clone(),
                    message: "cache mutation parent escapes the canonical cache root".to_owned(),
                });
            }
            if current == boundary {
                canonical_boundary = Some(resolved.clone());
            }
            if let Some(canonical_boundary) = &canonical_boundary
                && !resolved.starts_with(canonical_boundary)
            {
                return Err(BackendError::CacheSafety {
                    path: current,
                    message: "cache mutation parent escapes its ownership boundary".to_owned(),
                });
            }
        }
        if canonical_boundary.is_none() {
            return Err(BackendError::CacheSafety {
                path: boundary.to_path_buf(),
                message: "mutation boundary was not reached from the cache root".to_owned(),
            });
        }
        Ok(true)
    }

    fn canonical_owned_regular(&self, path: &Path, boundary: &Path) -> Result<PathBuf> {
        if !path.starts_with(boundary) {
            return Err(BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "cache path is outside its logical ownership boundary".to_owned(),
            });
        }
        let canonical_boundary = fs::canonicalize(boundary)?;
        let canonical_root = fs::canonicalize(&self.root)?;
        if !canonical_boundary.starts_with(&canonical_root) {
            return Err(BackendError::CacheSafety {
                path: boundary.to_path_buf(),
                message: "cache ownership boundary resolves outside the cache root".to_owned(),
            });
        }
        let canonical = fs::canonicalize(path).map_err(|error| BackendError::CacheSafety {
            path: path.to_path_buf(),
            message: format!("cache artifact cannot be resolved: {error}"),
        })?;
        if !canonical.starts_with(&canonical_boundary) {
            return Err(BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "cache symlink resolves outside its ownership boundary".to_owned(),
            });
        }
        if !fs::metadata(&canonical)?.is_file() {
            return Err(BackendError::CacheSafety {
                path: path.to_path_buf(),
                message: "cache artifact does not resolve to a regular file".to_owned(),
            });
        }
        Ok(canonical)
    }

    fn create_contained_dir(&self, boundary: &Path, directory: &Path) -> Result<()> {
        if boundary != self.root && boundary.starts_with(&self.root) {
            self.create_contained_dir(&self.root, boundary)?;
        }
        let relative = directory
            .strip_prefix(boundary)
            .map_err(|_| BackendError::CacheSafety {
                path: directory.to_path_buf(),
                message: "directory is outside its logical ownership boundary".to_owned(),
            })?;
        fs::create_dir_all(boundary)?;
        let canonical_boundary = fs::canonicalize(boundary)?;
        let mut current = boundary.to_path_buf();
        for component in relative.components() {
            let Component::Normal(component) = component else {
                return Err(BackendError::CacheSafety {
                    path: directory.to_path_buf(),
                    message: "directory contains a non-normal path component".to_owned(),
                });
            };
            current.push(component);
            match fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.is_dir() || metadata.file_type().is_symlink() => {}
                Ok(_) => {
                    return Err(BackendError::CacheSafety {
                        path: current,
                        message: "cache directory component is not a directory".to_owned(),
                    });
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    if let Err(create_error) = fs::create_dir(&current)
                        && create_error.kind() != ErrorKind::AlreadyExists
                    {
                        return Err(create_error.into());
                    }
                }
                Err(error) => return Err(error.into()),
            }
            let resolved = fs::canonicalize(&current)?;
            if !resolved.starts_with(&canonical_boundary) || !fs::metadata(&resolved)?.is_dir() {
                return Err(BackendError::CacheSafety {
                    path: current,
                    message: "cache directory symlink escapes its ownership boundary".to_owned(),
                });
            }
        }
        Ok(())
    }

    fn receipt_path(&self, model: &ModelEntry) -> PathBuf {
        self.root
            .join("openjev/receipts")
            .join(format!("{}.json", model.sha256))
    }

    fn invalidate_receipt(&self, model: &ModelEntry) -> Result<()> {
        let directory = self.root.join("openjev/receipts");
        let destination = self.receipt_path(model);
        if path_present(&destination)? {
            self.remove_controlled_entry(&destination, &directory)?;
        }
        Ok(())
    }

    fn write_receipt(&self, model: &ModelEntry) -> Result<()> {
        let directory = self.root.join("openjev/receipts");
        self.create_contained_dir(&self.root, &directory)?;
        let destination = self.receipt_path(model);
        let temporary = directory.join(format!(".{}.{}.tmp", model.sha256, std::process::id()));
        self.require_mutation_parent(&temporary, &directory)?;
        self.require_mutation_parent(&destination, &directory)?;
        if path_present(&temporary)? {
            self.remove_controlled_entry(&temporary, &directory)?;
        }
        let receipt = Receipt {
            schema: "openjev-cache-receipt-v1".to_owned(),
            model_id: model.id.clone(),
            repo: model.repo.clone(),
            revision: model.revision.clone(),
            file: model.file.clone(),
            bytes: model.bytes,
            sha256: model.sha256.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|error| BackendError::Receipt(error.to_string()))?;
        {
            let mut file = File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        self.publish_receipt(&temporary, &destination, &directory)?;
        Ok(())
    }

    #[cfg(not(windows))]
    fn publish_receipt(
        &self,
        temporary: &Path,
        destination: &Path,
        directory: &Path,
    ) -> Result<()> {
        self.require_mutation_parent(temporary, directory)?;
        self.require_mutation_parent(destination, directory)?;
        fs::rename(temporary, destination)?;
        Ok(())
    }

    #[cfg(windows)]
    fn publish_receipt(
        &self,
        temporary: &Path,
        destination: &Path,
        directory: &Path,
    ) -> Result<()> {
        self.require_mutation_parent(temporary, directory)?;
        self.require_mutation_parent(destination, directory)?;
        if path_present(destination)? {
            self.remove_controlled_entry(destination, directory)?;
        }
        fs::rename(temporary, destination)?;
        Ok(())
    }
}

fn path_present(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub fn hash_file(path: &Path) -> Result<(u64, String)> {
    let file = File::open(path)?;
    let expected_capacity = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        bytes = bytes
            .checked_add(u64::try_from(count).expect("buffer length fits in u64"))
            .ok_or_else(|| BackendError::Receipt("artifact length overflow".to_owned()))?;
    }
    if bytes != expected_capacity {
        return Err(BackendError::ArtifactChanged {
            path: path.to_path_buf(),
            before: expected_capacity,
            after: bytes,
        });
    }
    Ok((bytes, digest_hex(&hasher.finalize())))
}

fn digest_hex(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn repo_folder(model: &ModelEntry) -> String {
    format!("models--{}", model.repo.replace('/', "--"))
}

#[cfg(test)]
mod tests {
    use std::{
        process::Command,
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use openjev_core::PromptProfile;
    use tempfile::TempDir;

    use super::*;
    use crate::{NativeReferenceSpec, TemplateEquivalenceSpec};

    fn tiny_model(bytes: &[u8]) -> ModelEntry {
        let sha256 = digest_hex(&Sha256::digest(bytes));
        ModelEntry {
            id: "tiny".to_owned(),
            repo: "test/tiny".to_owned(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            file: "tiny.gguf".to_owned(),
            bytes: u64::try_from(bytes.len()).unwrap(),
            sha256,
            quant: "test".to_owned(),
            profile: PromptProfile::Qwen3,
            native_reference: NativeReferenceSpec {
                source: "test/native".to_owned(),
                revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                dtype: "bfloat16".to_owned(),
                tokenizer_artifact: "tokenizer_config.json".to_owned(),
                tokenizer_sha256: "0".repeat(64),
                template_sha256: "1".repeat(64),
            },
            template_equivalence: None::<TemplateEquivalenceSpec>,
        }
    }

    fn write_fetch(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)?;
        Ok(path.to_path_buf())
    }

    fn external_snapshot(cache: &ModelCache, model: &ModelEntry) -> PathBuf {
        cache.default_huggingface_path(model).unwrap()
    }

    #[test]
    fn cache_root_precedence_is_explicit() {
        let explicit = Path::new("/explicit");
        assert_eq!(ModelCache::resolve_root(Some(explicit)).unwrap(), explicit);
    }

    #[test]
    fn malformed_paths_fail_before_filesystem_access() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("cache");
        let cache = ModelCache::new(root.clone());
        let outside = temp.path().join("outside-cache.txt");
        fs::write(&outside, b"untouched").unwrap();
        let mut model = tiny_model(b"untouched");
        model.file = outside.to_string_lossy().into_owned();
        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| panic!("invalid model must not fetch"),
            )
            .unwrap_err();
        assert!(matches!(error, BackendError::Manifest(_)));
        assert_eq!(fs::read(&outside).unwrap(), b"untouched");
        assert!(!root.exists());
    }

    #[test]
    fn offline_hit_verifies_and_offline_miss_never_fetches() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().to_path_buf());
        let contents = b"verified";
        let model = tiny_model(contents);
        let path = cache.model_path(&model);
        write_fetch(&path, contents).unwrap();
        let hit = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: false,
                },
                |_, _| panic!("offline hit must not fetch"),
            )
            .unwrap();
        assert!(hit.cache_hit);

        let mut missing = tiny_model(b"missing");
        missing.file = "missing.gguf".to_owned();
        let error = cache
            .ensure_with(
                &missing,
                CacheOptions {
                    offline: true,
                    repair: false,
                },
                |_, _| panic!("offline miss must not fetch"),
            )
            .unwrap_err();
        assert!(matches!(error, BackendError::OfflineMiss { .. }));
    }

    #[test]
    fn corrupt_hit_is_an_integrity_error_without_silent_fetch() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().to_path_buf());
        let model = tiny_model(b"correct");
        write_fetch(&cache.model_path(&model), b"corrupt").unwrap();
        let error = cache
            .ensure_with(&model, CacheOptions::default(), |_, _| {
                panic!("corrupt cache must not be silently repaired")
            })
            .unwrap_err();
        assert!(matches!(error, BackendError::Integrity { .. }));
        assert_eq!(fs::read(cache.model_path(&model)).unwrap(), b"corrupt");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_owned_hub_symlink_escape_before_fetch() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("openjev");
        let outside = temp.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("hub")).unwrap();
        let cache = ModelCache::new(root);
        let model = tiny_model(b"correct");
        let error = cache
            .ensure_with(&model, CacheOptions::default(), |_, _| {
                panic!("escaping hub must fail before fetch")
            })
            .unwrap_err();
        assert!(matches!(error, BackendError::CacheSafety { .. }));
        assert!(fs::read_dir(outside).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn repair_rejects_hub_parent_symlink_without_moving_outside_snapshot() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("openjev");
        let outside = temp.path().join("outside-hub");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("hub")).unwrap();
        let cache = ModelCache::with_external_hub(root, temp.path().join("external-hf"));
        let model = tiny_model(b"correct");
        let snapshot = cache.model_path(&model);
        write_fetch(&snapshot, b"outside sentinel").unwrap();
        let fetches = AtomicUsize::new(0);

        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| {
                    fetches.fetch_add(1, Ordering::SeqCst);
                    unreachable!("offline repair must not fetch")
                },
            )
            .unwrap_err();

        assert!(matches!(error, BackendError::CacheSafety { .. }));
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
        assert_eq!(fs::read(&snapshot).unwrap(), b"outside sentinel");
    }

    #[cfg(unix)]
    #[test]
    fn offline_miss_rejects_receipts_parent_symlink_without_deleting_outside_receipt() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("openjev");
        let outside = temp.path().join("outside-receipts");
        fs::create_dir_all(root.join("openjev")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("openjev/receipts")).unwrap();
        let cache = ModelCache::with_external_hub(root, temp.path().join("external-hf"));
        let model = tiny_model(b"correct");
        let outside_receipt = outside.join(format!("{}.json", model.sha256));
        fs::write(&outside_receipt, b"outside receipt sentinel").unwrap();
        let fetches = AtomicUsize::new(0);

        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: false,
                },
                |_, _| {
                    fetches.fetch_add(1, Ordering::SeqCst);
                    unreachable!("offline miss must not fetch")
                },
            )
            .unwrap_err();

        assert!(matches!(error, BackendError::CacheSafety { .. }));
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
        assert_eq!(
            fs::read(outside_receipt).unwrap(),
            b"outside receipt sentinel"
        );
    }

    #[cfg(unix)]
    #[test]
    fn repair_rejects_nested_blob_parent_symlink_without_moving_outside_blob() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("openjev");
        let outside = temp.path().join("outside-blobs");
        let cache = ModelCache::with_external_hub(root, temp.path().join("external-hf"));
        let model = tiny_model(b"correct");
        let repository = cache.hub_root().join(repo_folder(&model));
        fs::create_dir_all(&repository).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, repository.join("blobs")).unwrap();
        let outside_blob = outside.join(&model.sha256);
        fs::write(&outside_blob, b"outside blob sentinel").unwrap();
        let fetches = AtomicUsize::new(0);

        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| {
                    fetches.fetch_add(1, Ordering::SeqCst);
                    unreachable!("offline repair must not fetch")
                },
            )
            .unwrap_err();

        assert!(matches!(error, BackendError::CacheSafety { .. }));
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
        assert_eq!(fs::read(outside_blob).unwrap(), b"outside blob sentinel");
    }

    #[cfg(unix)]
    #[test]
    fn repair_of_owned_leaf_snapshot_symlink_retains_normal_behavior() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        let blob = cache.blob_path(&model);
        let snapshot = cache.model_path(&model);
        write_fetch(&blob, b"corrupt").unwrap();
        fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        symlink(Path::new("../../blobs").join(&model.sha256), &snapshot).unwrap();
        write_fetch(&external_snapshot(&cache, &model), b"correct").unwrap();

        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| panic!("valid external source must avoid fetch"),
            )
            .unwrap();

        assert!(artifact.cache_hit);
        assert!(artifact.imported_from_huggingface_cache);
        assert_eq!(fs::read(&blob).unwrap(), b"correct");
        assert_eq!(fs::read(&snapshot).unwrap(), b"correct");
        assert!(fs::symlink_metadata(&snapshot).unwrap().is_file());
    }

    #[cfg(unix)]
    #[test]
    fn imports_normal_hf_relative_snapshot_symlink_offline() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let external = temp.path().join("hf");
        let cache = ModelCache::with_external_hub(temp.path().join("openjev"), external.clone());
        let contents = b"hf blob";
        let model = tiny_model(contents);
        let source = external_snapshot(&cache, &model);
        let external_blob = external
            .join(repo_folder(&model))
            .join("blobs")
            .join(&model.sha256);
        write_fetch(&external_blob, contents).unwrap();
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        symlink(Path::new("../../blobs").join(&model.sha256), &source).unwrap();

        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: false,
                },
                |_, _| panic!("healthy external cache import must not fetch"),
            )
            .unwrap();
        assert!(artifact.cache_hit);
        assert!(artifact.imported_from_huggingface_cache);
        assert!(
            fs::symlink_metadata(cache.blob_path(&model))
                .unwrap()
                .file_type()
                .is_file()
        );
        fs::remove_file(source).unwrap();
        assert_eq!(fs::read(artifact.path).unwrap(), contents);
    }

    #[cfg(unix)]
    #[test]
    fn repair_replaces_dangling_owned_blob_symlink() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        let blob = cache.blob_path(&model);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        symlink("missing-target", &blob).unwrap();
        let external = external_snapshot(&cache, &model);
        write_fetch(&external, b"correct").unwrap();

        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| panic!("valid external source must avoid fetch"),
            )
            .unwrap();
        assert_eq!(fs::read(artifact.path).unwrap(), b"correct");
        assert!(fs::symlink_metadata(blob).unwrap().is_file());
    }

    #[test]
    fn repair_bypasses_corrupt_external_source_and_fetches() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        let source = external_snapshot(&cache, &model);
        write_fetch(&source, b"corrupt external").unwrap();
        let external_before = fs::read(&source).unwrap();
        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: false,
                    repair: true,
                },
                |expected, _| write_fetch(&cache.model_path(expected), b"correct"),
            )
            .unwrap();
        assert!(!artifact.cache_hit);
        assert_eq!(fs::read(source).unwrap(), external_before);
        assert_eq!(fs::read(artifact.path).unwrap(), b"correct");
    }

    #[test]
    fn offline_repair_bypasses_corrupt_external_and_returns_miss() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        let source = external_snapshot(&cache, &model);
        write_fetch(&source, b"corrupt external").unwrap();
        let before = fs::read(&source).unwrap();
        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| panic!("offline repair must not fetch"),
            )
            .unwrap_err();
        assert!(matches!(error, BackendError::OfflineMiss { .. }));
        assert_eq!(fs::read(source).unwrap(), before);
    }

    #[test]
    fn repair_replaces_owned_corrupt_blob_from_valid_external_source() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        let blob = cache.blob_path(&model);
        let snapshot = cache.model_path(&model);
        write_fetch(&blob, b"corrupt").unwrap();
        if let Some(parent) = snapshot.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::hard_link(&blob, &snapshot).unwrap();
        let external = external_snapshot(&cache, &model);
        write_fetch(&external, b"correct").unwrap();

        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: true,
                },
                |_, _| panic!("valid alternate must avoid fetch"),
            )
            .unwrap();
        assert!(artifact.cache_hit);
        assert!(artifact.imported_from_huggingface_cache);
        assert_eq!(fs::read(&blob).unwrap(), b"correct");
        assert_eq!(fs::read(&snapshot).unwrap(), b"correct");
        let quarantine = cache.root.join("openjev/quarantine");
        assert!(
            fs::read_dir(quarantine)
                .unwrap()
                .any(|entry| fs::read(entry.unwrap().path()).unwrap() == b"corrupt")
        );
    }

    #[test]
    fn failed_repair_leaves_no_verified_receipt() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::with_external_hub(
            temp.path().join("openjev"),
            temp.path().join("external-hf"),
        );
        let model = tiny_model(b"correct");
        write_fetch(&cache.model_path(&model), b"corrupt").unwrap();
        let receipt = cache.receipt_path(&model);
        write_fetch(&receipt, b"stale receipt").unwrap();
        let error = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: false,
                    repair: true,
                },
                |_, _| Err(BackendError::Download("interrupted".to_owned())),
            )
            .unwrap_err();
        assert!(matches!(error, BackendError::Download(_)));
        assert!(!path_present(&receipt).unwrap());
        assert!(!path_present(&cache.model_path(&model)).unwrap());
    }

    #[test]
    fn healthy_cache_never_fetches_even_with_repair_enabled() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().to_path_buf());
        let model = tiny_model(b"healthy");
        write_fetch(&cache.model_path(&model), b"healthy").unwrap();
        let artifact = cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: false,
                    repair: true,
                },
                |_, _| panic!("healthy cache must not fetch"),
            )
            .unwrap();
        assert!(artifact.cache_hit);
    }

    #[test]
    fn concurrent_threads_reuse_one_mock_download() {
        let temp = TempDir::new().unwrap();
        let cache = Arc::new(ModelCache::new(temp.path().to_path_buf()));
        let model = Arc::new(tiny_model(b"one download"));
        let starts = Arc::new(Barrier::new(3));
        let fetches = Arc::new(AtomicUsize::new(0));
        let mut joins = Vec::new();
        for _ in 0..2 {
            let cache = Arc::clone(&cache);
            let model = Arc::clone(&model);
            let starts = Arc::clone(&starts);
            let fetches = Arc::clone(&fetches);
            joins.push(std::thread::spawn(move || {
                starts.wait();
                cache
                    .ensure_with(&model, CacheOptions::default(), |expected, _| {
                        fetches.fetch_add(1, Ordering::SeqCst);
                        write_fetch(&cache.model_path(expected), b"one download")
                    })
                    .unwrap()
            }));
        }
        starts.wait();
        let results: Vec<_> = joins.into_iter().map(|join| join.join().unwrap()).collect();
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert_eq!(results.iter().filter(|result| result.cache_hit).count(), 1);
        assert_eq!(results.iter().filter(|result| !result.cache_hit).count(), 1);
    }

    #[test]
    fn concurrent_processes_fetch_once_and_reuse_verified_path() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("cache");
        let marker = temp.path().join("fetch-count");
        let executable = std::env::current_exe().unwrap();
        let spawn = || {
            Command::new(&executable)
                .args([
                    "--exact",
                    "cache::tests::cross_process_cache_worker",
                    "--nocapture",
                ])
                .env("OPENJEV_CACHE_PROCESS_ROOT", &root)
                .env("OPENJEV_CACHE_PROCESS_MARKER", &marker)
                .spawn()
                .unwrap()
        };
        let mut first = spawn();
        let mut second = spawn();
        assert!(first.wait().unwrap().success());
        assert!(second.wait().unwrap().success());
        assert_eq!(fs::read(&marker).unwrap(), b"fetch\n");
        let cache = ModelCache::new(root);
        let model = tiny_model(b"cross process");
        cache
            .ensure_with(
                &model,
                CacheOptions {
                    offline: true,
                    repair: false,
                },
                |_, _| panic!("parent verification must reuse the artifact"),
            )
            .unwrap();
    }

    #[test]
    fn cross_process_cache_worker() {
        let Some(root) = std::env::var_os("OPENJEV_CACHE_PROCESS_ROOT") else {
            return;
        };
        let marker =
            PathBuf::from(std::env::var_os("OPENJEV_CACHE_PROCESS_MARKER").expect("worker marker"));
        let cache = ModelCache::new(PathBuf::from(root));
        let model = tiny_model(b"cross process");
        cache
            .ensure_with(&model, CacheOptions::default(), |expected, _| {
                let mut count = OpenOptions::new().create(true).append(true).open(&marker)?;
                count.write_all(b"fetch\n")?;
                count.sync_all()?;
                std::thread::sleep(Duration::from_millis(200));
                write_fetch(&cache.model_path(expected), b"cross process")
            })
            .unwrap();
    }
}
