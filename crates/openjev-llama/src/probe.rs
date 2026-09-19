use std::{
    fmt::Write as _,
    fs::{self, File},
    io::Write as _,
    path::PathBuf,
};

use openjev_core::{Device, GpuLayersRequested, PromptProfile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{BackendError, ModelCache, Result};

pub const PROBE_SCHEMA: &str = "openjev-probe-receipt-v1";
pub const PROBE_SUITE_VERSION: &str = "openjev-m5-probe-suite-v1";
pub const NATIVE_PIN: &str =
    "llama-cpp-2/0.1.156 llama.cpp/e79e4bf660e19f2ad851e06c6913f7a8c5852621";
pub const MAX_ABS_SLOT_LOGIT: f64 = 1e-3;
pub const MAX_PROBABILITY_DELTA: f64 = 1e-4;

const PROBE_CASE_IDS: [&str; 5] = [
    "binary-short-1-branch",
    "three-way-ragged-2-branches-multichunk",
    "sixteen-way-long-state-21-branches",
    "changed-state-isolation",
    "repeated-copy-clear-cycles",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProbeMode {
    Shared,
    Batch,
}

impl ProbeMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Batch => "batch",
        }
    }
}

/// An in-process capability created only after a passing exact receipt has
/// been loaded and validated from the selected local cache.
#[derive(Clone, Debug)]
pub struct ProbeEligibility {
    receipt: ProbeReceipt,
}

impl ProbeEligibility {
    #[must_use]
    pub fn probe_id(&self) -> &str {
        &self.receipt.probe_id
    }

    #[must_use]
    pub fn mode(&self) -> ProbeMode {
        self.receipt.mode
    }
}

/// Parent-owned fail-closed authorization transition for one exact probe key.
///
/// Construction publishes a durable suspension marker before the probe child
/// is launched and holds the exact-key process lock until this value is
/// dropped. Dropping without successfully publishing a passing exact receipt
/// intentionally leaves the previous authorization suspended.
#[derive(Debug)]
pub struct ProbePublication {
    cache: ModelCache,
    model_id: String,
    mode: ProbeMode,
    configuration: ProbeConfiguration,
    probe_id: String,
    suspension_path: PathBuf,
    _lock: File,
}

impl ProbePublication {
    #[must_use]
    pub fn probe_id(&self) -> &str {
        &self.probe_id
    }

    /// Publish a fully validated exact child result while the suspension and
    /// process lock are still held. A failed receipt may be retained as a
    /// diagnostic record, but only a passing receipt clears suspension and
    /// becomes eligible.
    pub fn publish_child_result(
        &self,
        receipt: &ProbeReceipt,
        child_exited_successfully: bool,
    ) -> Result<PathBuf> {
        receipt.validate()?;
        if receipt.passed && !child_exited_successfully {
            return Err(BackendError::Receipt(
                "a passing probe candidate from a nonzero child cannot be published".to_owned(),
            ));
        }
        if receipt.model_id != self.model_id
            || receipt.mode != self.mode
            || receipt.configuration != self.configuration
            || receipt.probe_id != self.probe_id
        {
            return Err(BackendError::Receipt(
                "probe child receipt does not match the parent-established model/configuration"
                    .to_owned(),
            ));
        }
        let path = write_receipt(&self.cache, receipt)?;
        if receipt.passed {
            let directory = safe_probe_directory(&self.cache)?;
            self.cache
                .remove_controlled_entry(&self.suspension_path, &directory)?;
        }
        Ok(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeConfiguration {
    pub artifact_sha256: String,
    pub native_pin: String,
    pub probe_suite_version: String,
    pub device: Device,
    pub gpu_layers_requested: GpuLayersRequested,
    pub offload_kqv: bool,
    pub op_offload: bool,
    pub threads: u32,
    pub n_ctx: Option<u32>,
    pub max_tokens: u32,
    pub max_context_tokens: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub kv_unified: bool,
    pub profile: PromptProfile,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeCaseResult {
    pub id: String,
    pub status: ProbeCaseStatus,
    pub rows: u32,
    pub max_abs_slot_logit: Option<f64>,
    pub max_probability_delta: Option<f64>,
    pub same_first_argmax: Option<bool>,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProbeCaseStatus {
    Passed,
    Failed,
    UnrunAfterDecisiveFailure,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeReceipt {
    pub schema: String,
    pub probe_id: String,
    pub model_id: String,
    pub mode: ProbeMode,
    pub configuration: ProbeConfiguration,
    pub passed: bool,
    pub max_abs_slot_logit: Option<f64>,
    pub max_probability_delta: Option<f64>,
    pub same_first_argmax: bool,
    pub cases: Vec<ProbeCaseResult>,
    pub failure_reason: Option<String>,
}

impl ProbeReceipt {
    pub fn new(
        model_id: String,
        mode: ProbeMode,
        configuration: ProbeConfiguration,
        cases: Vec<ProbeCaseResult>,
        failure_reason: Option<String>,
    ) -> Result<Self> {
        let mut max_logit: Option<f64> = None;
        let mut max_probability: Option<f64> = None;
        let mut same_argmax = true;
        for case in &cases {
            if let Some(value) = case.max_abs_slot_logit {
                max_logit = Some(max_logit.map_or(value, |current| current.max(value)));
            }
            if let Some(value) = case.max_probability_delta {
                max_probability = Some(max_probability.map_or(value, |current| current.max(value)));
            }
            if case.same_first_argmax == Some(false) {
                same_argmax = false;
            }
        }
        let all_cases_passed = !cases.is_empty()
            && cases
                .iter()
                .all(|case| case.status == ProbeCaseStatus::Passed);
        let passed = all_cases_passed
            && failure_reason.is_none()
            && max_logit.is_some_and(|value| value <= MAX_ABS_SLOT_LOGIT)
            && max_probability.is_some_and(|value| value <= MAX_PROBABILITY_DELTA)
            && same_argmax;
        let probe_id = probe_id(mode, &configuration)?;
        let receipt = Self {
            schema: PROBE_SCHEMA.to_owned(),
            probe_id,
            model_id,
            mode,
            configuration,
            passed,
            max_abs_slot_logit: max_logit,
            max_probability_delta: max_probability,
            same_first_argmax: same_argmax,
            cases,
            failure_reason,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != PROBE_SCHEMA || self.model_id.is_empty() {
            return Err(BackendError::Receipt(
                "probe receipt schema or model identity is invalid".to_owned(),
            ));
        }
        validate_configuration(&self.configuration)?;
        let expected = probe_id(self.mode, &self.configuration)?;
        if self.probe_id != expected {
            return Err(BackendError::Receipt(
                "probe receipt ID does not match its exact configuration payload".to_owned(),
            ));
        }
        if self.cases.len() != PROBE_CASE_IDS.len() {
            return Err(BackendError::Receipt(
                "probe receipt is incomplete for its declared suite version".to_owned(),
            ));
        }
        let expected_rows = [1, 2, 21, 2, self.configuration.n_seq_max + 1];
        let mut saw_failure = false;
        for ((case, expected_id), expected_rows) in
            self.cases.iter().zip(PROBE_CASE_IDS).zip(expected_rows)
        {
            if case.id != expected_id {
                return Err(BackendError::Receipt(
                    "probe receipt case identity/order does not match its declared suite version"
                        .to_owned(),
                ));
            }
            for value in [case.max_abs_slot_logit, case.max_probability_delta]
                .into_iter()
                .flatten()
            {
                if !value.is_finite() || value < 0.0 {
                    return Err(BackendError::Receipt(
                        "probe receipt contains invalid numerical deltas".to_owned(),
                    ));
                }
            }
            let complete_measurements = case.max_abs_slot_logit.is_some()
                && case.max_probability_delta.is_some()
                && case.same_first_argmax.is_some();
            let no_measurements = case.max_abs_slot_logit.is_none()
                && case.max_probability_delta.is_none()
                && case.same_first_argmax.is_none();
            match case.status {
                ProbeCaseStatus::Passed => {
                    if saw_failure || case.rows != expected_rows || !complete_measurements {
                        return Err(BackendError::Receipt(
                            "passed probe case lacks the complete ordered suite evidence"
                                .to_owned(),
                        ));
                    }
                }
                ProbeCaseStatus::Failed => {
                    saw_failure = true;
                    if case.detail.as_ref().is_none_or(String::is_empty)
                        || !((case.rows == expected_rows && complete_measurements)
                            || (case.rows == 0 && no_measurements))
                    {
                        return Err(BackendError::Receipt(
                            "failed probe case lacks a reason or has partial evidence".to_owned(),
                        ));
                    }
                }
                ProbeCaseStatus::UnrunAfterDecisiveFailure => {
                    if !saw_failure
                        || case.rows != 0
                        || !no_measurements
                        || case.detail.as_ref().is_none_or(String::is_empty)
                    {
                        return Err(BackendError::Receipt(
                            "unrun probe case must follow a decisive failure and contain no measurements"
                                .to_owned(),
                        ));
                    }
                }
            }
        }
        if self.passed {
            if saw_failure || self.failure_reason.is_some() {
                return Err(BackendError::Receipt(
                    "passing probe receipt cannot contain a failure".to_owned(),
                ));
            }
        } else if !saw_failure || self.failure_reason.as_ref().is_none_or(String::is_empty) {
            return Err(BackendError::Receipt(
                "failed probe receipt requires a decisive case and summary reason".to_owned(),
            ));
        }
        let recomputed = Self::new_unchecked_summary(&self.cases, self.failure_reason.as_deref());
        if self.max_abs_slot_logit != recomputed.0
            || self.max_probability_delta != recomputed.1
            || self.same_first_argmax != recomputed.2
            || self.passed != recomputed.3
        {
            return Err(BackendError::Receipt(
                "probe receipt summary is inconsistent with its cases and frozen tolerances"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn new_unchecked_summary(
        cases: &[ProbeCaseResult],
        failure_reason: Option<&str>,
    ) -> (Option<f64>, Option<f64>, bool, bool) {
        let max_logit = cases
            .iter()
            .filter_map(|case| case.max_abs_slot_logit)
            .reduce(f64::max);
        let max_probability = cases
            .iter()
            .filter_map(|case| case.max_probability_delta)
            .reduce(f64::max);
        let same_argmax = cases
            .iter()
            .all(|case| case.same_first_argmax != Some(false));
        let passed = !cases.is_empty()
            && cases
                .iter()
                .all(|case| case.status == ProbeCaseStatus::Passed)
            && failure_reason.is_none()
            && max_logit.is_some_and(|value| value <= MAX_ABS_SLOT_LOGIT)
            && max_probability.is_some_and(|value| value <= MAX_PROBABILITY_DELTA)
            && same_argmax;
        (max_logit, max_probability, same_argmax, passed)
    }
}

pub fn probe_id(mode: ProbeMode, configuration: &ProbeConfiguration) -> Result<String> {
    validate_configuration(configuration)?;
    #[derive(Serialize)]
    struct Identity<'a> {
        schema: &'static str,
        mode: ProbeMode,
        configuration: &'a ProbeConfiguration,
    }
    let bytes = serde_json::to_vec(&Identity {
        schema: PROBE_SCHEMA,
        mode,
        configuration,
    })
    .map_err(|error| BackendError::Receipt(error.to_string()))?;
    Ok(hex_digest(&Sha256::digest(bytes)))
}

pub fn begin_probe_publication(
    cache: &ModelCache,
    model_id: &str,
    mode: ProbeMode,
    configuration: &ProbeConfiguration,
) -> Result<ProbePublication> {
    if model_id.is_empty() {
        return Err(BackendError::Receipt(
            "probe parent model identity must not be empty".to_owned(),
        ));
    }
    let expected = probe_id(mode, configuration)?;
    let lock = cache.lock_sha256(&expected)?;
    let directory = safe_probe_directory(cache)?;
    let suspension_path = suspension_path(cache, &expected);
    publish_suspension(cache, &directory, &suspension_path, &expected)?;
    Ok(ProbePublication {
        cache: cache.clone(),
        model_id: model_id.to_owned(),
        mode,
        configuration: configuration.clone(),
        probe_id: expected,
        suspension_path,
        _lock: lock,
    })
}

pub fn load_passing_receipt(
    cache: &ModelCache,
    model_id: &str,
    mode: ProbeMode,
    configuration: &ProbeConfiguration,
) -> Result<ProbeEligibility> {
    let expected = probe_id(mode, configuration)?;
    reject_suspended(cache, &expected)?;
    let path = receipt_path(cache, &expected);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        BackendError::Receipt(format!("inspect probe receipt {}: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BackendError::CacheSafety {
            path,
            message: "probe receipt must be an owned regular file, not a symlink".to_owned(),
        });
    }
    let canonical_path = fs::canonicalize(&path)?;
    let canonical_directory = fs::canonicalize(safe_probe_directory(cache)?)?;
    if !canonical_path.starts_with(&canonical_directory) {
        return Err(BackendError::CacheSafety {
            path,
            message: "probe receipt resolves outside the selected cache root".to_owned(),
        });
    }
    let bytes = fs::read(canonical_path)?;
    let receipt: ProbeReceipt = serde_json::from_slice(&bytes)
        .map_err(|error| BackendError::Receipt(format!("parse {}: {error}", path.display())))?;
    receipt.validate()?;
    if receipt.model_id != model_id
        || receipt.mode != mode
        || receipt.configuration != *configuration
        || receipt.probe_id != expected
        || !receipt.passed
    {
        return Err(BackendError::Receipt(
            "probe receipt is not a passing exact model/configuration match".to_owned(),
        ));
    }
    // Close the normal race in which a reprobe starts after the first marker
    // check but before this receipt has been completely validated.
    reject_suspended(cache, &expected)?;
    Ok(ProbeEligibility { receipt })
}

pub fn write_receipt(cache: &ModelCache, receipt: &ProbeReceipt) -> Result<PathBuf> {
    receipt.validate()?;
    let directory = safe_probe_directory(cache)?;
    let path = directory.join(format!("{}.json", receipt.probe_id));
    let temporary = directory.join(format!(".{}.tmp-{}", receipt.probe_id, std::process::id()));
    let bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| BackendError::Receipt(error.to_string()))?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| BackendError::Receipt(error.to_string()))?;
    let result = (|| {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, &path)?;
        Ok::<(), std::io::Error>(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(BackendError::Receipt(error.to_string()));
    }
    Ok(path)
}

#[must_use]
pub fn receipt_path(cache: &ModelCache, probe_id: &str) -> PathBuf {
    cache
        .root()
        .join("openjev")
        .join("probes")
        .join(format!("{probe_id}.json"))
}

#[must_use]
fn suspension_path(cache: &ModelCache, probe_id: &str) -> PathBuf {
    cache
        .root()
        .join("openjev")
        .join("probes")
        .join(format!("{probe_id}.suspended"))
}

fn reject_suspended(cache: &ModelCache, probe_id: &str) -> Result<()> {
    let path = suspension_path(cache, probe_id);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(BackendError::Receipt(format!(
            "probe authorization {probe_id} is suspended pending a successful exact reprobe"
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn publish_suspension(
    cache: &ModelCache,
    directory: &std::path::Path,
    destination: &std::path::Path,
    probe_id: &str,
) -> Result<()> {
    match fs::symlink_metadata(destination) {
        Ok(_) => {
            // A suspension left by an earlier failed/crashed reprobe is already
            // the desired fail-closed state. Do not introduce a replacement gap.
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let temporary = directory.join(format!(".{probe_id}.suspend-{}", std::process::id()));
    cache.remove_controlled_entry(&temporary, directory)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| BackendError::Receipt(error.to_string()))?;
    let result = (|| {
        file.write_all(b"openjev-probe-authorization-suspended-v1\n")?;
        file.sync_all()?;
        fs::rename(&temporary, destination)?;
        Ok::<(), std::io::Error>(())
    })();
    if let Err(error) = result {
        let _ = cache.remove_controlled_entry(&temporary, directory);
        return Err(BackendError::Receipt(error.to_string()));
    }
    Ok(())
}

fn validate_configuration(configuration: &ProbeConfiguration) -> Result<()> {
    if configuration.artifact_sha256.len() != 64
        || !configuration
            .artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || configuration.native_pin != NATIVE_PIN
        || configuration.probe_suite_version != PROBE_SUITE_VERSION
        || configuration.threads == 0
        || configuration.max_tokens == 0
        || configuration.max_context_tokens == 0
        || configuration.n_batch == 0
        || configuration.n_ubatch == 0
        || configuration.n_seq_max == 0
        || configuration.n_seq_max > 64
        || configuration.n_ubatch > configuration.n_batch
        || !configuration.kv_unified
        || (configuration.device == Device::Cpu
            && (configuration.gpu_layers_requested != GpuLayersRequested::Count(0)
                || configuration.offload_kqv
                || configuration.op_offload))
        || (configuration.device != Device::Cpu
            && (!configuration.offload_kqv || !configuration.op_offload))
    {
        return Err(BackendError::Receipt(
            "invalid or incomplete exact probe configuration".to_owned(),
        ));
    }
    Ok(())
}

fn safe_probe_directory(cache: &ModelCache) -> Result<PathBuf> {
    let probes = cache.root().join("openjev/probes");
    cache.create_contained_dir(cache.root(), &probes)?;
    Ok(probes)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn configuration() -> ProbeConfiguration {
        ProbeConfiguration {
            artifact_sha256: "a".repeat(64),
            native_pin: NATIVE_PIN.to_owned(),
            probe_suite_version: PROBE_SUITE_VERSION.to_owned(),
            device: Device::Cpu,
            gpu_layers_requested: GpuLayersRequested::Count(0),
            offload_kqv: false,
            op_offload: false,
            threads: 4,
            n_ctx: Some(4096),
            max_tokens: 4096,
            max_context_tokens: 32_768,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 32,
            kv_unified: true,
            profile: PromptProfile::Qwen3,
        }
    }

    fn passing_cases(configuration: &ProbeConfiguration) -> Vec<ProbeCaseResult> {
        [1, 2, 21, 2, configuration.n_seq_max + 1]
            .into_iter()
            .zip(PROBE_CASE_IDS)
            .map(|(rows, id)| ProbeCaseResult {
                id: id.to_owned(),
                status: ProbeCaseStatus::Passed,
                rows,
                max_abs_slot_logit: Some(0.0),
                max_probability_delta: Some(0.0),
                same_first_argmax: Some(true),
                detail: None,
            })
            .collect()
    }

    fn passing_receipt() -> ProbeReceipt {
        let configuration = configuration();
        ProbeReceipt::new(
            "model".to_owned(),
            ProbeMode::Shared,
            configuration.clone(),
            passing_cases(&configuration),
            None,
        )
        .unwrap()
    }

    fn failed_receipt() -> ProbeReceipt {
        let configuration = configuration();
        let mut cases = passing_cases(&configuration);
        cases[0].status = ProbeCaseStatus::Failed;
        cases[0].max_abs_slot_logit = Some(MAX_ABS_SLOT_LOGIT * 2.0);
        cases[0].detail = Some("synthetic lifecycle failure".to_owned());
        for case in &mut cases[1..] {
            case.status = ProbeCaseStatus::UnrunAfterDecisiveFailure;
            case.rows = 0;
            case.max_abs_slot_logit = None;
            case.max_probability_delta = None;
            case.same_first_argmax = None;
            case.detail = Some("not run after decisive failure".to_owned());
        }
        ProbeReceipt::new(
            "model".to_owned(),
            ProbeMode::Shared,
            configuration,
            cases,
            Some("synthetic lifecycle failure".to_owned()),
        )
        .unwrap()
    }

    #[test]
    fn receipt_identity_covers_every_configuration_field() {
        let base = configuration();
        let id = probe_id(ProbeMode::Shared, &base).unwrap();
        let mut changed = base.clone();
        changed.threads += 1;
        assert_ne!(id, probe_id(ProbeMode::Shared, &changed).unwrap());
        assert_ne!(id, probe_id(ProbeMode::Batch, &base).unwrap());
    }

    #[test]
    fn malformed_or_forged_receipts_are_rejected_but_consistent_local_receipts_load() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let receipt = passing_receipt();
        write_receipt(&cache, &receipt).unwrap();
        load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();

        let path = receipt_path(&cache, &receipt.probe_id);
        let mut forged = receipt;
        forged.probe_id = "0".repeat(64);
        fs::write(&path, serde_json::to_vec(&forged).unwrap()).unwrap();
        assert!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn receipt_leaf_symlink_is_never_trusted() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let receipt = passing_receipt();
        let directory = safe_probe_directory(&cache).unwrap();
        let outside = temp.path().join("outside.json");
        fs::write(&outside, serde_json::to_vec(&receipt).unwrap()).unwrap();
        symlink(
            &outside,
            directory.join(format!("{}.json", receipt.probe_id)),
        )
        .unwrap();

        assert!(matches!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()),
            Err(BackendError::CacheSafety { .. })
        ));
    }

    #[test]
    fn reprobe_suspends_preexisting_pass_until_an_exact_new_pass_is_published() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let receipt = passing_receipt();
        write_receipt(&cache, &receipt).unwrap();
        load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();

        let publication =
            begin_probe_publication(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();
        assert!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).is_err(),
            "the old passing receipt must be ineligible before child launch"
        );
        drop(publication);
        assert!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).is_err(),
            "crash, malformed output, or nonpublication must never restore the old pass"
        );

        let publication =
            begin_probe_publication(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();
        publication.publish_child_result(&receipt, true).unwrap();
        drop(publication);
        load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();
    }

    #[test]
    fn failed_child_record_remains_ineligible_and_parent_identity_is_authoritative() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let passing = passing_receipt();
        write_receipt(&cache, &passing).unwrap();

        let publication =
            begin_probe_publication(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();
        let mut wrong_model = passing.clone();
        wrong_model.model_id = "child-controlled-model".to_owned();
        assert!(
            publication
                .publish_child_result(&wrong_model, true)
                .is_err()
        );
        assert!(
            publication.publish_child_result(&passing, false).is_err(),
            "a nonzero child cannot publish a passing candidate"
        );
        assert!(
            publication
                .publish_child_result(&failed_receipt(), false)
                .is_ok()
        );
        drop(publication);
        assert!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).is_err()
        );
    }

    #[test]
    fn publication_failure_never_reactivates_preexisting_pass() {
        let temp = TempDir::new().unwrap();
        let cache = ModelCache::new(temp.path().join("cache"));
        let receipt = passing_receipt();
        write_receipt(&cache, &receipt).unwrap();
        let publication =
            begin_probe_publication(&cache, "model", ProbeMode::Shared, &configuration()).unwrap();

        let path = receipt_path(&cache, &receipt.probe_id);
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(publication.publish_child_result(&receipt, true).is_err());
        drop(publication);
        assert!(
            load_passing_receipt(&cache, "model", ProbeMode::Shared, &configuration()).is_err()
        );
    }

    #[test]
    fn frozen_tolerances_cannot_be_relaxed_by_receipt_data() {
        let configuration = configuration();
        let mut cases = passing_cases(&configuration);
        cases[0].status = ProbeCaseStatus::Failed;
        cases[0].max_abs_slot_logit = Some(MAX_ABS_SLOT_LOGIT * 1.01);
        cases[0].detail = Some("frozen tolerance failure".to_owned());
        for case in &mut cases[1..] {
            case.status = ProbeCaseStatus::UnrunAfterDecisiveFailure;
            case.rows = 0;
            case.max_abs_slot_logit = None;
            case.max_probability_delta = None;
            case.same_first_argmax = None;
            case.detail = Some("not run after decisive failure".to_owned());
        }
        let receipt = ProbeReceipt::new(
            "model".to_owned(),
            ProbeMode::Shared,
            configuration,
            cases,
            Some("frozen tolerance failure".to_owned()),
        )
        .unwrap();
        assert!(!receipt.passed);
    }

    #[test]
    fn incomplete_or_reordered_suite_cannot_activate_a_receipt() {
        let mut receipt = passing_receipt();
        receipt.cases.pop();
        assert!(receipt.validate().is_err());

        let mut receipt = passing_receipt();
        receipt.cases.swap(0, 1);
        assert!(receipt.validate().is_err());
    }
}
