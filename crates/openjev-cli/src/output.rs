use std::{fs::File, fs::OpenOptions, io::Write, path::Path};

use openjev_core::{ExecutionMode, Readout};
use serde::Serialize;

use crate::CliError;

#[derive(Debug, Serialize)]
struct CompactExecution<'a> {
    requested_mode: ExecutionMode,
    effective_mode: ExecutionMode,
    fallback_reason: &'a str,
}

/// Stable, low-clutter projection for decision consumers.
///
/// This is intentionally a separate schema from the full Readout. It retains
/// semantic option alignment and the exact probability honesty label while
/// excluding model, token, prompt, timing, and raw-logit diagnostics.
#[derive(Debug, Serialize)]
struct CompactReadout<'a> {
    schema: &'static str,
    id: &'a str,
    choice: &'a str,
    option_ids: &'a [String],
    probabilities: &'a [f64],
    probability_status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    p_yes: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    argmax_level: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level_values: Option<&'a [f64]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence_status: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    execution: Option<CompactExecution<'a>>,
}

impl<'a> From<&'a Readout> for CompactReadout<'a> {
    fn from(row: &'a Readout) -> Self {
        let execution = row
            .execution
            .fallback_reason
            .as_deref()
            .map(|reason| CompactExecution {
                requested_mode: row.execution.requested_mode,
                effective_mode: row.execution.effective_mode,
                fallback_reason: reason,
            });
        Self {
            schema: "openjev-compact-v1",
            id: &row.id,
            choice: &row.choice,
            option_ids: &row.option_ids,
            probabilities: &row.probabilities,
            probability_status: &row.probability_status,
            p_yes: row.p_yes,
            expected_value: row.expected_value,
            argmax_level: row.argmax_level.as_deref(),
            level_values: row.level_values.as_deref(),
            confidence: row.confidence,
            confidence_status: row.confidence_status.as_deref(),
            execution,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HelpOutput {
    pub schema: &'static str,
    pub command: String,
    pub usage: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct VersionOutput {
    pub schema: &'static str,
    pub version: &'static str,
    pub build: &'static str,
}

#[derive(Debug, Serialize)]
pub struct WriteSummary {
    pub schema: &'static str,
    pub path: String,
    pub written: usize,
    pub failed: usize,
}

pub fn write_json<W: Write + ?Sized, T: Serialize>(
    writer: &mut W,
    value: &T,
    pretty: bool,
) -> std::io::Result<()> {
    if pretty {
        serde_json::to_writer_pretty(&mut *writer, value)
    } else {
        serde_json::to_writer(&mut *writer, value)
    }
    .map_err(std::io::Error::other)?;
    writer.write_all(b"\n")
}

pub fn write_jsonl<W: Write + ?Sized, T: Serialize>(
    writer: &mut W,
    rows: &[T],
) -> std::io::Result<()> {
    for row in rows {
        write_json(writer, row, false)?;
    }
    Ok(())
}

pub fn write_readout<W: Write + ?Sized>(
    writer: &mut W,
    row: &Readout,
    pretty: bool,
    compact: bool,
) -> std::io::Result<()> {
    if compact {
        write_json(writer, &CompactReadout::from(row), pretty)
    } else {
        write_json(writer, row, pretty)
    }
}

pub fn write_readout_jsonl<W: Write + ?Sized>(
    writer: &mut W,
    rows: &[Readout],
    compact: bool,
) -> std::io::Result<()> {
    for row in rows {
        write_readout(writer, row, false, compact)?;
    }
    Ok(())
}

pub fn create_jsonl_new(path: &Path) -> Result<File, CliError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            CliError::runtime(
                "output_io",
                format!("create-only output {}: {error}", path.display()),
            )
        })
}

pub fn sync_jsonl(file: &File, path: &Path) -> Result<(), CliError> {
    file.sync_all().map_err(|error| {
        CliError::runtime(
            "output_io",
            format!("sync output {}: {error}", path.display()),
        )
    })
}
