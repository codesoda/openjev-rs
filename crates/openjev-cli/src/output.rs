use std::{fs::File, fs::OpenOptions, io::Write, path::Path};

use serde::Serialize;

use crate::CliError;

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
