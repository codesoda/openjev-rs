pub mod args;

use std::{ffi::OsString, io::Write};

use clap::{CommandFactory, Parser, error::ErrorKind};
use openjev_core::ErrorRecord;
use serde::Serialize;

use crate::args::Cli;

#[derive(Serialize)]
struct HelpOutput {
    schema: &'static str,
    command: &'static str,
    usage: String,
    text: String,
}

#[derive(Serialize)]
struct VersionOutput {
    schema: &'static str,
    version: &'static str,
    build: &'static str,
}

pub fn run<I, T, W, E>(arguments: I, stdout: &mut W, stderr: &mut E) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    W: Write,
    E: Write,
{
    match Cli::try_parse_from(arguments) {
        Ok(_cli) => {
            let error = ErrorRecord::new(
                "backend_unavailable",
                "M1 provides parser and schema surfaces only; production inference starts in M2",
            );
            if write_json(stderr, &error, false).is_err() {
                return 1;
            }
            1
        }
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            let mut command = Cli::command();
            let usage = command.render_usage().to_string();
            let text = error.to_string();
            let output = HelpOutput {
                schema: "openjev-help-v1",
                command: "openjev",
                usage,
                text,
            };
            i32::from(write_json(stdout, &output, false).is_err())
        }
        Err(error) if error.kind() == ErrorKind::DisplayVersion => {
            let output = VersionOutput {
                schema: "openjev-version-v1",
                version: env!("CARGO_PKG_VERSION"),
                build: "backend-disabled-m1",
            };
            i32::from(write_json(stdout, &output, false).is_err())
        }
        Err(error) => {
            let output = ErrorRecord::new("usage", error.to_string());
            if write_json(stderr, &output, false).is_err() {
                return 2;
            }
            2
        }
    }
}

fn write_json<W: Write, T: Serialize>(
    writer: &mut W,
    value: &T,
    pretty: bool,
) -> std::io::Result<()> {
    let result = if pretty {
        serde_json::to_writer_pretty(&mut *writer, value)
    } else {
        serde_json::to_writer(&mut *writer, value)
    };
    result.map_err(std::io::Error::other)?;
    writer.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invoke(arguments: &[&str]) -> (i32, serde_json::Value, serde_json::Value) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run(arguments, &mut stdout, &mut stderr);
        let stdout = if stdout.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&stdout).unwrap()
        };
        let stderr = if stderr.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&stderr).unwrap()
        };
        (code, stdout, stderr)
    }

    #[test]
    fn help_and_version_are_json_only() {
        let (code, stdout, stderr) = invoke(&["openjev", "--help"]);
        assert_eq!(code, 0);
        assert_eq!(stdout["schema"], "openjev-help-v1");
        assert!(stderr.is_null());

        let (code, stdout, stderr) = invoke(&["openjev", "--version"]);
        assert_eq!(code, 0);
        assert_eq!(stdout["schema"], "openjev-version-v1");
        assert!(stderr.is_null());
    }

    #[test]
    fn scoring_never_returns_fake_inference() {
        let (code, stdout, stderr) = invoke(&[
            "openjev",
            "decide",
            "--question",
            "q",
            "--option",
            "a",
            "--option",
            "b",
            "--state",
            "s",
        ]);
        assert_eq!(code, 1);
        assert!(stdout.is_null());
        assert_eq!(stderr["error"]["code"], "backend_unavailable");
    }

    #[test]
    fn usage_errors_are_json_on_stderr() {
        let (code, stdout, stderr) = invoke(&["openjev", "unknown"]);
        assert_eq!(code, 2);
        assert!(stdout.is_null());
        assert_eq!(stderr["schema"], "openjev-error-v1");
    }

    #[test]
    fn parses_each_command_surface() {
        for arguments in [
            vec!["openjev", "noul", "--question", "q", "--state", "s"],
            vec![
                "openjev",
                "score",
                "--question",
                "q",
                "--level",
                "low",
                "--level",
                "high",
                "--state-json",
                "[1]",
            ],
            vec!["openjev", "ask", "--json", "{}"],
            vec!["openjev", "run", "--mode", "batch"],
            vec![
                "openjev",
                "models",
                "probe",
                "qwen3-0.6b",
                "--mode",
                "shared",
            ],
            vec!["openjev", "eval", "--fixture", "authored144"],
            vec!["openjev", "bench", "--state-file", "s", "--questions", "q"],
            vec!["openjev", "calibrate", "--input", "rows.jsonl"],
        ] {
            let (code, stdout, stderr) = invoke(&arguments);
            assert_eq!(code, 1, "{arguments:?}");
            assert!(stdout.is_null());
            assert_eq!(stderr["error"]["code"], "backend_unavailable");
        }
    }
}
