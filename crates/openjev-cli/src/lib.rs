pub mod args;
pub mod commands;
pub mod input;
mod m6;
mod m6_bench;
pub mod output;
pub mod server;

use std::{
    ffi::OsString,
    io::{Cursor, Read, Write},
};

use clap::{CommandFactory, Parser, error::ErrorKind};
use openjev_core::{Decision, ErrorRecord, ExecutionMode};
#[cfg(test)]
use serde_json::Value;

use crate::{
    args::{Cli, Command, ModelsCommand},
    commands::{Adapter, DecisionScorer},
    output::{HelpOutput, VersionOutput, WriteSummary},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErrorClass {
    Validation,
    Runtime,
}

#[derive(Debug)]
pub struct CliError {
    code: String,
    message: String,
    class: ErrorClass,
    id: Option<String>,
    unparsed: bool,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

impl CliError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self {
            code: "usage".to_owned(),
            message: message.into(),
            class: ErrorClass::Validation,
            id: None,
            unparsed: false,
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self {
            code: "validation".to_owned(),
            message: message.into(),
            class: ErrorClass::Validation,
            id: None,
            unparsed: false,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported".to_owned(),
            message: message.into(),
            class: ErrorClass::Validation,
            id: None,
            unparsed: false,
        }
    }

    pub fn runtime(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            class: ErrorClass::Runtime,
            id: None,
            unparsed: false,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn unparsed(mut self) -> Self {
        self.unparsed = true;
        self
    }

    pub fn from_core_validation(error: openjev_core::OpenJevError) -> Self {
        Self::validation(error.to_string())
    }

    pub fn from_runtime_core(error: openjev_core::OpenJevError) -> Self {
        Self::runtime(error.code(), error.to_string())
    }

    pub fn from_backend_validation(error: openjev_llama::BackendError) -> Self {
        Self::validation(error.to_string())
    }

    pub fn from_backend_runtime(error: openjev_llama::BackendError) -> Self {
        let code = match error {
            openjev_llama::BackendError::Unavailable => "backend_unavailable",
            openjev_llama::BackendError::OfflineMiss { .. } => "offline_miss",
            openjev_llama::BackendError::Integrity { .. }
            | openjev_llama::BackendError::CallerIntegrity { .. }
            | openjev_llama::BackendError::ArtifactChanged { .. } => "artifact_integrity",
            openjev_llama::BackendError::UnknownModel(_) => "unknown_model",
            openjev_llama::BackendError::Configuration(_) => "invalid_configuration",
            openjev_llama::BackendError::ModelLoad(_) => "model_load",
            openjev_llama::BackendError::Decode(_) => "decode",
            openjev_llama::BackendError::Worker(_) => "worker",
            _ => "backend",
        };
        Self::runtime(code, error.to_string())
    }

    fn exit_code(&self) -> i32 {
        match self.class {
            ErrorClass::Validation => 2,
            ErrorClass::Runtime => 1,
        }
    }

    fn record(&self) -> ErrorRecord {
        let mut record = ErrorRecord::new(self.code.clone(), self.message.clone());
        record.id.clone_from(&self.id);
        if self.unparsed {
            record.parse_status = Some("unparsed".to_owned());
        }
        record
    }
}

/// Compatibility entry point for callers that do not provide stdin.
///
/// stdin is treated as a TTY, so commands needing state/input fail rather than
/// reading an implicit empty stream.
pub fn run<I, T, W, E>(arguments: I, stdout: &mut W, stderr: &mut E) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    W: Write,
    E: Write,
{
    run_with_io(
        arguments,
        &mut Cursor::new(Vec::<u8>::new()),
        true,
        stdout,
        stderr,
    )
}

pub fn run_with_io<I, T, R, W, E>(
    arguments: I,
    stdin: &mut R,
    stdin_is_terminal: bool,
    stdout: &mut W,
    stderr: &mut E,
) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    R: Read,
    W: Write,
    E: Write,
{
    let arguments: Vec<OsString> = arguments.into_iter().map(Into::into).collect();
    match Cli::try_parse_from(arguments) {
        Ok(cli) => match execute(cli, stdin, stdin_is_terminal, stdout, stderr) {
            Ok(code) => code,
            Err(error) => emit_error(stderr, &error, false),
        },
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            let text = error.to_string();
            let (command, usage) = help_metadata(&text);
            let output = HelpOutput {
                schema: "openjev-help-v1",
                command,
                usage,
                text,
            };
            i32::from(output::write_json(stdout, &output, false).is_err())
        }
        Err(error) if error.kind() == ErrorKind::DisplayVersion => {
            let output = VersionOutput {
                schema: "openjev-version-v1",
                version: env!("CARGO_PKG_VERSION"),
                build: build_identity(),
            };
            i32::from(output::write_json(stdout, &output, false).is_err())
        }
        Err(error) => {
            let output = CliError {
                code: "usage".to_owned(),
                message: error.to_string(),
                class: ErrorClass::Validation,
                id: None,
                unparsed: false,
            };
            emit_error(stderr, &output, false)
        }
    }
}

fn execute<R: Read, W: Write, E: Write>(
    cli: Cli,
    stdin: &mut R,
    stdin_is_terminal: bool,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    let pretty = cli.global.pretty;
    let compact = cli.global.compact;
    let has_server_option = cli.host.is_some()
        || cli.port.is_some()
        || cli.request_timeout_secs.is_some()
        || cli.api_key_env.is_some();
    if cli.serve {
        if cli.command.is_some() {
            return Err(CliError::validation(
                "--serve cannot be combined with a CLI command",
            ));
        }
        server::validate_server_global_args(&cli.global)?;
        let options = server::ServeOptions::from_cli(
            cli.host,
            cli.port,
            cli.request_timeout_secs,
            cli.api_key_env.as_deref(),
        )?;
        let require_shared = cli.global.require_shared;
        let config = commands::scoring_config(&cli.global)?;
        return server::run(config, options, require_shared).map(|()| 0);
    }
    if has_server_option {
        return Err(CliError::validation(
            "--host, --port, --request-timeout-secs, and --api-key-env require --serve",
        ));
    }
    let command = cli
        .command
        .ok_or_else(|| CliError::usage("a command or --serve is required"))?;
    if compact
        && !matches!(
            &command,
            Command::Decide(_)
                | Command::Noul(_)
                | Command::Score(_)
                | Command::Ask(_)
                | Command::Run(_)
        )
    {
        return Err(CliError::validation(
            "--compact applies only to decision commands: decide, noul, score, ask, and run",
        ));
    }
    match command {
        Command::Decide(args) => {
            let state = input::read_state(&args.state, stdin, stdin_is_terminal)?;
            let items = commands::decide_items(args, state)?;
            commands::validate_unique_ids(&items)?;
            let requested_mode = commands::mode_for_decide(items.len());
            reject_pretty_multi(pretty, items.len())?;
            commands::check_require_shared(&cli.global, requested_mode)?;
            let config = commands::scoring_config(&cli.global)?;
            let group_id =
                (items.len() > 1).then(|| format!("openjev-group-{}", std::process::id()));
            let rows = score_all(
                &config,
                &items,
                requested_mode,
                group_id.as_deref(),
                cli.global.require_shared,
                stderr,
            )?;
            write_rows(stdout, &rows, pretty, compact)?;
            Ok(0)
        }
        Command::Noul(args) => {
            let state = input::read_state(&args.state, stdin, stdin_is_terminal)?;
            let item = commands::noul_item(args, state)?;
            commands::check_require_shared(&cli.global, ExecutionMode::Direct)?;
            let config = commands::scoring_config(&cli.global)?;
            let rows = score_all(&config, &[item], ExecutionMode::Direct, None, false, stderr)?;
            write_rows(stdout, &rows, pretty, compact)?;
            Ok(0)
        }
        Command::Score(args) => {
            let state = input::read_state(&args.state, stdin, stdin_is_terminal)?;
            let item = commands::score_item(args, state)?;
            commands::check_require_shared(&cli.global, ExecutionMode::Direct)?;
            let config = commands::scoring_config(&cli.global)?;
            let rows = score_all(&config, &[item], ExecutionMode::Direct, None, false, stderr)?;
            write_rows(stdout, &rows, pretty, compact)?;
            Ok(0)
        }
        Command::Ask(args) => {
            let decision = input::read_decision(
                args.json.as_deref(),
                args.input.as_deref(),
                stdin,
                stdin_is_terminal,
            )?;
            let item = Adapter::Choice(decision);
            commands::check_require_shared(&cli.global, ExecutionMode::Direct)?;
            let config = commands::scoring_config(&cli.global)?;
            let rows = score_all(&config, &[item], ExecutionMode::Direct, None, false, stderr)?;
            write_rows(stdout, &rows, pretty, compact)?;
            Ok(0)
        }
        Command::Run(args) => {
            if pretty {
                return Err(CliError::validation(
                    "--pretty is not valid for run JSONL output",
                ));
            }
            let rows = input::read_jsonl(args.input.as_deref(), stdin, stdin_is_terminal)?;
            validate_run_ids(&rows)?;
            let requested_mode = commands::mode_from_arg(args.mode);
            if requested_mode == ExecutionMode::Shared {
                commands::validate_shared_states(&rows)?;
            }
            commands::check_require_shared(&cli.global, requested_mode)?;
            if let Some(path) = &args.output {
                commands::preflight_output(path, args.input.as_deref())?;
            }
            let config = commands::scoring_config(&cli.global)?;
            execute_run(
                &config,
                rows,
                requested_mode,
                cli.global.require_shared,
                compact,
                args.output.as_deref(),
                stdout,
                stderr,
            )
        }
        Command::Models(args) => match args.command.unwrap_or(ModelsCommand::List) {
            ModelsCommand::List => {
                let output = commands::models_list(&cli.global)?;
                output::write_json(stdout, &output, pretty)
                    .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
                Ok(0)
            }
            ModelsCommand::Pull { id, repair } => {
                let output = models_resolve(&cli.global, &id, true, repair)?;
                output::write_json(stdout, &output, pretty)
                    .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
                Ok(0)
            }
            ModelsCommand::Path { id } => {
                let output = models_resolve(&cli.global, &id, false, false)?;
                output::write_json(stdout, &output, pretty)
                    .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
                Ok(0)
            }
            ModelsCommand::Probe { id, mode } => {
                if cli.global.model.is_some() {
                    return Err(CliError::validation(
                        "use the models probe positional ID instead of global --model",
                    ));
                }
                commands::reject_unimplemented_postprocessing(&cli.global)?;
                models_probe(&cli.global, &id, mode, stdout, stderr)
            }
        },
        Command::Eval(args) => m6::execute_eval(&cli.global, args, pretty, stdout, stderr),
        Command::Bench(args) => m6_bench::execute_bench(&cli.global, args, pretty, stdout, stderr),
        Command::Calibrate(_) => {
            commands::reject_unimplemented_postprocessing(&cli.global)?;
            Err(CliError::runtime(
                "not_implemented",
                "calibrate is an M7 surface and is not implemented in M4",
            ))
        }
    }
}

fn reject_pretty_multi(pretty: bool, count: usize) -> Result<(), CliError> {
    if pretty && count != 1 {
        Err(CliError::validation(
            "--pretty is only valid for a single JSON object, not JSONL",
        ))
    } else {
        Ok(())
    }
}

fn write_rows<W: Write>(
    writer: &mut W,
    rows: &[openjev_core::Readout],
    pretty: bool,
    compact: bool,
) -> Result<(), CliError> {
    if rows.len() == 1 {
        output::write_readout(writer, &rows[0], pretty, compact)
    } else {
        output::write_readout_jsonl(writer, rows, compact)
    }
    .map_err(|error| CliError::runtime("output_io", error.to_string()))
}

fn validate_run_ids(rows: &[Decision]) -> Result<(), CliError> {
    let mut ids = std::collections::HashSet::with_capacity(rows.len());
    for row in rows {
        if !ids.insert(row.id.as_str()) {
            return Err(CliError::validation(format!(
                "duplicate decision ID {:?}",
                row.id
            )));
        }
    }
    Ok(())
}

fn warn_fallback_reason(
    writer: &mut (impl Write + ?Sized),
    mode: ExecutionMode,
    reason: &str,
) -> Result<(), CliError> {
    writeln!(
        writer,
        "warning: requested {mode:?}; using fresh serial full-prompt fallback: {reason}"
    )
    .map_err(|error| CliError::runtime("stderr_io", error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn execute_run<W: Write, E: Write>(
    config: &commands::ScoringConfig,
    decisions: Vec<Decision>,
    requested_mode: ExecutionMode,
    require_shared: bool,
    compact: bool,
    output_path: Option<&std::path::Path>,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    let items: Vec<_> = decisions.into_iter().map(Adapter::Choice).collect();
    if require_shared {
        require_shared_eligibility(config)?;
    }
    // Input and output alias validation has already completed. Reserve the
    // create-only destination before model resolution/scoring so a racing
    // creator cannot cause inference whose rows have nowhere safe to go. If
    // later model startup fails, this deliberately leaves an empty file.
    let mut output_file = output_path.map(output::create_jsonl_new).transpose()?;
    let mut scorer = load_scorer(config)?;
    let group_id = matches!(requested_mode, ExecutionMode::Shared | ExecutionMode::Batch)
        .then(|| format!("openjev-group-{}", std::process::id()));
    let outcome = if let Some(file) = output_file.as_mut() {
        execute_run_groups_and_shutdown(
            scorer.as_mut(),
            &items,
            requested_mode,
            config.confidence,
            config.max_sequences,
            group_id.as_deref(),
            require_shared,
            compact,
            file,
            stderr,
        )?
    } else {
        execute_run_groups_and_shutdown(
            scorer.as_mut(),
            &items,
            requested_mode,
            config.confidence,
            config.max_sequences,
            group_id.as_deref(),
            require_shared,
            compact,
            stdout,
            stderr,
        )?
    };

    if let (Some(path), Some(file)) = (output_path, output_file.as_ref()) {
        output::sync_jsonl(file, path)?;
        let summary = WriteSummary {
            schema: "openjev-write-summary-v1",
            path: path.display().to_string(),
            written: outcome.written,
            failed: outcome.failed,
        };
        output::write_json(stdout, &summary, false)
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
    }
    Ok(i32::from(outcome.failed > 0))
}

#[allow(clippy::too_many_arguments)]
fn execute_run_groups_and_shutdown<W: Write + ?Sized, E: Write + ?Sized>(
    scorer: &mut dyn DecisionScorer,
    items: &[Adapter],
    requested_mode: ExecutionMode,
    confidence: bool,
    max_sequences: u32,
    group_id: Option<&str>,
    require_shared: bool,
    compact: bool,
    writer: &mut W,
    stderr: &mut E,
) -> Result<RunOutcome, CliError> {
    let operation = (|| {
        if !matches!(requested_mode, ExecutionMode::Shared | ExecutionMode::Batch) {
            return stream_run_rows(
                scorer,
                items,
                requested_mode,
                confidence,
                group_id,
                commands::fallback_reason(requested_mode),
                compact,
                writer,
                stderr,
            );
        }
        let native_group_limit = match requested_mode {
            ExecutionMode::Shared => max_sequences.saturating_sub(1).max(1),
            ExecutionMode::Batch => max_sequences,
            ExecutionMode::Direct | ExecutionMode::Serial => unreachable!(),
        } as usize;
        let mut outcome = RunOutcome {
            written: 0,
            failed: 0,
        };
        for group in items.chunks(native_group_limit) {
            match attempt_native_group(scorer, group, requested_mode, confidence, group_id) {
                Ok(rows) => {
                    write_completed_group(writer, &rows, compact)?;
                    outcome.written += rows.len();
                }
                Err(reason) => {
                    if require_shared && requested_mode == ExecutionMode::Shared {
                        return Err(CliError::unsupported(format!(
                            "--require-shared cannot be satisfied: {reason}"
                        )));
                    }
                    warn_fallback_reason(stderr, requested_mode, &reason)?;
                    let serial = stream_run_rows(
                        scorer,
                        group,
                        requested_mode,
                        confidence,
                        group_id,
                        Some(&reason),
                        compact,
                        writer,
                        stderr,
                    )?;
                    outcome.written += serial.written;
                    outcome.failed += serial.failed;
                }
            }
        }
        Ok(outcome)
    })();
    let shutdown = scorer.shutdown();
    match (operation, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

pub(crate) fn attempt_native_group(
    scorer: &mut dyn DecisionScorer,
    items: &[Adapter],
    mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
) -> Result<Vec<openjev_core::Readout>, String> {
    let probe_id = scorer.probe_id(mode)?;
    let decisions: Vec<_> = items.iter().map(|item| item.decision().clone()).collect();
    let raw = match mode {
        ExecutionMode::Shared => scorer.score_shared(decisions, probe_id.clone()),
        ExecutionMode::Batch => scorer.score_batch(decisions, probe_id.clone()),
        ExecutionMode::Direct | ExecutionMode::Serial => {
            return Err("group dispatch requires shared or batch mode".to_owned());
        }
    }
    .map_err(|error| {
        format!(
            "native {mode:?} attempt failed; tentative group discarded: {}",
            error.message
        )
    })?;
    if raw.len() != items.len() {
        return Err(format!(
            "native {mode:?} returned {} rows for {} inputs; tentative group discarded",
            raw.len(),
            items.len()
        ));
    }
    raw.into_iter()
        .zip(items)
        .map(|(readout, item)| {
            if readout.id != item.decision().id
                || readout.execution.requested_mode != mode
                || readout.execution.effective_mode != mode
                || readout.execution.probe_id.as_deref() != Some(probe_id.as_str())
            {
                return Err(format!(
                    "native {mode:?} result identity/metadata mismatch; tentative group discarded"
                ));
            }
            commands::adapt_group_readout(item, readout, confidence, group_id)
                .map_err(|error| format!("native {mode:?} adaptation failed: {}", error.message))
        })
        .collect()
}

fn write_completed_group<W: Write + ?Sized>(
    writer: &mut W,
    rows: &[openjev_core::Readout],
    compact: bool,
) -> Result<(), CliError> {
    // Shared/batch inference completes the bounded group (and all of its
    // internal waves) before any row is observable. Once complete, preserve
    // input order and flush each row.
    for row in rows {
        write_run_readout(writer, row, compact)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunOutcome {
    written: usize,
    failed: usize,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn stream_run_and_shutdown<W: Write + ?Sized, E: Write + ?Sized>(
    scorer: &mut dyn DecisionScorer,
    items: &[Adapter],
    requested_mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
    writer: &mut W,
    stderr: &mut E,
) -> Result<RunOutcome, CliError> {
    stream_run_and_shutdown_with_reason(
        scorer,
        items,
        requested_mode,
        confidence,
        group_id,
        commands::fallback_reason(requested_mode),
        writer,
        stderr,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn stream_run_and_shutdown_with_reason<W: Write + ?Sized, E: Write + ?Sized>(
    scorer: &mut dyn DecisionScorer,
    items: &[Adapter],
    requested_mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
    fallback_reason: Option<&str>,
    writer: &mut W,
    stderr: &mut E,
) -> Result<RunOutcome, CliError> {
    let streamed = stream_run_rows(
        scorer,
        items,
        requested_mode,
        confidence,
        group_id,
        fallback_reason,
        false,
        writer,
        stderr,
    );
    let shutdown = scorer.shutdown();
    match (streamed, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_run_rows<W: Write + ?Sized, E: Write + ?Sized>(
    scorer: &mut dyn DecisionScorer,
    items: &[Adapter],
    requested_mode: ExecutionMode,
    confidence: bool,
    group_id: Option<&str>,
    fallback_reason: Option<&str>,
    compact: bool,
    writer: &mut W,
    stderr: &mut E,
) -> Result<RunOutcome, CliError> {
    let mut outcome = RunOutcome {
        written: 0,
        failed: 0,
    };
    for item in items {
        match commands::score_item_with_reason(
            scorer,
            item,
            requested_mode,
            confidence,
            group_id,
            fallback_reason,
        ) {
            Ok(row) => write_run_readout(writer, &row, compact)?,
            Err(error) => {
                outcome.failed += 1;
                let error = error.with_id(item.decision().id.clone());
                write_run_row(writer, &error.record())?;
                writeln!(
                    stderr,
                    "row {:?} failed: {}",
                    item.decision().id,
                    error.message
                )
                .map_err(|write_error| CliError::runtime("stderr_io", write_error.to_string()))?;
            }
        }
        outcome.written += 1;
    }
    Ok(outcome)
}

fn write_run_row<W: Write + ?Sized, T: serde::Serialize>(
    writer: &mut W,
    row: &T,
) -> Result<(), CliError> {
    output::write_json(writer, row, false)
        .and_then(|()| writer.flush())
        .map_err(|error| CliError::runtime("output_io", error.to_string()))
}

fn write_run_readout<W: Write + ?Sized>(
    writer: &mut W,
    row: &openjev_core::Readout,
    compact: bool,
) -> Result<(), CliError> {
    output::write_readout(writer, row, false, compact)
        .and_then(|()| writer.flush())
        .map_err(|error| CliError::runtime("output_io", error.to_string()))
}

fn score_all(
    config: &commands::ScoringConfig,
    items: &[Adapter],
    requested_mode: ExecutionMode,
    group_id: Option<&str>,
    require_shared: bool,
    stderr: &mut (impl Write + ?Sized),
) -> Result<Vec<openjev_core::Readout>, CliError> {
    if require_shared {
        require_shared_eligibility(config)?;
    }
    let mut scorer = load_scorer(config)?;
    let result = if matches!(requested_mode, ExecutionMode::Shared | ExecutionMode::Batch) {
        let group_limit = match requested_mode {
            ExecutionMode::Shared => config.max_sequences.saturating_sub(1).max(1),
            ExecutionMode::Batch => config.max_sequences,
            ExecutionMode::Direct | ExecutionMode::Serial => unreachable!(),
        } as usize;
        let mut completed = Vec::with_capacity(items.len());
        let mut failure = None;
        for group in items.chunks(group_limit) {
            match attempt_native_group(
                scorer.as_mut(),
                group,
                requested_mode,
                config.confidence,
                group_id,
            ) {
                Ok(mut rows) => completed.append(&mut rows),
                Err(reason) => {
                    if require_shared && requested_mode == ExecutionMode::Shared {
                        failure = Some(CliError::unsupported(format!(
                            "--require-shared cannot be satisfied: {reason}"
                        )));
                        break;
                    }
                    if let Err(error) = warn_fallback_reason(stderr, requested_mode, &reason) {
                        failure = Some(error);
                        break;
                    }
                    for item in group {
                        match commands::score_item_with_reason(
                            scorer.as_mut(),
                            item,
                            requested_mode,
                            config.confidence,
                            group_id,
                            Some(&reason),
                        ) {
                            Ok(row) => completed.push(row),
                            Err(error) => {
                                failure = Some(error);
                                break;
                            }
                        }
                    }
                    if failure.is_some() {
                        break;
                    }
                }
            }
        }
        failure.map_or(Ok(completed), Err)
    } else {
        items
            .iter()
            .map(|item| {
                commands::score_item_with_reason(
                    scorer.as_mut(),
                    item,
                    requested_mode,
                    config.confidence,
                    group_id,
                    None,
                )
            })
            .collect()
    };
    let shutdown = scorer.shutdown();
    match (result, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(rows), Ok(())) => Ok(rows),
    }
}

#[cfg(feature = "native")]
fn require_shared_eligibility(config: &commands::ScoringConfig) -> Result<(), CliError> {
    commands::require_probe_eligibility(config, openjev_llama::ProbeMode::Shared).map(|_| ())
}

#[cfg(not(feature = "native"))]
fn require_shared_eligibility(_config: &commands::ScoringConfig) -> Result<(), CliError> {
    Err(CliError::unsupported(
        "--require-shared cannot be satisfied by a backend-disabled build",
    ))
}

#[cfg(feature = "native")]
pub(crate) fn load_scorer(
    config: &commands::ScoringConfig,
) -> Result<Box<dyn DecisionScorer>, CliError> {
    commands::NativeScorer::load(config).map(|scorer| Box::new(scorer) as Box<dyn DecisionScorer>)
}

#[cfg(not(feature = "native"))]
pub(crate) fn load_scorer(
    _config: &commands::ScoringConfig,
) -> Result<Box<dyn DecisionScorer>, CliError> {
    Err(CliError::runtime(
        "backend_unavailable",
        "production scoring requires building openjev-cli with native, metal, or cuda",
    ))
}

#[cfg(feature = "native")]
fn models_probe<W: Write, E: Write>(
    global: &args::GlobalArgs,
    id: &str,
    mode: args::ProbeModeArg,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    let mode = match mode {
        args::ProbeModeArg::Shared => openjev_llama::ProbeMode::Shared,
        args::ProbeModeArg::Batch => openjev_llama::ProbeMode::Batch,
    };
    if std::env::var("OPENJEV_PROBE_CHILD").as_deref() == Ok("1") {
        let receipt = commands::run_probe_child(global, id, mode)?;
        let enabled = receipt.passed;
        let report = commands::ProbeCommandReport {
            schema: "openjev-probe-report-v1".to_owned(),
            process_status: "completed".to_owned(),
            failure_reason: receipt.failure_reason.clone(),
            receipt: Some(receipt),
            receipt_path: None,
            enabled,
        };
        output::write_json(stdout, &report, global.pretty)
            .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
        return Ok(i32::from(!enabled));
    }

    // Establish the exact parent-owned key and suspend any prior passing
    // authorization before a child can initialize llama.cpp or create a context.
    // The held key lock serializes ordinary concurrent reprobes. Every early
    // return below intentionally leaves suspension in place.
    let publication = commands::prepare_probe_publication(global, id, mode)?;
    let executable = std::env::current_exe()
        .map_err(|error| CliError::runtime("probe_parent", error.to_string()))?;
    let mut command = std::process::Command::new(executable);
    append_probe_global_args(&mut command, global);
    command
        .args(["models", "probe", id, "--mode", mode.as_str()])
        .env("OPENJEV_PROBE_CHILD", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = command
        .output()
        .map_err(|error| CliError::runtime("probe_parent", error.to_string()))?;
    stderr
        .write_all(&child.stderr)
        .map_err(|error| CliError::runtime("stderr_io", error.to_string()))?;
    let parsed = serde_json::from_slice::<commands::ProbeCommandReport>(&child.stdout).ok();
    let mut report = if let Some(report) = parsed {
        if report.schema != "openjev-probe-report-v1" {
            commands::ProbeCommandReport {
                schema: "openjev-probe-report-v1".to_owned(),
                process_status: "invalid-child-report".to_owned(),
                receipt: None,
                receipt_path: None,
                enabled: false,
                failure_reason: Some("probe child returned an unexpected report schema".to_owned()),
            }
        } else {
            report
        }
    } else {
        let status = child.status.code().map_or_else(
            || "terminated-by-signal".to_owned(),
            |code| format!("exit-{code}"),
        );
        commands::ProbeCommandReport {
            schema: "openjev-probe-report-v1".to_owned(),
            process_status: status,
            receipt: None,
            receipt_path: None,
            enabled: false,
            failure_reason: Some(if child.stdout.is_empty() {
                "probe child crashed or failed before producing a JSON report".to_owned()
            } else {
                "probe child produced malformed JSON; no success receipt was accepted".to_owned()
            }),
        }
    };
    report.enabled = false;
    report.receipt_path = None;
    if report.process_status == "completed"
        && let Some(receipt) = report.receipt.clone()
    {
        if receipt.passed && !child.status.success() {
            report.process_status = "child-nonzero-after-passing-report".to_owned();
            report.failure_reason = Some(
                "probe child did not exit successfully; its passing candidate was not published"
                    .to_owned(),
            );
        } else {
            match publication.publish_child_result(&receipt, child.status.success()) {
                Ok(path) => {
                    report.receipt_path = Some(path.display().to_string());
                    report.enabled = receipt.passed && child.status.success();
                }
                Err(error) => {
                    report.process_status = "invalid-child-report".to_owned();
                    report.failure_reason = Some(format!(
                        "probe child receipt was not published by the parent: {error}"
                    ));
                }
            }
        }
    }
    let enabled = report.enabled;
    output::write_json(stdout, &report, global.pretty)
        .map_err(|error| CliError::runtime("output_io", error.to_string()))?;
    Ok(i32::from(!enabled))
}

#[cfg(feature = "native")]
fn append_probe_global_args(command: &mut std::process::Command, global: &args::GlobalArgs) {
    if let Some(value) = &global.model_sha256 {
        command.args(["--model-sha256", value]);
    }
    if let Some(value) = global.template_profile {
        command.args(["--template-profile", value.as_str()]);
    }
    if let Some(value) = &global.cache_dir {
        command.arg("--cache-dir").arg(value);
    }
    if global.offline {
        command.arg("--offline");
    }
    if let Some(value) = global.device {
        command.args([
            "--device",
            match value {
                args::DeviceArg::Cpu => "cpu",
                args::DeviceArg::Metal => "metal",
                args::DeviceArg::Cuda => "cuda",
            },
        ]);
    }
    if let Some(value) = &global.gpu_layers {
        command.args(["--gpu-layers", value]);
    }
    for (name, value) in [
        ("--threads", global.threads),
        ("--n-ctx", global.n_ctx),
        ("--max-tokens", global.max_tokens),
        ("--max-context-tokens", global.max_context_tokens),
        ("--n-batch", global.n_batch),
        ("--n-ubatch", global.n_ubatch),
        ("--max-sequences", global.max_sequences),
    ] {
        if let Some(value) = value {
            command.arg(name).arg(value.to_string());
        }
    }
    if global.pretty {
        command.arg("--pretty");
    }
    if global.quiet {
        command.arg("--quiet");
    }
}

#[cfg(not(feature = "native"))]
fn models_probe<W: Write, E: Write>(
    _global: &args::GlobalArgs,
    _id: &str,
    _mode: args::ProbeModeArg,
    _stdout: &mut W,
    _stderr: &mut E,
) -> Result<i32, CliError> {
    Err(CliError::runtime(
        "backend_unavailable",
        "models probe requires building openjev-cli with native, metal, or cuda",
    ))
}

#[cfg(feature = "native")]
fn models_resolve(
    global: &args::GlobalArgs,
    id: &str,
    pull: bool,
    repair: bool,
) -> Result<commands::ModelPathOutput, CliError> {
    commands::models_resolve(global, id, pull, repair)
}

#[cfg(not(feature = "native"))]
fn models_resolve(
    global: &args::GlobalArgs,
    id: &str,
    pull: bool,
    _repair: bool,
) -> Result<commands::ModelPathOutput, CliError> {
    commands::reject_models_irrelevant(global, if pull { "pull" } else { "path" })?;
    if global.model.is_some() {
        return Err(CliError::validation(
            "use the models pull/path positional ID instead of global --model",
        ));
    }
    if pull {
        return Err(CliError::runtime(
            "backend_unavailable",
            "models pull requires building openjev-cli with native, metal, or cuda",
        ));
    }
    if global.model_sha256.is_some() || global.template_profile.is_some() {
        return Err(CliError::runtime(
            "backend_unavailable",
            "custom model path verification requires a native CLI build",
        ));
    }
    let registry =
        openjev_llama::ModelRegistry::bundled().map_err(CliError::from_backend_runtime)?;
    let cache = openjev_llama::ModelCache::from_precedence(global.cache_dir.as_deref())
        .map_err(CliError::from_backend_runtime)?;
    let entry = registry
        .resolve(id)
        .map_err(CliError::from_backend_runtime)?;
    let artifact = registry
        .path(&cache, id)
        .map_err(CliError::from_backend_runtime)?;
    Ok(commands::ModelPathOutput {
        schema: "openjev-model-path-v1",
        id: entry.id.clone(),
        path: artifact.path.display().to_string(),
        bytes: artifact.bytes,
        sha256: artifact.sha256,
        integrity: "manifest-sha256".to_owned(),
        cache_hit: artifact.cache_hit,
    })
}

fn emit_error(writer: &mut impl Write, error: &CliError, pretty: bool) -> i32 {
    let code = error.exit_code();
    if output::write_json(writer, &error.record(), pretty).is_err() {
        return if code == 0 { 1 } else { code };
    }
    code
}

fn help_metadata(rendered_help: &str) -> (String, String) {
    let mut root = Cli::command();
    root.build();
    find_help_metadata(&mut root, rendered_help).unwrap_or_else(|| {
        let name = root
            .get_bin_name()
            .unwrap_or_else(|| root.get_name())
            .to_owned();
        let usage = root.render_usage().to_string();
        (name, usage)
    })
}

fn find_help_metadata(
    command: &mut clap::Command,
    rendered_help: &str,
) -> Option<(String, String)> {
    let short = command.clone().render_help().to_string();
    let long = command.clone().render_long_help().to_string();
    if rendered_help == short || rendered_help == long {
        let name = command
            .get_bin_name()
            .unwrap_or_else(|| command.get_name())
            .to_owned();
        let usage = command.clone().render_usage().to_string();
        return Some((name, usage));
    }
    for subcommand in command.get_subcommands_mut() {
        if let Some(metadata) = find_help_metadata(subcommand, rendered_help) {
            return Some(metadata);
        }
    }
    None
}

const fn build_identity() -> &'static str {
    if cfg!(feature = "cuda") {
        "m5-native-cuda"
    } else if cfg!(feature = "metal") {
        "m5-native-metal"
    } else if cfg!(feature = "native") {
        "m5-native-cpu"
    } else {
        "m5-backend-disabled"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use openjev_core::{DecisionOption, StateValue};

    use super::*;

    fn invoke(arguments: &[&str]) -> (i32, Value, Value) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run(arguments, &mut stdout, &mut stderr);
        let stdout = if stdout.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&stdout).unwrap()
        };
        let stderr = if stderr.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&stderr).unwrap()
        };
        (code, stdout, stderr)
    }

    #[test]
    fn help_and_version_are_json_only_and_include_piped_examples() {
        let (code, stdout, stderr) = invoke(&["openjev", "--help"]);
        assert_eq!(code, 0);
        assert_eq!(stdout["schema"], "openjev-help-v1");
        assert!(stdout["text"].as_str().unwrap().contains("printf"));
        assert!(stderr.is_null());

        let (code, stdout, stderr) = invoke(&["openjev", "--version"]);
        assert_eq!(code, 0);
        assert_eq!(stdout["schema"], "openjev-version-v1");
        assert!(stderr.is_null());
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn explicit_state_is_validated_without_reading_stdin() {
        struct PanicRead;
        impl Read for PanicRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("explicit state must not read stdin")
            }
        }
        let mut stdin = PanicRead;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run_with_io(
            [
                "openjev",
                "decide",
                "--question",
                "q",
                "--option",
                "a",
                "--option",
                "b",
                "--state",
                " exact ",
            ],
            &mut stdin,
            false,
            &mut stdout,
            &mut stderr,
        );
        assert_eq!(code, 1);
        assert!(stdout.is_empty());
        let error: Value = serde_json::from_slice(&stderr).unwrap();
        assert_eq!(error["error"]["code"], "backend_unavailable");
    }

    #[test]
    fn tty_without_state_is_validation_error_before_backend() {
        let (code, stdout, stderr) = invoke(&[
            "openjev",
            "decide",
            "--question",
            "q",
            "--option",
            "a",
            "--option",
            "b",
        ]);
        assert_eq!(code, 2);
        assert!(stdout.is_null());
        assert_eq!(stderr["error"]["code"], "validation");
    }

    #[test]
    fn unsupported_m5_and_m7_options_fail_before_backend() {
        let (code, _, stderr) = invoke(&[
            "openjev",
            "--permute",
            "2",
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
        assert_eq!(code, 2);
        assert_eq!(stderr["error"]["code"], "unsupported");

        #[cfg(not(feature = "native"))]
        {
            let (code, _, stderr) = invoke(&[
                "openjev",
                "--require-shared",
                "decide",
                "--question",
                "q1",
                "--question",
                "q2",
                "--option",
                "a",
                "--option",
                "b",
                "--state",
                "s",
            ]);
            assert_eq!(code, 2);
            assert_eq!(stderr["error"]["code"], "unsupported");
        }
    }

    #[test]
    fn serve_alternative_and_server_only_flags_fail_before_model_loading() {
        let (code, stdout, stderr) = invoke(&["openjev"]);
        assert_eq!(code, 2);
        assert!(stdout.is_null());
        assert_eq!(stderr["error"]["code"], "usage");

        for arguments in [
            vec!["openjev", "--host", "127.0.0.1", "models"],
            vec!["openjev", "--serve", "models"],
            vec!["openjev", "--serve", "--compact"],
            vec!["openjev", "--serve", "--host", "0.0.0.0"],
        ] {
            let (code, stdout, stderr) = invoke(&arguments);
            assert_eq!(code, 2, "arguments: {arguments:?}");
            assert!(stdout.is_null());
            assert_eq!(stderr["error"]["code"], "validation");
        }
    }

    #[test]
    fn parse_failure_precedes_backend_loading() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut stdin = Cursor::new(b"{bad}\n".to_vec());
        let code = run_with_io(
            ["openjev", "run"],
            &mut stdin,
            false,
            &mut stdout,
            &mut stderr,
        );
        assert_eq!(code, 2);
        assert!(stdout.is_empty());
        let error: Value = serde_json::from_slice(&stderr).unwrap();
        assert_eq!(error["parse_status"], "unparsed");
    }

    #[derive(Default)]
    struct StreamProbeState {
        bytes: Vec<u8>,
        flushes: usize,
        calls: usize,
        shutdowns: usize,
        fail_output: bool,
    }

    struct ProbeWriter {
        state: Arc<Mutex<StreamProbeState>>,
    }

    impl Write for ProbeWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut state = self.state.lock().unwrap();
            if state.fail_output {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected output failure",
                ));
            }
            state.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            let mut state = self.state.lock().unwrap();
            if state.fail_output {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected output failure",
                ));
            }
            state.flushes += 1;
            Ok(())
        }
    }

    struct ProbeScorer {
        state: Arc<Mutex<StreamProbeState>>,
        fail_output_on_call: Option<usize>,
    }

    impl DecisionScorer for ProbeScorer {
        fn score_direct(&mut self, _decision: Decision) -> Result<openjev_core::Readout, CliError> {
            let mut state = self.state.lock().unwrap();
            state.calls += 1;
            let call = state.calls;
            if call > 1 {
                assert_eq!(state.flushes, call - 1);
                assert_eq!(
                    state.bytes.iter().filter(|byte| **byte == b'\n').count(),
                    call - 1
                );
            }
            if self.fail_output_on_call == Some(call) {
                state.fail_output = true;
            }
            drop(state);
            Err(CliError::runtime(
                "injected_row_failure",
                format!("injected failure {call}"),
            ))
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.state.lock().unwrap().shutdowns += 1;
            Ok(())
        }
    }

    fn run_items(count: usize) -> Vec<Adapter> {
        (1..=count)
            .map(|index| {
                Adapter::Choice(
                    Decision::new(
                        format!("row-{index}"),
                        StateValue::string("state").unwrap(),
                        "question",
                        vec![
                            DecisionOption {
                                id: "a".to_owned(),
                                description: "A".to_owned(),
                            },
                            DecisionOption {
                                id: "b".to_owned(),
                                description: "B".to_owned(),
                            },
                        ],
                    )
                    .unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn run_stream_flushes_each_row_before_starting_the_next_score() {
        let state = Arc::new(Mutex::new(StreamProbeState::default()));
        let mut scorer = ProbeScorer {
            state: Arc::clone(&state),
            fail_output_on_call: None,
        };
        let mut writer = ProbeWriter {
            state: Arc::clone(&state),
        };
        let mut stderr = Vec::new();

        let outcome = stream_run_and_shutdown(
            &mut scorer,
            &run_items(2),
            ExecutionMode::Direct,
            false,
            None,
            &mut writer,
            &mut stderr,
        )
        .unwrap();

        assert_eq!(
            outcome,
            RunOutcome {
                written: 2,
                failed: 2
            }
        );
        let state = state.lock().unwrap();
        assert_eq!(state.calls, 2);
        assert_eq!(state.flushes, 2);
        assert_eq!(state.shutdowns, 1);
        let rows: Vec<Value> = String::from_utf8(state.bytes.clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows[0]["id"], "row-1");
        assert_eq!(rows[1]["id"], "row-2");
    }

    #[test]
    fn shared_run_streams_bounded_groups_and_falls_back_whole_group() {
        let state = Arc::new(Mutex::new(StreamProbeState::default()));
        let mut scorer = ProbeScorer {
            state: Arc::clone(&state),
            fail_output_on_call: None,
        };
        let mut writer = ProbeWriter {
            state: Arc::clone(&state),
        };
        let mut stderr = Vec::new();

        let outcome = execute_run_groups_and_shutdown(
            &mut scorer,
            &run_items(3),
            ExecutionMode::Shared,
            false,
            2,
            Some("bounded-group"),
            false,
            false,
            &mut writer,
            &mut stderr,
        )
        .unwrap();

        assert_eq!(outcome.written, 3);
        assert_eq!(outcome.failed, 3);
        assert_eq!(state.lock().unwrap().shutdowns, 1);
        assert_eq!(
            String::from_utf8(stderr)
                .unwrap()
                .lines()
                .filter(|line| line.contains("fresh serial full-prompt fallback"))
                .count(),
            3
        );
    }

    #[test]
    fn run_output_failure_stops_later_scoring_and_still_shuts_down() {
        let state = Arc::new(Mutex::new(StreamProbeState::default()));
        let mut scorer = ProbeScorer {
            state: Arc::clone(&state),
            fail_output_on_call: Some(2),
        };
        let mut writer = ProbeWriter {
            state: Arc::clone(&state),
        };
        let mut stderr = Vec::new();

        let error = stream_run_and_shutdown(
            &mut scorer,
            &run_items(3),
            ExecutionMode::Direct,
            false,
            None,
            &mut writer,
            &mut stderr,
        )
        .unwrap_err();

        assert_eq!(error.code, "output_io");
        let state = state.lock().unwrap();
        assert_eq!(state.calls, 2, "third row must not be scored");
        assert_eq!(state.flushes, 1);
        assert_eq!(state.shutdowns, 1);
        assert_eq!(state.bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
    }
}
