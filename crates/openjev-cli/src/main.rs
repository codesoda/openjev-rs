use std::{io::IsTerminal as _, process::ExitCode};

use clap::Parser as _;

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().collect();
    // Parse with clap rather than scanning argv: option values are allowed to
    // equal "--quiet" and must never change logging policy accidentally. The
    // library entry point parses again to retain its structured help/errors.
    let quiet = openjev_cli::args::Cli::try_parse_from(arguments.clone())
        .ok()
        .is_some_and(|cli| cli.global.quiet);
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .with_max_level(if quiet {
            tracing::Level::WARN
        } else {
            tracing::Level::INFO
        });
    let _ = subscriber.try_init();
    let stdin_handle = std::io::stdin();
    let stdin_is_terminal = stdin_handle.is_terminal();
    let mut stdin = stdin_handle.lock();
    // Do not hold the process stderr lock across native inference: llama.cpp's
    // tracing callback writes from the owner thread and would deadlock against
    // a main-thread StderrLock. Stdout/stderr still lock internally per write.
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let code = openjev_cli::run_with_io(
        arguments,
        &mut stdin,
        stdin_is_terminal,
        &mut stdout,
        &mut stderr,
    );
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
