use std::{io::IsTerminal as _, process::ExitCode};

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .try_init();
    let stdin_handle = std::io::stdin();
    let stdin_is_terminal = stdin_handle.is_terminal();
    let mut stdin = stdin_handle.lock();
    // Do not hold the process stderr lock across native inference: llama.cpp's
    // tracing callback writes from the owner thread and would deadlock against
    // a main-thread StderrLock. Stdout/stderr still lock internally per write.
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let code = openjev_cli::run_with_io(
        std::env::args_os(),
        &mut stdin,
        stdin_is_terminal,
        &mut stdout,
        &mut stderr,
    );
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
