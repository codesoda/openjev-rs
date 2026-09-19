use std::process::ExitCode;

fn main() -> ExitCode {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    let code = openjev_cli::run(std::env::args_os(), &mut stdout, &mut stderr);
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
