use std::{fs, io::Read, path::Path};

use openjev_core::{Decision, StateValue};

use crate::{CliError, args::StateArgs};

pub fn read_state<R: Read>(
    args: &StateArgs,
    stdin: &mut R,
    stdin_is_terminal: bool,
) -> Result<StateValue, CliError> {
    if let Some(value) = &args.state {
        return StateValue::string(value.clone()).map_err(CliError::from_core_validation);
    }
    if let Some(path) = &args.state_file {
        let text = read_utf8_file(path)?;
        return StateValue::string(text).map_err(CliError::from_core_validation);
    }
    if let Some(value) = &args.state_json {
        return StateValue::parse_json(value).map_err(CliError::from_core_validation);
    }
    if let Some(path) = &args.state_json_file {
        let text = read_utf8_file(path)?;
        return StateValue::parse_json(&text).map_err(CliError::from_core_validation);
    }
    if stdin_is_terminal {
        return Err(CliError::validation(
            "state is required: use --state, --state-file, --state-json, --state-json-file, or pipe UTF-8 text on stdin",
        ));
    }
    let text = read_utf8(stdin, "stdin state")?;
    StateValue::string(text).map_err(CliError::from_core_validation)
}

pub fn read_decision<R: Read>(
    json: Option<&str>,
    input: Option<&Path>,
    stdin: &mut R,
    stdin_is_terminal: bool,
) -> Result<Decision, CliError> {
    let text = if let Some(json) = json {
        json.to_owned()
    } else if let Some(path) = input {
        read_utf8_file(path)?
    } else if stdin_is_terminal {
        return Err(CliError::validation(
            "ask requires --json, --input, or one Decision JSON object on piped stdin",
        ));
    } else {
        read_utf8(stdin, "stdin Decision JSON")?
    };
    Decision::from_json_str(&text).map_err(CliError::from_core_validation)
}

pub fn read_jsonl<R: Read>(
    input: Option<&Path>,
    stdin: &mut R,
    stdin_is_terminal: bool,
) -> Result<Vec<Decision>, CliError> {
    let text = if let Some(path) = input {
        read_utf8_file(path)?
    } else if stdin_is_terminal {
        return Err(CliError::validation(
            "run requires --input or Decision JSONL on piped stdin",
        ));
    } else {
        read_utf8(stdin, "stdin JSONL")?
    };
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = Decision::from_json_str(line).map_err(|error| {
            CliError::validation(format!("JSONL line {}: {error}", index + 1)).unparsed()
        })?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(CliError::validation(
            "run input must contain at least one nonblank JSONL row",
        ));
    }
    Ok(rows)
}

fn read_utf8_file(path: &Path) -> Result<String, CliError> {
    let bytes = fs::read(path)
        .map_err(|error| CliError::validation(format!("read {}: {error}", path.display())))?;
    String::from_utf8(bytes).map_err(|error| {
        CliError::validation(format!("{} is not valid UTF-8: {error}", path.display()))
    })
}

fn read_utf8(reader: &mut impl Read, label: &str) -> Result<String, CliError> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| CliError::validation(format!("read {label}: {error}")))?;
    String::from_utf8(bytes)
        .map_err(|error| CliError::validation(format!("{label} is not valid UTF-8: {error}")))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("openjev-m4-{name}-{}", std::process::id()))
    }

    #[test]
    fn text_and_json_state_sources_preserve_bytes_and_require_explicit_json() {
        let text_path = temporary("state.txt");
        let json_path = temporary("state.json");
        fs::write(&text_path, b"  file text\n").unwrap();
        fs::write(&json_path, br#"{"b": 1, "a": [true]}"#).unwrap();

        let mut stdin = Cursor::new(b"unused".to_vec());
        let text = read_state(
            &StateArgs {
                state_file: Some(text_path.clone()),
                ..StateArgs::default()
            },
            &mut stdin,
            false,
        )
        .unwrap();
        assert_eq!(text.as_value(), "  file text\n");

        let mut stdin = Cursor::new(b"unused".to_vec());
        let json = read_state(
            &StateArgs {
                state_json_file: Some(json_path.clone()),
                ..StateArgs::default()
            },
            &mut stdin,
            false,
        )
        .unwrap();
        assert_eq!(json.as_value().to_string(), r#"{"b":1,"a":[true]}"#);

        let mut stdin = Cursor::new(b" {\"looks\":\"json\"}\n".to_vec());
        let piped = read_state(&StateArgs::default(), &mut stdin, false).unwrap();
        assert_eq!(piped.as_value(), " {\"looks\":\"json\"}\n");

        fs::remove_file(text_path).unwrap();
        fs::remove_file(json_path).unwrap();
    }
}
