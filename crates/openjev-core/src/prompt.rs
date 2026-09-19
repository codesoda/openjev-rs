use std::{fmt::Write as _, str::FromStr};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    types::{Decision, DecisionOption, OpenJevError, Result, StateValue},
    validate::{object_path, validate_integer_json},
};

pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
pub const PROMPT_VERSION: &str = "direct-options-v1";
const LETTERS: &[u8; 16] = b"ABCDEFGHIJKLMNOP";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptProfile {
    #[serde(rename = "qwen3")]
    Qwen3,
    #[serde(rename = "qwen3.5")]
    Qwen35,
    #[serde(rename = "minicpm5")]
    MiniCpm5,
}

impl PromptProfile {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen3.5",
            Self::MiniCpm5 => "minicpm5",
        }
    }
}

impl FromStr for PromptProfile {
    type Err = OpenJevError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "qwen3" => Ok(Self::Qwen3),
            "qwen3.5" => Ok(Self::Qwen35),
            "minicpm5" => Ok(Self::MiniCpm5),
            _ => Err(OpenJevError::Template(format!(
                "unknown prompt profile {value:?}"
            ))),
        }
    }
}

impl std::fmt::Display for PromptProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPrompt {
    pub text: String,
    pub prompt_sha256: String,
    pub prompt_version: String,
    pub profile: PromptProfile,
}

pub fn prepare_prompt(decision: &Decision, profile: PromptProfile) -> Result<PreparedPrompt> {
    decision.validate()?;
    let payload = direct_payload(decision)?;
    let prefix = if profile == PromptProfile::MiniCpm5 {
        "<s>"
    } else {
        ""
    };
    let text = format!(
        "{prefix}<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let prompt_sha256 = sha256_hex(text.as_bytes());
    Ok(PreparedPrompt {
        text,
        prompt_sha256,
        prompt_version: PROMPT_VERSION.to_owned(),
        profile,
    })
}

/// Reproduce Python `shared._state_prefix` up to the tokenizer boundary.
///
/// The returned text is the rendered two-message placeholder prompt through
/// the serialized evidence object with its closing brace removed. Callers
/// must tokenize it without BOS and drop exactly one final token.
pub fn state_prefix_text(state: &StateValue, profile: PromptProfile) -> Result<String> {
    let placeholder = Decision::new(
        "prefix-only",
        state.clone(),
        "prefix boundary placeholder",
        vec![
            DecisionOption {
                id: "yes".to_owned(),
                description: "Yes".to_owned(),
            },
            DecisionOption {
                id: "no".to_owned(),
                description: "No".to_owned(),
            },
        ],
    )?;
    let payload = direct_payload(&placeholder)?;
    let prompt = prepare_prompt(&placeholder, profile)?.text;
    let mut occurrences = prompt.match_indices(&payload);
    let first = occurrences.next().map(|(index, _)| index);
    if first.is_none() || occurrences.next().is_some() {
        return Err(OpenJevError::Template(
            "cannot locate exactly one unmodified placeholder payload in the rendered prompt"
                .to_owned(),
        ));
    }
    let evidence = format!("{{\"evidence\": {}", python_json_dumps(state.as_value())?);
    if !payload.starts_with(&evidence) {
        return Err(OpenJevError::Template(
            "evidence serialization changed before the shared prefix boundary".to_owned(),
        ));
    }
    let index = first.expect("checked above");
    Ok(format!("{}{}", &prompt[..index], evidence))
}

pub fn direct_payload(decision: &Decision) -> Result<String> {
    decision.validate()?;
    let mut output = String::from("{\"evidence\": ");
    write_python_value(decision.state.as_value(), "$", &mut output)?;
    output.push_str(", \"criterion\": ");
    write_python_string(&decision.question, &mut output);
    output.push_str(", \"options\": [");
    for (index, option) in decision.options.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str("{\"letter\": \"");
        output.push(char::from(LETTERS[index]));
        output.push_str("\", \"description\": ");
        write_python_string(&option.description, &mut output);
        output.push('}');
    }
    output.push_str("]}");
    Ok(output)
}

/// Serialize finite integer-only JSON with Python `json.dumps` defaults and
/// `ensure_ascii=False`.
pub fn python_json_dumps(value: &Value) -> Result<String> {
    validate_integer_json(value, "$")?;
    let mut output = String::new();
    write_python_value(value, "$", &mut output)?;
    Ok(output)
}

fn write_python_value(value: &Value, path: &str, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                write!(output, "{value}").expect("writing to String cannot fail");
            } else if let Some(value) = value.as_u64() {
                write!(output, "{value}").expect("writing to String cannot fail");
            } else {
                return Err(OpenJevError::Serialization {
                    path: path.to_owned(),
                    message: "floating-point and out-of-range numbers are not supported".to_owned(),
                });
            }
        }
        Value::String(value) => write_python_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write_python_value(value, &format!("{path}[{index}]"), output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write_python_string(key, output);
                output.push_str(": ");
                write_python_value(value, &object_path(path, key), output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn write_python_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => {
                write!(output, "\\u{:04x}", u32::from(character))
                    .expect("writing to String cannot fail");
            }
            _ => output.push(character),
        }
    }
    output.push('"');
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Number};

    use super::*;
    use crate::{DecisionOption, StateValue};

    #[test]
    fn renders_profiles_and_fixed_key_order() {
        let mut object = Map::new();
        object.insert("z".to_owned(), Value::Number(Number::from(1)));
        object.insert("a".to_owned(), Value::String("雪\n".to_owned()));
        let decision = Decision::new(
            "id",
            StateValue::try_from(Value::Object(object)).unwrap(),
            " question ",
            vec![
                DecisionOption {
                    id: String::new(),
                    description: String::new(),
                },
                DecisionOption {
                    id: "x".to_owned(),
                    description: "quoted \" /".to_owned(),
                },
            ],
        )
        .unwrap();
        let qwen = prepare_prompt(&decision, PromptProfile::Qwen3).unwrap();
        let mini = prepare_prompt(&decision, PromptProfile::MiniCpm5).unwrap();
        assert!(qwen.text.contains(r#"{"evidence": {"z": 1, "a": "雪\n"}, "criterion": " question ", "options": [{"letter": "A", "description": ""}, {"letter": "B", "description": "quoted \" /"}]}"#));
        assert_eq!(mini.text, format!("<s>{}", qwen.text));
        assert_eq!(qwen.prompt_sha256.len(), 64);
    }

    #[test]
    fn reserved_json_keys_retain_python_prompt_bytes() {
        let decision = Decision::from_json_str(concat!(
            r#"{"id":"id","state":{"$serde_json::private::RawValue":"[1,2]","nested":[{"$serde_json::private::Number":"123"}]},"question":"q","options":["#,
            r#"{"id":"a","description":"A"},{"id":"b","description":"B"}]}"#,
        ))
        .unwrap();
        assert_eq!(
            direct_payload(&decision).unwrap(),
            r#"{"evidence": {"$serde_json::private::RawValue": "[1,2]", "nested": [{"$serde_json::private::Number": "123"}]}, "criterion": "q", "options": [{"letter": "A", "description": "A"}, {"letter": "B", "description": "B"}]}"#
        );
    }

    #[test]
    fn writes_control_characters_like_python() {
        let controls: String = (0..=31).map(char::from).collect();
        let rendered = python_json_dumps(&Value::String(controls)).unwrap();
        assert!(rendered.starts_with(r#""\u0000\u0001\u0002"#));
        assert!(rendered.contains(r#"\b\t\n"#));
        assert!(rendered.ends_with(r#"\u001e\u001f""#));
    }

    #[test]
    fn shared_prefix_text_uses_ordered_evidence_and_placeholder_once() {
        let state = StateValue::parse_json(r#"{"b": 2, "a": [true, null]}"#).unwrap();
        let text = state_prefix_text(&state, PromptProfile::Qwen3).unwrap();
        assert!(text.ends_with(r#"{"evidence": {"b": 2, "a": [true, null]}"#));
        assert_eq!(text.matches("prefix boundary placeholder").count(), 0);
        assert_eq!(text.matches("<|im_start|>user\n").count(), 1);
    }

    #[test]
    fn retained_input_metadata_never_enters_prompt() {
        let decision = Decision::from_json_str(
            r#"{"id":"x","state":"s","question":"q","options":[{"id":"a","description":"A"},{"id":"b","description":"B"}],"family":"retained"}"#,
        )
        .unwrap();
        assert_eq!(decision.metadata()["family"], "retained");
        let prompt = prepare_prompt(&decision, PromptProfile::Qwen3).unwrap();
        assert!(!prompt.text.contains("retained"));
        assert!(!prompt.text.contains("family"));
    }
}
