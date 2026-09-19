use std::{
    io::Write,
    process::{Command, Stdio},
};

use openjev_core::{Decision, parse_json_strict, prompt::direct_payload, python_json_dumps};
use serde_json::{Map, Number, Value};

fn python_dumps(value: &Value) -> String {
    let mut child = Command::new("python3")
        .args([
            "-c",
            "import json,sys; value=json.load(sys.stdin); sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False))",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("python3 is required for the differential test");
    serde_json::to_writer(child.stdin.as_mut().unwrap(), value).unwrap();
    child.stdin.take().unwrap().flush().unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn matches_python_stdlib_across_nested_order_limits_and_unicode() {
    let controls: String = (0..=31).map(char::from).collect();
    let mut nested = Map::new();
    nested.insert(
        "second".to_owned(),
        Value::String("雪 / \" \\ \u{2028}".to_owned()),
    );
    nested.insert("first".to_owned(), Value::String(controls));
    nested.insert(
        "signed_min".to_owned(),
        Value::Number(Number::from(i64::MIN)),
    );
    nested.insert(
        "unsigned_max".to_owned(),
        Value::Number(Number::from(u64::MAX)),
    );
    nested.insert(
        "values".to_owned(),
        Value::Array(vec![Value::Null, Value::Bool(true), Value::Bool(false)]),
    );
    let value = Value::Object(nested);
    assert_eq!(python_json_dumps(&value).unwrap(), python_dumps(&value));
}

#[test]
fn parsed_insertion_order_matches_python() {
    let value =
        parse_json_strict(r#"{"z": {"β": 2, "a": 1}, "a": [3, {"y": 4, "x": 5}]}"#).unwrap();
    assert_eq!(python_json_dumps(&value).unwrap(), python_dumps(&value));
}

#[test]
fn reserved_keys_in_decision_state_match_python_prompt_payload() {
    let decision = Decision::from_json_str(concat!(
        r#"{"id":"id","state":{"$serde_json::private::RawValue":"[1,2]","nested":[{"$serde_json::private::Number":"123"}]},"question":"q","options":["#,
        r#"{"id":"a","description":"A"},{"id":"b","description":"B"}]}"#,
    ))
    .unwrap();

    let mut payload = Map::new();
    payload.insert("evidence".to_owned(), decision.state.as_value().clone());
    payload.insert(
        "criterion".to_owned(),
        Value::String(decision.question.clone()),
    );
    payload.insert(
        "options".to_owned(),
        Value::Array(
            decision
                .options
                .iter()
                .enumerate()
                .map(|(index, option)| {
                    let mut rendered = Map::new();
                    rendered.insert(
                        "letter".to_owned(),
                        Value::String(char::from(b'A' + u8::try_from(index).unwrap()).to_string()),
                    );
                    rendered.insert(
                        "description".to_owned(),
                        Value::String(option.description.clone()),
                    );
                    Value::Object(rendered)
                })
                .collect(),
        ),
    );

    assert_eq!(
        direct_payload(&decision).unwrap(),
        python_dumps(&Value::Object(payload))
    );
}

#[test]
fn library_values_with_floats_are_rejected_recursively() {
    let value = serde_json::json!({"outer": [1, {"float": 1.0}]});
    let error = python_json_dumps(&value).unwrap_err().to_string();
    assert!(error.contains("$.outer[1].float"));
}
