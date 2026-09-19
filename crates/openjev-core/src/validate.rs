use std::{collections::HashSet, mem};

use serde::{Deserialize, Deserializer, de};
use serde_json::{Map, Number, Value, value::RawValue};

use crate::types::{OpenJevError, Result};

/// Maximum number of nested JSON arrays/objects accepted by public ingestion.
///
/// Scalars may occur immediately inside the deepest accepted container. The
/// bound applies equally to the hand parser, raw Serde JSON routes, and
/// already-built `Value` validation.
pub const MAX_JSON_DEPTH: usize = 128;

/// Parse one JSON value while preserving object order and rejecting duplicate keys.
///
/// Numeric tokens are deliberately restricted to the `i64`/`u64` integer range,
/// and nesting is bounded by [`MAX_JSON_DEPTH`].
pub fn parse_json_strict(input: &str) -> Result<Value> {
    let mut parser = Parser {
        input,
        bytes: input.as_bytes(),
        position: 0,
    };
    let value = parser.parse_value("$", 0)?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return Err(parser.syntax("$", "trailing characters after JSON value"));
    }
    Ok(value)
}

/// Capture one complete JSON value without losing its lexical representation,
/// then feed that text through [`parse_json_strict`].
///
/// The boxed form supports `serde_json` string, slice, reader, and owned
/// [`Value`] deserializers. An owned `Value` is necessarily reserialized by
/// `serde_json`, so source spelling and duplicate keys that were already lost
/// cannot be recovered. That upstream recursive serialization is not bounded
/// by this crate's depth check; use `StateValue::try_from` for an untrusted
/// already-built tree.
pub(crate) fn deserialize_strict_json_value<'de, D>(
    deserializer: D,
) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Box::<RawValue>::deserialize(deserializer)?;
    parse_json_strict(raw.get()).map_err(de::Error::custom)
}

/// Validate that every number in an already-built value is an `i64` or `u64`
/// integer and that nesting does not exceed [`MAX_JSON_DEPTH`].
pub fn validate_integer_json(value: &Value, path: &str) -> Result<()> {
    let mut pending = vec![(value, path.to_owned(), 0usize)];
    while let Some((value, path, depth)) = pending.pop() {
        match value {
            Value::Number(number) => {
                if number.as_i64().is_none() && number.as_u64().is_none() {
                    return Err(OpenJevError::Serialization {
                        path,
                        message: "floating-point and out-of-range numbers are not supported"
                            .to_owned(),
                    });
                }
            }
            Value::Array(values) => {
                check_depth(depth, &path)?;
                for (index, value) in values.iter().enumerate().rev() {
                    pending.push((value, format!("{path}[{index}]"), depth + 1));
                }
            }
            Value::Object(values) => {
                check_depth(depth, &path)?;
                for (key, value) in values.iter().rev() {
                    pending.push((value, object_path(&path, key), depth + 1));
                }
            }
            Value::Null | Value::Bool(_) | Value::String(_) => {}
        }
    }
    Ok(())
}

/// Dispose a potentially very deep `Value` without recursively dropping it.
pub(crate) fn drop_value_iteratively(value: Value) {
    let mut pending = vec![value];
    while let Some(mut value) = pending.pop() {
        match &mut value {
            Value::Array(values) => pending.append(values),
            Value::Object(values) => {
                for (_, child) in mem::take(values) {
                    pending.push(child);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
}

fn check_depth(depth: usize, path: &str) -> Result<()> {
    if depth < MAX_JSON_DEPTH {
        Ok(())
    } else {
        Err(OpenJevError::Serialization {
            path: path.to_owned(),
            message: format!("JSON nesting exceeds maximum depth {MAX_JSON_DEPTH}"),
        })
    }
}

struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    position: usize,
}

impl Parser<'_> {
    fn parse_value(&mut self, path: &str, depth: usize) -> Result<Value> {
        self.skip_whitespace();
        let Some(byte) = self.bytes.get(self.position).copied() else {
            return Err(self.syntax(path, "unexpected end of input"));
        };
        match byte {
            b'n' => {
                self.consume_literal(path, b"null")?;
                Ok(Value::Null)
            }
            b't' => {
                self.consume_literal(path, b"true")?;
                Ok(Value::Bool(true))
            }
            b'f' => {
                self.consume_literal(path, b"false")?;
                Ok(Value::Bool(false))
            }
            b'"' => self.parse_string(path).map(Value::String),
            b'[' => self.parse_array(path, depth),
            b'{' => self.parse_object(path, depth),
            b'-' | b'0'..=b'9' => self.parse_number(path),
            _ => Err(self.syntax(path, "expected a JSON value")),
        }
    }

    fn parse_array(&mut self, path: &str, depth: usize) -> Result<Value> {
        check_depth(depth, path)?;
        self.position += 1;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume_if(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            let child_path = format!("{path}[{}]", values.len());
            values.push(self.parse_value(&child_path, depth + 1)?);
            self.skip_whitespace();
            if self.consume_if(b']') {
                return Ok(Value::Array(values));
            }
            if !self.consume_if(b',') {
                return Err(self.syntax(path, "expected ',' or ']'"));
            }
        }
    }

    fn parse_object(&mut self, path: &str, depth: usize) -> Result<Value> {
        check_depth(depth, path)?;
        self.position += 1;
        self.skip_whitespace();
        let mut values = Map::new();
        let mut keys = HashSet::new();
        if self.consume_if(b'}') {
            return Ok(Value::Object(values));
        }
        loop {
            self.skip_whitespace();
            if self.bytes.get(self.position) != Some(&b'"') {
                return Err(self.syntax(path, "object key must be a string"));
            }
            let key = self.parse_string(path)?;
            let child_path = object_path(path, &key);
            if !keys.insert(key.clone()) {
                return Err(OpenJevError::Serialization {
                    path: child_path,
                    message: format!("duplicate JSON key {key:?}"),
                });
            }
            self.skip_whitespace();
            if !self.consume_if(b':') {
                return Err(self.syntax(path, "expected ':' after object key"));
            }
            let value = self.parse_value(&child_path, depth + 1)?;
            values.insert(key, value);
            self.skip_whitespace();
            if self.consume_if(b'}') {
                return Ok(Value::Object(values));
            }
            if !self.consume_if(b',') {
                return Err(self.syntax(path, "expected ',' or '}'"));
            }
        }
    }

    fn parse_string(&mut self, path: &str) -> Result<String> {
        let start = self.position;
        self.position += 1;
        let mut escaped = false;
        while let Some(byte) = self.bytes.get(self.position).copied() {
            if escaped {
                escaped = false;
                self.position += 1;
                continue;
            }
            match byte {
                b'\\' => {
                    escaped = true;
                    self.position += 1;
                }
                b'"' => {
                    self.position += 1;
                    let token = &self.input[start..self.position];
                    return serde_json::from_str(token).map_err(|error| {
                        self.syntax(path, &format!("invalid JSON string: {error}"))
                    });
                }
                0x00..=0x1f => {
                    return Err(self.syntax(path, "unescaped control character in string"));
                }
                _ => self.position += 1,
            }
        }
        Err(self.syntax(path, "unterminated JSON string"))
    }

    fn parse_number(&mut self, path: &str) -> Result<Value> {
        let start = self.position;
        if self.consume_if(b'-') && self.position == self.bytes.len() {
            return Err(self.syntax(path, "invalid number"));
        }
        match self.bytes.get(self.position).copied() {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => {
                self.position += 1;
                while matches!(self.bytes.get(self.position), Some(b'0'..=b'9')) {
                    self.position += 1;
                }
            }
            _ => return Err(self.syntax(path, "invalid number")),
        }
        let mut is_float = false;
        if self.consume_if(b'.') {
            is_float = true;
            if !matches!(self.bytes.get(self.position), Some(b'0'..=b'9')) {
                return Err(self.syntax(path, "fraction requires a digit"));
            }
            while matches!(self.bytes.get(self.position), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        if matches!(self.bytes.get(self.position), Some(b'e' | b'E')) {
            is_float = true;
            self.position += 1;
            if matches!(self.bytes.get(self.position), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if !matches!(self.bytes.get(self.position), Some(b'0'..=b'9')) {
                return Err(self.syntax(path, "exponent requires a digit"));
            }
            while matches!(self.bytes.get(self.position), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        let token = &self.input[start..self.position];
        if is_float {
            return Err(OpenJevError::Serialization {
                path: path.to_owned(),
                message: format!("floating-point number {token:?} is not supported"),
            });
        }
        if token.starts_with('-') {
            let value = token
                .parse::<i64>()
                .map_err(|_| OpenJevError::Serialization {
                    path: path.to_owned(),
                    message: format!("integer {token:?} is below the i64 range"),
                })?;
            Ok(Value::Number(Number::from(value)))
        } else {
            let value = token
                .parse::<u64>()
                .map_err(|_| OpenJevError::Serialization {
                    path: path.to_owned(),
                    message: format!("integer {token:?} exceeds the u64 range"),
                })?;
            Ok(Value::Number(Number::from(value)))
        }
    }

    fn consume_literal(&mut self, path: &str, literal: &[u8]) -> Result<()> {
        if self.bytes.get(self.position..self.position + literal.len()) == Some(literal) {
            self.position += literal.len();
            Ok(())
        } else {
            Err(self.syntax(path, "invalid literal"))
        }
    }

    fn consume_if(&mut self, expected: u8) -> bool {
        if self.bytes.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(
            self.bytes.get(self.position),
            Some(b' ' | b'\n' | b'\r' | b'\t')
        ) {
            self.position += 1;
        }
    }

    fn syntax(&self, path: &str, message: &str) -> OpenJevError {
        OpenJevError::Serialization {
            path: path.to_owned(),
            message: format!("{message} at byte {}", self.position),
        }
    }
}

pub(crate) fn object_path(parent: &str, key: &str) -> String {
    if !key.is_empty()
        && key.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_alphanumeric() && (index > 0 || !byte.is_ascii_digit())
        })
    {
        format!("{parent}.{key}")
    } else {
        let quoted = serde_json::to_string(key).unwrap_or_else(|_| "\"<invalid>\"".to_owned());
        format!("{parent}[{quoted}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_order_and_normalizes_negative_zero() {
        let value = parse_json_strict(r#"{"b": -0, "a": [1, null]}"#).unwrap();
        assert_eq!(value.to_string(), r#"{"b":0,"a":[1,null]}"#);
    }

    #[test]
    fn rejects_duplicates_at_nested_path() {
        let error = parse_json_strict(r#"{"outer": {"x": 1, "x": 2}}"#).unwrap_err();
        assert!(error.to_string().contains("$.outer.x"));
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn rejects_float_and_overflow_at_paths() {
        for (text, path) in [
            (r#"{"a":[1.0]}"#, "$.a[0]"),
            (r#"{"a":[1e0]}"#, "$.a[0]"),
            (r#"{"a":[-0.0]}"#, "$.a[0]"),
            (r#"{"a":18446744073709551616}"#, "$.a"),
            (r#"{"a":-9223372036854775809}"#, "$.a"),
        ] {
            assert!(
                parse_json_strict(text)
                    .unwrap_err()
                    .to_string()
                    .contains(path)
            );
        }
    }

    #[test]
    fn rejects_lone_surrogate() {
        assert!(parse_json_strict(r#""\ud800""#).is_err());
    }

    #[test]
    fn deeply_nested_raw_json_returns_error_without_recursing_unboundedly() {
        let input = format!("{}0{}", "[".repeat(5_000), "]".repeat(5_000));
        let error = parse_json_strict(&input).unwrap_err();
        assert!(error.to_string().contains("maximum depth 128"));
    }

    #[test]
    fn deeply_nested_built_value_is_rejected_and_dropped_iteratively() {
        let mut value = Value::Null;
        for _ in 0..5_000 {
            value = Value::Array(vec![value]);
        }
        let error = crate::StateValue::try_from(value).unwrap_err();
        assert!(error.to_string().contains("maximum depth 128"));
    }
}
