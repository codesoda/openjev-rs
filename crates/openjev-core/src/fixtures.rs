//! Embedded, credited M6 evaluation fixtures.
//!
//! These bytes are copied unchanged from the preserved SemIf reference
//! snapshot so installed binaries never depend on the caller's working
//! directory. See `THIRD_PARTY.md` and `assets/README.md`.

#[cfg(test)]
use sha2::{Digest, Sha256};

use crate::{Decision, GoldRow, OpenJevError, Prediction, types::Result};

pub const AUTHORED144_JSONL: &str = include_str!("../assets/authored144.jsonl");
pub const PERTURBATIONS108_JSONL: &str = include_str!("../assets/perturbations108.jsonl");
pub const BROWSER_LADDER_QWEN3_JSONL: &str =
    include_str!("../assets/browser-ladder-qwen3-0.6b.predictions.jsonl");

pub const AUTHORED144_SHA256: &str =
    "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079";
pub const PERTURBATIONS108_SHA256: &str =
    "1dd7ccf80518d0e34886478ca23982aa726e9daccd343b9e95cedaf6b569bec4";
pub const BROWSER_LADDER_QWEN3_SHA256: &str =
    "f0a626a7442879a14d5fbbe702638e047358644151e0b71552d81fef7ac515ac";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalFixture {
    Authored144,
    Perturbations108,
}

impl EvalFixture {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Authored144 => "authored144",
            Self::Perturbations108 => "perturbations108",
        }
    }

    #[must_use]
    pub const fn jsonl(self) -> &'static str {
        match self {
            Self::Authored144 => AUTHORED144_JSONL,
            Self::Perturbations108 => PERTURBATIONS108_JSONL,
        }
    }

    #[must_use]
    pub const fn sha256(self) -> &'static str {
        match self {
            Self::Authored144 => AUTHORED144_SHA256,
            Self::Perturbations108 => PERTURBATIONS108_SHA256,
        }
    }

    #[must_use]
    pub const fn expected_rows(self) -> usize {
        match self {
            Self::Authored144 => 144,
            Self::Perturbations108 => 108,
        }
    }
}

pub fn fixture_gold(fixture: EvalFixture) -> Result<Vec<GoldRow>> {
    parse_jsonl(fixture.jsonl(), fixture.name())
}

pub fn fixture_decisions(fixture: EvalFixture) -> Result<Vec<Decision>> {
    fixture
        .jsonl()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            Decision::from_json_str(line).map_err(|error| OpenJevError::Serialization {
                path: format!("embedded {} line {}", fixture.name(), index + 1),
                message: error.to_string(),
            })
        })
        .collect()
}

pub fn browser_ladder_predictions() -> Result<Vec<Prediction>> {
    parse_jsonl(BROWSER_LADDER_QWEN3_JSONL, "browser-ladder-qwen3-0.6b")
}

fn parse_jsonl<T>(text: &str, name: &str) -> Result<Vec<T>>
where
    T: serde::de::DeserializeOwned,
{
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| OpenJevError::Serialization {
                path: format!("embedded {name} line {}", index + 1),
                message: error.to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256(text: &str) -> String {
        Sha256::digest(text.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn embedded_assets_have_frozen_hashes_and_counts() {
        for fixture in [EvalFixture::Authored144, EvalFixture::Perturbations108] {
            assert_eq!(sha256(fixture.jsonl()), fixture.sha256());
            assert_eq!(
                fixture_gold(fixture).unwrap().len(),
                fixture.expected_rows()
            );
            assert_eq!(
                fixture_decisions(fixture).unwrap().len(),
                fixture.expected_rows()
            );
        }
        assert_eq!(
            sha256(BROWSER_LADDER_QWEN3_JSONL),
            BROWSER_LADDER_QWEN3_SHA256
        );
        assert_eq!(browser_ladder_predictions().unwrap().len(), 252);
    }
}
