use crate::types::{OpenJevError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct NumericReadout {
    pub option_logits: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub choice_index: usize,
    pub allowed_token_mass: f64,
    pub full_vocab_argmax_id: u32,
    pub full_vocab_log_normalizer: f64,
}

pub fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    if values.len() < 2 {
        return Err(OpenJevError::Validation {
            path: "$.option_logits".to_owned(),
            message: "need at least two logits".to_owned(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OpenJevError::Validation {
            path: "$.option_logits".to_owned(),
            message: "option logits must be finite".to_owned(),
        });
    }
    let maximum = values[first_argmax(values)?];
    let weights: Vec<_> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

pub fn first_argmax(values: &[f64]) -> Result<usize> {
    let Some((&first, remaining)) = values.split_first() else {
        return Err(OpenJevError::Validation {
            path: "$".to_owned(),
            message: "argmax requires a nonempty vector".to_owned(),
        });
    };
    if !first.is_finite() {
        return Err(OpenJevError::Validation {
            path: "$[0]".to_owned(),
            message: "argmax values must be finite".to_owned(),
        });
    }
    let mut best_index = 0;
    let mut best = first;
    for (offset, value) in remaining.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(OpenJevError::Validation {
                path: format!("$[{}]", offset + 1),
                message: "argmax values must be finite".to_owned(),
            });
        }
        if value > best {
            best = value;
            best_index = offset + 1;
        }
    }
    Ok(best_index)
}

pub fn normalized_margin(probabilities: &[f64]) -> Result<f64> {
    let count = probabilities.len();
    if count < 2
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
    {
        return Err(OpenJevError::Validation {
            path: "$.probabilities".to_owned(),
            message: "need a normalized finite probability vector".to_owned(),
        });
    }
    let maximum = probabilities[first_argmax(probabilities)?];
    let baseline = 1.0 / count as f64;
    Ok((maximum - baseline) / (1.0 - baseline))
}

pub fn read_logits(option_logits: &[f32], vocabulary_logits: &[f32]) -> Result<NumericReadout> {
    let option_logits: Vec<f64> = option_logits
        .iter()
        .map(|value| f64::from(*value))
        .collect();
    let probabilities = softmax(&option_logits)?;
    // The emitted choice follows the emitted f64 probabilities. Tiny raw-logit
    // differences can legitimately round away during exponentiation.
    let choice_index = first_argmax(&probabilities)?;

    if vocabulary_logits.is_empty() {
        return Err(OpenJevError::Validation {
            path: "$.vocabulary_logits".to_owned(),
            message: "vocabulary logits must not be empty".to_owned(),
        });
    }
    let mut maximum = f64::NEG_INFINITY;
    let mut maximum_index = 0usize;
    for (index, value) in vocabulary_logits.iter().copied().map(f64::from).enumerate() {
        if value.is_nan() || value == f64::INFINITY {
            return Err(OpenJevError::Validation {
                path: format!("$.vocabulary_logits[{index}]"),
                message: "vocabulary logits reject NaN and positive infinity".to_owned(),
            });
        }
        if value > maximum {
            maximum = value;
            maximum_index = index;
        }
    }
    if maximum == f64::NEG_INFINITY {
        return Err(OpenJevError::Validation {
            path: "$.vocabulary_logits".to_owned(),
            message: "vocabulary logits cannot all be negative infinity".to_owned(),
        });
    }
    let full_sum: f64 = vocabulary_logits
        .iter()
        .copied()
        .map(f64::from)
        .filter(|value| *value != f64::NEG_INFINITY)
        .map(|value| (value - maximum).exp())
        .sum();
    let full_vocab_log_normalizer = maximum + full_sum.ln();
    let option_maximum = option_logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // Use one origin for numerator and denominator. Subtracting two rounded
    // log-normalizers loses log(K) entirely for very large common offsets.
    let mass_origin = maximum.max(option_maximum);
    let option_mass_sum: f64 = option_logits
        .iter()
        .map(|value| (value - mass_origin).exp())
        .sum();
    let vocabulary_mass_sum: f64 = vocabulary_logits
        .iter()
        .copied()
        .map(f64::from)
        .filter(|value| *value != f64::NEG_INFINITY)
        .map(|value| (value - mass_origin).exp())
        .sum();
    if vocabulary_mass_sum == 0.0 || !vocabulary_mass_sum.is_finite() {
        return Err(OpenJevError::Validation {
            path: "$.allowed_token_mass".to_owned(),
            message: "selected logits cannot be reconciled with the full vocabulary".to_owned(),
        });
    }
    let mut allowed_token_mass = option_mass_sum / vocabulary_mass_sum;
    const ROUNDING_TOLERANCE: f64 = 1e-12;
    if !allowed_token_mass.is_finite() || allowed_token_mass < 0.0 {
        return Err(OpenJevError::Validation {
            path: "$.allowed_token_mass".to_owned(),
            message: "computed token mass is nonfinite or negative".to_owned(),
        });
    }
    if allowed_token_mass > 1.0 {
        if allowed_token_mass <= 1.0 + ROUNDING_TOLERANCE {
            allowed_token_mass = 1.0;
        } else {
            return Err(OpenJevError::Validation {
                path: "$.allowed_token_mass".to_owned(),
                message: "selected logits imply mass greater than the full vocabulary".to_owned(),
            });
        }
    }
    let full_vocab_argmax_id =
        u32::try_from(maximum_index).map_err(|_| OpenJevError::Validation {
            path: "$.full_vocab_argmax_id".to_owned(),
            message: "vocabulary index exceeds u32".to_owned(),
        })?;
    Ok(NumericReadout {
        option_logits,
        probabilities,
        choice_index,
        allowed_token_mass,
        full_vocab_argmax_id,
        full_vocab_log_normalizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_f64_softmax_and_first_ties() {
        let probabilities = softmax(&[1000.0, 1000.0, 999.0]).unwrap();
        assert_eq!(first_argmax(&probabilities).unwrap(), 0);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!((normalized_margin(&[0.5, 0.5]).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn vocabulary_ties_and_masked_entries_are_deterministic() {
        let result = read_logits(&[2.0, 1.0], &[2.0, 2.0, f32::NEG_INFINITY, 1.0]).unwrap();
        assert_eq!(result.choice_index, 0);
        assert_eq!(result.full_vocab_argmax_id, 0);
        assert!(result.allowed_token_mass > 0.0 && result.allowed_token_mass < 1.0);
    }

    #[test]
    fn mass_is_invariant_to_extreme_common_f32_offsets() {
        let base = read_logits(&[0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]).unwrap();
        let positive = read_logits(&[1.0e20, 1.0e20], &[1.0e20, 1.0e20, 1.0e20, 1.0e20]).unwrap();
        let negative =
            read_logits(&[-1.0e20, -1.0e20], &[-1.0e20, -1.0e20, -1.0e20, -1.0e20]).unwrap();
        for result in [base, positive, negative] {
            assert!((result.allowed_token_mass - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn choice_uses_first_probability_tie_after_rounding() {
        let result = read_logits(&[0.0, 1.0e-20], &[0.0, 1.0e-20]).unwrap();
        assert_eq!(result.probabilities, [0.5, 0.5]);
        assert_eq!(result.choice_index, 0);
    }

    #[test]
    fn rejects_nonfinite_and_contradictory_values() {
        assert!(read_logits(&[f32::NAN, 1.0], &[1.0]).is_err());
        assert!(read_logits(&[1.0, 0.0], &[f32::NEG_INFINITY]).is_err());
        assert!(read_logits(&[10.0, 9.0], &[0.0, -1.0]).is_err());
        assert!(read_logits(&[1.0, 0.0], &[f32::INFINITY]).is_err());
    }
}
