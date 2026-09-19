use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    numerics::first_argmax,
    types::{DecisionOption, OpenJevError, Result, serialize_f64, serialize_opt_f64},
};

const NLL_FLOOR: f64 = 1e-12;
const MASS_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoldRow {
    pub id: String,
    pub group_id: String,
    pub family: String,
    pub options: Vec<DecisionOption>,
    pub label: usize,
    #[serde(flatten)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProbabilityInput {
    Array(Vec<f64>),
    ById(BTreeMap<String, f64>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub id: String,
    #[serde(default)]
    pub probabilities: Option<ProbabilityInput>,
    #[serde(default)]
    pub option_ids: Option<Vec<String>>,
    #[serde(default)]
    pub prediction_id: Option<String>,
    #[serde(default)]
    pub parse_status: Option<String>,
    #[serde(default)]
    pub error: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvalSummary {
    pub n: usize,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub accuracy: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub balanced_accuracy: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub macro_f1: Option<f64>,
    pub source_groups: usize,
    pub invalid_or_missing: usize,
    pub probability_rows: usize,
    #[serde(serialize_with = "serialize_f64")]
    pub probability_coverage: f64,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub nll: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub brier: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub nll_valid_distributions_only: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub brier_valid_distributions_only: Option<f64>,
    #[serde(serialize_with = "serialize_f64")]
    pub nll_probability_floor: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvalErrorRow {
    pub id: String,
    pub status: EvalStatus,
    pub gold_id: String,
    pub predicted_id: Option<String>,
    pub message: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalStatus {
    Missing,
    Invalid,
    NativeDecision,
    Distribution,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvalReport {
    pub schema: String,
    pub available_gold: usize,
    pub scored: usize,
    pub evaluated: usize,
    #[serde(serialize_with = "serialize_f64")]
    pub coverage: f64,
    pub missing: usize,
    pub invalid: usize,
    pub family_results: BTreeMap<String, EvalSummary>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub mean_family_balanced_accuracy: Option<f64>,
    #[serde(serialize_with = "serialize_opt_f64")]
    pub mean_family_macro_f1: Option<f64>,
    pub overall: EvalSummary,
    pub errors: Vec<EvalErrorRow>,
    pub fixture_sha256: Option<String>,
    pub model_sha256: Option<String>,
    pub config_sha256: Option<String>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug)]
struct AlignedRow {
    id: String,
    group_id: String,
    family: String,
    gold_id: String,
    predicted_id: Option<String>,
    correct: bool,
    status: EvalStatus,
    probabilities: Option<Vec<f64>>,
    nll: Option<f64>,
    brier: Option<f64>,
    message: Option<String>,
}

pub fn evaluate(gold: &[GoldRow], predictions: &[Prediction]) -> Result<EvalReport> {
    let mut gold_ids = HashSet::with_capacity(gold.len());
    for row in gold {
        validate_gold(row)?;
        if !gold_ids.insert(row.id.as_str()) {
            return duplicate_error("gold", &row.id);
        }
    }
    let mut indexed = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if !gold_ids.contains(prediction.id.as_str()) {
            return Err(OpenJevError::Validation {
                path: "$.predictions".to_owned(),
                message: format!("unknown prediction ID {:?}", prediction.id),
            });
        }
        if indexed.insert(prediction.id.as_str(), prediction).is_some() {
            return duplicate_error("predictions", &prediction.id);
        }
    }

    let mut rows = Vec::with_capacity(gold.len());
    for item in gold {
        rows.push(align_one(item, indexed.get(item.id.as_str()).copied()));
    }

    let mut family_rows: BTreeMap<String, Vec<&AlignedRow>> = BTreeMap::new();
    for row in &rows {
        family_rows.entry(row.family.clone()).or_default().push(row);
    }
    let family_results: BTreeMap<_, _> = family_rows
        .iter()
        .map(|(family, rows)| (family.clone(), summarize(rows)))
        .collect();
    let mean_family_balanced_accuracy = mean_present(
        family_results
            .values()
            .map(|summary| summary.balanced_accuracy),
    );
    let mean_family_macro_f1 =
        mean_present(family_results.values().map(|summary| summary.macro_f1));
    let missing = rows
        .iter()
        .filter(|row| row.status == EvalStatus::Missing)
        .count();
    let invalid = rows
        .iter()
        .filter(|row| row.status == EvalStatus::Invalid)
        .count();
    let covered = rows.len() - missing - invalid;
    let errors = rows
        .iter()
        .filter(|row| !row.correct)
        .map(|row| EvalErrorRow {
            id: row.id.clone(),
            status: row.status,
            gold_id: row.gold_id.clone(),
            predicted_id: row.predicted_id.clone(),
            message: row.message.clone(),
        })
        .collect();
    let all: Vec<_> = rows.iter().collect();
    Ok(EvalReport {
        schema: "openjev-eval-v1".to_owned(),
        available_gold: gold.len(),
        scored: predictions.len(),
        evaluated: rows.len(),
        coverage: ratio(covered, rows.len()),
        missing,
        invalid,
        family_results,
        mean_family_balanced_accuracy,
        mean_family_macro_f1,
        overall: summarize(&all),
        errors,
        fixture_sha256: None,
        model_sha256: None,
        config_sha256: None,
        limitations: vec![
            "All gold rows count in accuracy, including missing/invalid/unparsed predictions."
                .to_owned(),
            "Native decisions have no probability estimates; one-hot distributions are not fabricated."
                .to_owned(),
            "This is the M1 probability/coverage subset; bootstrap and reliability fields are omitted."
                .to_owned(),
        ],
    })
}

fn align_one(gold: &GoldRow, prediction: Option<&Prediction>) -> AlignedRow {
    let ids: Vec<_> = gold
        .options
        .iter()
        .map(|option| option.id.clone())
        .collect();
    let gold_id = ids[gold.label].clone();
    let mut row = AlignedRow {
        id: gold.id.clone(),
        group_id: gold.group_id.clone(),
        family: gold.family.clone(),
        gold_id: gold_id.clone(),
        predicted_id: None,
        correct: false,
        status: EvalStatus::Missing,
        probabilities: None,
        nll: None,
        brier: None,
        message: None,
    };
    let Some(prediction) = prediction else {
        return row;
    };
    if prediction.parse_status.as_deref() == Some("unparsed") || prediction.error.is_some() {
        row.status = EvalStatus::Invalid;
        row.message = Some("unparsed or error prediction".to_owned());
        return row;
    }
    let aligned = match &prediction.probabilities {
        Some(probabilities) => {
            align_probabilities(probabilities, prediction.option_ids.as_deref(), &ids)
        }
        None => match &prediction.prediction_id {
            Some(choice) if ids.contains(choice) => {
                row.status = EvalStatus::NativeDecision;
                row.predicted_id = Some(choice.clone());
                row.correct = choice == &gold_id;
                return row;
            }
            _ => Err("missing or out-of-set native prediction".to_owned()),
        },
    };
    match aligned {
        Ok(probabilities) => {
            let index = first_argmax(&probabilities).expect("validated probabilities are nonempty");
            row.status = EvalStatus::Distribution;
            row.predicted_id = Some(ids[index].clone());
            row.correct = index == gold.label;
            row.nll = Some(-probabilities[gold.label].max(NLL_FLOOR).ln());
            row.brier = Some(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| {
                        let target = if index == gold.label { 1.0 } else { 0.0 };
                        (probability - target).powi(2)
                    })
                    .sum(),
            );
            row.probabilities = Some(probabilities);
        }
        Err(message) => {
            row.status = EvalStatus::Invalid;
            row.message = Some(message);
        }
    }
    row
}

fn align_probabilities(
    input: &ProbabilityInput,
    prediction_ids: Option<&[String]>,
    gold_ids: &[String],
) -> std::result::Result<Vec<f64>, String> {
    let values = match input {
        ProbabilityInput::Array(values) => {
            if values.len() != gold_ids.len() {
                return Err("wrong probability vector length".to_owned());
            }
            if let Some(prediction_ids) = prediction_ids {
                if prediction_ids.len() != values.len()
                    || prediction_ids.iter().collect::<HashSet<_>>().len() != values.len()
                    || prediction_ids.iter().collect::<HashSet<_>>()
                        != gold_ids.iter().collect::<HashSet<_>>()
                {
                    return Err("prediction option IDs differ from gold option IDs".to_owned());
                }
                let indexed: HashMap<_, _> = prediction_ids
                    .iter()
                    .zip(values)
                    .map(|(id, value)| (id.as_str(), *value))
                    .collect();
                gold_ids.iter().map(|id| indexed[id.as_str()]).collect()
            } else {
                values.clone()
            }
        }
        ProbabilityInput::ById(values) => {
            if values.keys().collect::<HashSet<_>>() != gold_ids.iter().collect::<HashSet<_>>() {
                return Err("probability keys differ from gold option IDs".to_owned());
            }
            gold_ids.iter().map(|id| values[id]).collect()
        }
    };
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("invalid or nonfinite probability".to_owned());
    }
    if (values.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE {
        return Err("probabilities do not sum to one".to_owned());
    }
    Ok(values)
}

fn validate_gold(row: &GoldRow) -> Result<()> {
    if row.id.is_empty() || row.group_id.is_empty() || row.family.is_empty() {
        return Err(OpenJevError::Validation {
            path: "$.gold".to_owned(),
            message: "gold id, group_id, and family must be nonempty".to_owned(),
        });
    }
    let ids: HashSet<_> = row
        .options
        .iter()
        .map(|option| option.id.as_str())
        .collect();
    if ids.len() != row.options.len() || row.label >= row.options.len() {
        return Err(OpenJevError::Validation {
            path: format!("$.gold[{:?}]", row.id),
            message: "invalid gold options or label".to_owned(),
        });
    }
    Ok(())
}

fn summarize(rows: &[&AlignedRow]) -> EvalSummary {
    if rows.is_empty() {
        return EvalSummary {
            n: 0,
            accuracy: None,
            balanced_accuracy: None,
            macro_f1: None,
            source_groups: 0,
            invalid_or_missing: 0,
            probability_rows: 0,
            probability_coverage: 0.0,
            nll: None,
            brier: None,
            nll_valid_distributions_only: None,
            brier_valid_distributions_only: None,
            nll_probability_floor: NLL_FLOOR,
        };
    }
    let labels: HashSet<_> = rows.iter().map(|row| row.gold_id.as_str()).collect();
    let mut recalls = Vec::with_capacity(labels.len());
    let mut f1s = Vec::with_capacity(labels.len());
    for label in labels {
        let true_positive = rows
            .iter()
            .filter(|row| row.gold_id == label && row.predicted_id.as_deref() == Some(label))
            .count();
        let false_positive = rows
            .iter()
            .filter(|row| row.gold_id != label && row.predicted_id.as_deref() == Some(label))
            .count();
        let false_negative = rows
            .iter()
            .filter(|row| row.gold_id == label && row.predicted_id.as_deref() != Some(label))
            .count();
        recalls.push(ratio(true_positive, true_positive + false_negative));
        f1s.push(ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ));
    }
    let valid: Vec<_> = rows
        .iter()
        .filter(|row| row.status == EvalStatus::Distribution)
        .collect();
    let mean_nll = mean(valid.iter().filter_map(|row| row.nll));
    let mean_brier = mean(valid.iter().filter_map(|row| row.brier));
    EvalSummary {
        n: rows.len(),
        accuracy: Some(ratio(
            rows.iter().filter(|row| row.correct).count(),
            rows.len(),
        )),
        balanced_accuracy: mean(recalls),
        macro_f1: mean(f1s),
        source_groups: rows
            .iter()
            .map(|row| row.group_id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        invalid_or_missing: rows
            .iter()
            .filter(|row| matches!(row.status, EvalStatus::Invalid | EvalStatus::Missing))
            .count(),
        probability_rows: valid.len(),
        probability_coverage: ratio(valid.len(), rows.len()),
        nll: (valid.len() == rows.len()).then_some(mean_nll).flatten(),
        brier: (valid.len() == rows.len()).then_some(mean_brier).flatten(),
        nll_valid_distributions_only: mean_nll,
        brier_valid_distributions_only: mean_brier,
        nll_probability_floor: NLL_FLOOR,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values: Vec<_> = values.into_iter().collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_present(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    mean(values.into_iter().flatten())
}

fn duplicate_error<T>(name: &str, id: &str) -> Result<T> {
    Err(OpenJevError::Validation {
        path: format!("$.{name}"),
        message: format!("duplicate ID {id:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gold(id: &str, family: &str, label: usize) -> GoldRow {
        GoldRow {
            id: id.to_owned(),
            group_id: format!("group-{id}"),
            family: family.to_owned(),
            options: vec![
                DecisionOption {
                    id: "a".to_owned(),
                    description: String::new(),
                },
                DecisionOption {
                    id: "b".to_owned(),
                    description: String::new(),
                },
            ],
            label,
            metadata: Map::new(),
        }
    }

    #[test]
    fn missing_and_invalid_rows_remain_in_denominators() {
        let gold = vec![
            gold("one", "f", 0),
            gold("two", "f", 1),
            gold("three", "f", 1),
        ];
        let predictions = vec![
            Prediction {
                id: "one".to_owned(),
                probabilities: Some(ProbabilityInput::Array(vec![0.5, 0.5])),
                option_ids: None,
                prediction_id: None,
                parse_status: None,
                error: None,
            },
            Prediction {
                id: "two".to_owned(),
                probabilities: Some(ProbabilityInput::Array(vec![0.9, 0.9])),
                option_ids: None,
                prediction_id: None,
                parse_status: None,
                error: None,
            },
        ];
        let report = evaluate(&gold, &predictions).unwrap();
        assert_eq!(report.missing, 1);
        assert_eq!(report.invalid, 1);
        assert_eq!(report.overall.n, 3);
        assert_eq!(report.overall.accuracy, Some(1.0 / 3.0));
        assert_eq!(report.overall.probability_rows, 1);
        assert_eq!(report.overall.nll, None);
    }

    #[test]
    fn aligns_semantic_ids_and_breaks_ties_in_gold_order() {
        let gold = vec![gold("one", "f", 0)];
        let predictions = vec![Prediction {
            id: "one".to_owned(),
            probabilities: Some(ProbabilityInput::Array(vec![0.5, 0.5])),
            option_ids: Some(vec!["b".to_owned(), "a".to_owned()]),
            prediction_id: None,
            parse_status: None,
            error: None,
        }];
        let report = evaluate(&gold, &predictions).unwrap();
        assert_eq!(report.overall.accuracy, Some(1.0));
    }

    #[test]
    fn rejects_unknown_and_duplicate_ids() {
        let gold = vec![gold("one", "f", 0)];
        let unknown = Prediction {
            id: "other".to_owned(),
            probabilities: None,
            option_ids: None,
            prediction_id: Some("a".to_owned()),
            parse_status: None,
            error: None,
        };
        assert!(evaluate(&gold, &[unknown]).is_err());
        assert!(evaluate(&[gold[0].clone(), gold[0].clone()], &[]).is_err());
    }
}
