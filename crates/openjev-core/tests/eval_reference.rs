use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use openjev_core::{
    DecisionOption, EvalFixture, GoldRow, Prediction, ProbabilityInput, browser_ladder_predictions,
    evaluate, fixture_gold,
};
use serde_json::{Value, json};

fn evaluator_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/semif-py/benchmarks/evaluate.py")
}

fn gold(id: &str, family: &str, label: usize) -> GoldRow {
    GoldRow {
        id: id.to_owned(),
        group_id: format!("group-{id}"),
        family: family.to_owned(),
        options: vec![
            DecisionOption {
                id: "left".to_owned(),
                description: "Left".to_owned(),
            },
            DecisionOption {
                id: "right".to_owned(),
                description: "Right".to_owned(),
            },
        ],
        label,
        metadata: serde_json::Map::new(),
    }
}

fn python_reference(gold: &[GoldRow], predictions: &[Prediction]) -> Value {
    let payload = json!({"gold": gold, "predictions": predictions});
    let script = r#"
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('reference_eval',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=json.load(sys.stdin); rows=m.align(p['gold'],p['predictions']); report=m.evaluate(p['gold'],p['predictions'])
out={
 'available_gold':report['available_gold'], 'scored':report['scored'],
 'evaluated':report['evaluated'], 'coverage':report['coverage'],
 'missing':report['missing'], 'invalid':report['invalid'],
 'mean_family_balanced_accuracy':report['mean_family_balanced_accuracy'],
 'mean_family_macro_f1':report['mean_family_macro_f1'],
 'overall':m.summarize(rows), 'family_results':report['family_results']}
sys.stdout.write(json.dumps(out,allow_nan=False))
"#;
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(evaluator_path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("python3 is required for the evaluation reference test");
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &payload).unwrap();
    child.stdin.take().unwrap().flush().unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64()
}

const SUMMARY_KEYS: &[&str] = &[
    "n",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "source_groups",
    "invalid_or_missing",
    "probability_rows",
    "probability_coverage",
    "nll",
    "brier",
    "nll_valid_distributions_only",
    "brier_valid_distributions_only",
    "nll_probability_floor",
];

fn assert_metric(left: &Value, right: &Value, path: &str) {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => assert!(
            (left - right).abs() <= 1e-12,
            "{path}: {left:?} != {right:?}"
        ),
        _ => assert_eq!(left, right, "{path}"),
    }
}

fn assert_python_subset(gold: &[GoldRow], predictions: &[Prediction]) {
    let rust = serde_json::to_value(evaluate(gold, predictions).unwrap()).unwrap();
    let python = python_reference(gold, predictions);
    for key in [
        "available_gold",
        "scored",
        "evaluated",
        "coverage",
        "missing",
        "invalid",
        "mean_family_balanced_accuracy",
        "mean_family_macro_f1",
    ] {
        assert_metric(&rust[key], &python[key], key);
    }
    for key in SUMMARY_KEYS {
        assert_metric(
            &rust["overall"][key],
            &python["overall"][key],
            &format!("overall.{key}"),
        );
    }
    for family in rust["family_results"].as_object().unwrap().keys() {
        for key in SUMMARY_KEYS {
            assert_metric(
                &rust["family_results"][family][key],
                &python["family_results"][family][key],
                &format!("{family}.{key}"),
            );
        }
    }
}

#[test]
fn full_frozen_qwen_fixtures_match_python_probability_subset() {
    let all_predictions = browser_ladder_predictions().unwrap();
    for fixture in [EvalFixture::Authored144, EvalFixture::Perturbations108] {
        let gold = fixture_gold(fixture).unwrap();
        let ids: std::collections::HashSet<_> = gold.iter().map(|row| row.id.as_str()).collect();
        let predictions: Vec<_> = all_predictions
            .iter()
            .filter(|prediction| ids.contains(prediction.id.as_str()))
            .cloned()
            .collect();
        assert_eq!(predictions.len(), fixture.expected_rows());
        assert_python_subset(&gold, &predictions);
    }
}

#[test]
fn hand_metrics_match_upstream_probability_subset() {
    let gold = vec![
        gold("one", "family-a", 0),
        gold("two", "family-a", 1),
        gold("three", "family-b", 1),
        gold("four", "family-b", 0),
    ];
    let predictions = vec![
        Prediction {
            id: "one".to_owned(),
            probabilities: Some(ProbabilityInput::Array(vec![0.5, 0.5])),
            option_ids: Some(vec!["right".to_owned(), "left".to_owned()]),
            prediction_id: None,
            parse_status: None,
            error: None,
        },
        Prediction {
            id: "two".to_owned(),
            probabilities: Some(ProbabilityInput::ById(std::collections::BTreeMap::from([
                ("left".to_owned(), 0.1),
                ("right".to_owned(), 0.9),
            ]))),
            option_ids: None,
            prediction_id: None,
            parse_status: None,
            error: None,
        },
        Prediction {
            id: "three".to_owned(),
            probabilities: Some(ProbabilityInput::Array(vec![0.8, 0.8])),
            option_ids: None,
            prediction_id: None,
            parse_status: None,
            error: None,
        },
    ];
    let rust = serde_json::to_value(evaluate(&gold, &predictions).unwrap()).unwrap();
    let python = python_reference(&gold, &predictions);
    for key in [
        "available_gold",
        "scored",
        "evaluated",
        "missing",
        "invalid",
    ] {
        assert_eq!(rust[key], python[key], "{key}");
    }
    for key in [
        "coverage",
        "mean_family_balanced_accuracy",
        "mean_family_macro_f1",
    ] {
        assert_eq!(number(&rust[key]), number(&python[key]), "{key}");
    }
    for key in [
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "source_groups",
        "invalid_or_missing",
        "probability_rows",
        "probability_coverage",
        "nll",
        "brier",
        "nll_valid_distributions_only",
        "brier_valid_distributions_only",
        "nll_probability_floor",
    ] {
        assert_eq!(
            rust["overall"][key], python["overall"][key],
            "overall.{key}"
        );
    }
    for family in ["family-a", "family-b"] {
        for key in [
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "source_groups",
            "invalid_or_missing",
            "probability_rows",
            "probability_coverage",
            "nll",
            "brier",
            "nll_valid_distributions_only",
            "brier_valid_distributions_only",
            "nll_probability_floor",
        ] {
            assert_eq!(
                rust["family_results"][family][key], python["family_results"][family][key],
                "{family}.{key}"
            );
        }
    }
}
