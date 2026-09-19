use crate::{
    numerics::first_argmax,
    types::{Decision, DecisionOption, OpenJevError, Primitive, Readout, Result, StateValue},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Noul {
    decision: Decision,
}

impl Noul {
    pub fn new(
        id: impl Into<String>,
        state: StateValue,
        question: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            decision: Decision::new(
                id,
                state,
                question,
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
            )?,
        })
    }

    #[must_use]
    pub const fn decision(&self) -> &Decision {
        &self.decision
    }

    pub fn adapt(&self, mut readout: Readout) -> Result<Readout> {
        ensure_alignment(&self.decision, &readout)?;
        readout.primitive = Primitive::Noul;
        readout.p_yes = Some(readout.probabilities[0]);
        readout.level_values = None;
        readout.expected_value = None;
        readout.argmax_level = None;
        readout.validate()?;
        Ok(readout)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoreLevel {
    pub id: String,
    pub description: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    decision: Decision,
    values: Vec<f64>,
}

impl Score {
    pub fn new(
        id: impl Into<String>,
        state: StateValue,
        question: impl Into<String>,
        levels: Vec<ScoreLevel>,
    ) -> Result<Self> {
        if levels.iter().any(|level| !level.value.is_finite()) {
            return Err(OpenJevError::Validation {
                path: "$.levels".to_owned(),
                message: "score level values must be finite".to_owned(),
            });
        }
        let values = levels.iter().map(|level| level.value).collect();
        let options = levels
            .into_iter()
            .map(|level| DecisionOption {
                id: level.id,
                description: level.description,
            })
            .collect();
        Ok(Self {
            decision: Decision::new(id, state, question, options)?,
            values,
        })
    }

    pub fn with_default_values(
        id: impl Into<String>,
        state: StateValue,
        question: impl Into<String>,
        levels: Vec<DecisionOption>,
    ) -> Result<Self> {
        let levels = levels
            .into_iter()
            .enumerate()
            .map(|(index, level)| ScoreLevel {
                id: level.id,
                description: level.description,
                value: index as f64,
            })
            .collect();
        Self::new(id, state, question, levels)
    }

    #[must_use]
    pub const fn decision(&self) -> &Decision {
        &self.decision
    }

    pub fn adapt(&self, mut readout: Readout) -> Result<Readout> {
        ensure_alignment(&self.decision, &readout)?;
        let expected_value = readout
            .probabilities
            .iter()
            .zip(&self.values)
            .map(|(probability, value)| probability * value)
            .sum();
        let index = first_argmax(&readout.probabilities)?;
        readout.primitive = Primitive::Score;
        readout.p_yes = None;
        readout.level_values = Some(self.values.clone());
        readout.expected_value = Some(expected_value);
        readout.argmax_level = Some(self.decision.options[index].id.clone());
        readout.validate()?;
        Ok(readout)
    }
}

fn ensure_alignment(decision: &Decision, readout: &Readout) -> Result<()> {
    let ids: Vec<_> = decision
        .options
        .iter()
        .map(|option| option.id.clone())
        .collect();
    if readout.id != decision.id || readout.option_ids != ids {
        return Err(OpenJevError::Validation {
            path: "$".to_owned(),
            message: "readout does not align with primitive decision".to_owned(),
        });
    }
    if readout.probabilities.len() != decision.options.len() {
        return Err(OpenJevError::Validation {
            path: "$.probabilities".to_owned(),
            message: "probabilities do not align with primitive levels".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Device, ExecutionMetadata, ExecutionMode, GpuLayersRequested, GpuLayersStatus, Integrity,
        ModelMetadata, PromptProfile, TemplateMetadataStatus,
    };

    fn readout(id: &str, option_ids: Vec<String>, probabilities: Vec<f64>) -> Readout {
        let choice_index = crate::first_argmax(&probabilities).unwrap();
        Readout {
            schema: "openjev-readout-v1".to_owned(),
            id: id.to_owned(),
            primitive: Primitive::Choice,
            choice: option_ids[choice_index].clone(),
            choice_index,
            option_logits: vec![0.0; option_ids.len()],
            answer_token_ids: (0..u32::try_from(option_ids.len()).unwrap()).collect(),
            allowed_token_mass: 0.5,
            full_vocab_argmax_id: 0,
            full_vocab_log_normalizer: 1.0,
            input_tokens: 1,
            forward_seconds: None,
            total_seconds: None,
            prompt_sha256: "0".repeat(64),
            prompt_version: "direct-options-v1".to_owned(),
            model: ModelMetadata {
                id: "test".to_owned(),
                source: "test".to_owned(),
                revision: "test".to_owned(),
                file: "test".to_owned(),
                quant: "test".to_owned(),
                backend: "test".to_owned(),
                artifact_sha256: "0".repeat(64),
                integrity: Integrity::LocalUnverified,
                dtype: "test".to_owned(),
                native_reference: None,
                template_profile: PromptProfile::Qwen3,
                template_sha256: None,
                template_override: true,
                template_status: TemplateMetadataStatus::OverrideUnverified,
                template_equivalence_evidence: None,
                serving_config: None,
                adapter: None,
                adapter_sha256: None,
                adapter_revision: None,
                torch_version: None,
                transformers_version: None,
            },
            readout:
                "native full-vocabulary last-position logits restricted to declared answer slots"
                    .to_owned(),
            probability_status: "conditional option score; uncalibrated as decision confidence"
                .to_owned(),
            limitations: crate::standard_limitations(),
            execution: ExecutionMetadata {
                requested_mode: ExecutionMode::Direct,
                effective_mode: ExecutionMode::Direct,
                fallback_reason: None,
                device: Device::Cpu,
                device_name: "test".to_owned(),
                gpu_layers_requested: GpuLayersRequested::Count(0),
                gpu_layers_actual: Some(0),
                gpu_layers_status: GpuLayersStatus::KnownDisabled,
                threads: 1,
                n_ctx_requested: None,
                n_ctx_actual: 1,
                max_tokens: 1,
                n_batch: 1,
                n_ubatch: 1,
                n_seq_max: 1,
                kv_unified: true,
                waves: 1,
                probe_id: None,
                run_id: "test".to_owned(),
                group_id: None,
            },
            confidence: None,
            confidence_status: None,
            p_yes: None,
            level_values: None,
            expected_value: None,
            argmax_level: None,
            cache_hit: None,
            prefix_tokens: None,
            prefix_sha256: None,
            prefill_seconds: None,
            copy_seconds: None,
            suffix_forward_seconds: None,
            shared_timing: None,
            postprocess: None,
            probabilities,
            option_ids,
        }
    }

    #[test]
    fn noul_uses_yes_first() {
        let noul = Noul::new("n", StateValue::string("s").unwrap(), "q").unwrap();
        let result = noul
            .adapt(readout(
                "n",
                vec!["yes".to_owned(), "no".to_owned()],
                vec![0.25, 0.75],
            ))
            .unwrap();
        assert_eq!(result.p_yes, Some(0.25));
    }

    #[test]
    fn score_preserves_distribution_and_first_tie() {
        let score = Score::new(
            "s",
            StateValue::string("s").unwrap(),
            "q",
            vec![
                ScoreLevel {
                    id: "low".to_owned(),
                    description: "Low".to_owned(),
                    value: -1.0,
                },
                ScoreLevel {
                    id: "high".to_owned(),
                    description: "High".to_owned(),
                    value: 3.0,
                },
            ],
        )
        .unwrap();
        let result = score
            .adapt(readout(
                "s",
                vec!["low".to_owned(), "high".to_owned()],
                vec![0.5, 0.5],
            ))
            .unwrap();
        assert_eq!(result.probabilities, [0.5, 0.5]);
        assert_eq!(result.expected_value, Some(1.0));
        assert_eq!(result.argmax_level.as_deref(), Some("low"));
    }

    #[test]
    fn readout_rejects_nonfinite_json_instead_of_emitting_null() {
        let mut result = readout(
            "n",
            vec!["yes".to_owned(), "no".to_owned()],
            vec![0.25, 0.75],
        );
        result.option_logits[0] = f64::NAN;
        assert!(serde_json::to_string(&result).is_err());
    }
}
