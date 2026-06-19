use std::error::Error;
use std::fmt::{Display, Formatter};

use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::core::{ModelInputs, TrainingTargets, TrainingTensors};
use crate::dataset::{FeatureSchema, MarketDataset, MarketSequence};

pub struct TrainingTensorBuilder<'a> {
    schema: &'a FeatureSchema,
}

impl<'a> TrainingTensorBuilder<'a> {
    pub fn new(schema: &'a FeatureSchema) -> Self {
        Self { schema }
    }

    pub fn build(
        &self,
        dataset: &MarketDataset,
        device: &Device,
    ) -> Result<TrainingTensors, BuildError> {
        let sequence_count = dataset.sequences.len();
        let sequence_length = dataset.sequences[0].seq_len;
        let categorical_count = self.schema.model_input.categorical_features.len();
        let numeric_count = self.schema.model_input.numeric_features.len();
        let boolean_target_count = self.schema.target_groups.boolean.len();
        let numeric_target_count = self.schema.target_groups.numeric.len();

        let mut categorical =
            Vec::with_capacity(sequence_count * sequence_length * categorical_count);
        let mut numeric = Vec::with_capacity(sequence_count * sequence_length * numeric_count);
        let mut boolean_targets = Vec::with_capacity(sequence_count * boolean_target_count);
        let mut numeric_targets = Vec::with_capacity(sequence_count * numeric_target_count);

        for sequence in &dataset.sequences {
            if sequence.seq_len != sequence_length {
                return Err(BuildError::SequenceLengthMismatch {
                    sample_id: sequence.sample_id.clone(),
                    expected: sequence_length,
                    actual: sequence.seq_len,
                });
            }
            self.append_inputs(sequence, &mut categorical, &mut numeric)?;
            self.append_targets(sequence, &mut boolean_targets, &mut numeric_targets)?;
        }

        Ok(TrainingTensors {
            inputs: ModelInputs {
                categorical: Tensor::from_slice(
                    &categorical,
                    (sequence_count, sequence_length, categorical_count),
                    device,
                )?,
                numeric: Tensor::from_slice(
                    &numeric,
                    (sequence_count, sequence_length, numeric_count),
                    device,
                )?,
            },
            targets: TrainingTargets {
                boolean: Tensor::from_slice(
                    &boolean_targets,
                    (sequence_count, boolean_target_count),
                    device,
                )?,
                numeric: Tensor::from_slice(
                    &numeric_targets,
                    (sequence_count, numeric_target_count),
                    device,
                )?,
            },
        })
    }

    fn append_inputs(
        &self,
        sequence: &MarketSequence,
        categorical: &mut Vec<u32>,
        numeric: &mut Vec<f32>,
    ) -> Result<(), BuildError> {
        for step in 0..sequence.seq_len {
            for feature in &self.schema.model_input.categorical_features {
                let value = required_value(sequence, step, feature)?;
                let category = value.as_str().ok_or_else(|| BuildError::InvalidValue {
                    sample_id: sequence.sample_id.clone(),
                    feature: feature.clone(),
                    value: value.clone(),
                })?;
                let id = self
                    .schema
                    .categorical_vocab
                    .get(feature)
                    .and_then(|vocab| vocab.get(category))
                    .copied()
                    .unwrap_or(self.schema.model_input.categorical_unknown_id);
                categorical.push(id);
            }

            for feature in &self.schema.model_input.numeric_features {
                let value = required_value(sequence, step, feature)?;
                numeric.push(numeric_value(sequence, feature, value)?);
            }
        }
        Ok(())
    }

    fn append_targets(
        &self,
        sequence: &MarketSequence,
        boolean_targets: &mut Vec<f32>,
        numeric_targets: &mut Vec<f32>,
    ) -> Result<(), BuildError> {
        for target in &self.schema.target_groups.boolean {
            let value =
                sequence
                    .future_outcomes
                    .get(target)
                    .ok_or_else(|| BuildError::MissingTarget {
                        sample_id: sequence.sample_id.clone(),
                        target: target.clone(),
                    })?;
            let value = value.as_bool().ok_or_else(|| BuildError::InvalidValue {
                sample_id: sequence.sample_id.clone(),
                feature: target.clone(),
                value: value.clone(),
            })?;
            boolean_targets.push(if value { 1.0 } else { 0.0 });
        }

        for target in &self.schema.target_groups.numeric {
            let value =
                sequence
                    .future_outcomes
                    .get(target)
                    .ok_or_else(|| BuildError::MissingTarget {
                        sample_id: sequence.sample_id.clone(),
                        target: target.clone(),
                    })?;
            numeric_targets.push(numeric_value(sequence, target, value)?);
        }
        Ok(())
    }
}

fn required_value<'a>(
    sequence: &'a MarketSequence,
    step: usize,
    feature: &str,
) -> Result<&'a Value, BuildError> {
    sequence
        .value_at(step, feature)
        .ok_or_else(|| BuildError::MissingFeature {
            sample_id: sequence.sample_id.clone(),
            step,
            feature: feature.to_owned(),
        })
}

fn numeric_value(
    sequence: &MarketSequence,
    feature: &str,
    value: &Value,
) -> Result<f32, BuildError> {
    match value {
        Value::Number(number) => number.as_f64().map(|value| value as f32),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
    .ok_or_else(|| BuildError::InvalidValue {
        sample_id: sequence.sample_id.clone(),
        feature: feature.to_owned(),
        value: value.clone(),
    })
}

#[derive(Debug)]
pub enum BuildError {
    SequenceLengthMismatch {
        sample_id: String,
        expected: usize,
        actual: usize,
    },
    MissingFeature {
        sample_id: String,
        step: usize,
        feature: String,
    },
    MissingTarget {
        sample_id: String,
        target: String,
    },
    InvalidValue {
        sample_id: String,
        feature: String,
        value: Value,
    },
    Candle(candle_core::Error),
}

impl From<candle_core::Error> for BuildError {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}

impl Display for BuildError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for BuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Candle(source) => Some(source),
            _ => None,
        }
    }
}
