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
        let categorical_target_count = self.schema.target_groups.categorical.len();
        let boolean_target_count = self.schema.target_groups.boolean.len();
        let numeric_target_count = self.schema.target_groups.numeric.len();

        let mut categorical =
            Vec::with_capacity(sequence_count * sequence_length * categorical_count);
        let mut numeric = Vec::with_capacity(sequence_count * sequence_length * numeric_count);
        let mut categorical_targets = Vec::with_capacity(sequence_count * categorical_target_count);
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
            self.append_targets(
                sequence,
                &mut categorical_targets,
                &mut boolean_targets,
                &mut numeric_targets,
            )?;
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
                categorical: Tensor::from_slice(
                    &categorical_targets,
                    (sequence_count, categorical_target_count),
                    device,
                )?,
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
        categorical_targets: &mut Vec<u32>,
        boolean_targets: &mut Vec<f32>,
        numeric_targets: &mut Vec<f32>,
    ) -> Result<(), BuildError> {
        for target in &self.schema.target_groups.categorical {
            let value =
                sequence
                    .future_outcomes
                    .get(target)
                    .ok_or_else(|| BuildError::MissingTarget {
                        sample_id: sequence.sample_id.clone(),
                        target: target.clone(),
                    })?;
            let category = value.as_str().ok_or_else(|| BuildError::InvalidValue {
                sample_id: sequence.sample_id.clone(),
                feature: target.clone(),
                value: value.clone(),
            })?;
            let id = self
                .schema
                .categorical_vocab
                .get(target)
                .and_then(|vocab| vocab.get(category))
                .copied()
                .ok_or_else(|| BuildError::UnknownTargetCategory {
                    sample_id: sequence.sample_id.clone(),
                    target: target.clone(),
                    category: category.to_owned(),
                })?;
            categorical_targets.push(id);
        }

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
    UnknownTargetCategory {
        sample_id: String,
        target: String,
        category: String,
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

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use serde_json::json;

    use super::TrainingTensorBuilder;
    use crate::dataset::{FeatureSchema, MarketDataset, MarketSequence};

    #[test]
    fn categorical_target_is_encoded_as_class_id() {
        let schema: FeatureSchema = serde_json::from_value(json!({
            "version": 1,
            "categorical_vocab": {
                "phase": { "stable": 1 },
                "market_outcome": {
                    "CONTINUATION": 0,
                    "FAILED": 1,
                    "NO_MOVE": 2
                }
            },
            "model_input": {
                "categorical_features": ["phase"],
                "categorical_unknown_id": 0,
                "numeric_features": ["strength"]
            },
            "target_groups": {
                "categorical": ["market_outcome"],
                "boolean": ["valid"],
                "numeric": ["realized_range_atr"]
            }
        }))
        .unwrap();
        let sequence: MarketSequence = serde_json::from_value(json!({
            "sample_id": "sample-1",
            "source_name": "test",
            "seq_len": 1,
            "state_features": [{
                "snapshot_id": "snapshot-1",
                "values": { "phase": "stable", "strength": 0.8 }
            }],
            "range_telemetry": [{ "snapshot_id": "snapshot-1", "values": {} }],
            "cycle_context": [{ "snapshot_id": "snapshot-1", "values": {} }],
            "episode_context": [{ "snapshot_id": "snapshot-1", "values": {} }],
            "future_outcomes": {
                "market_outcome": "FAILED",
                "valid": true,
                "realized_range_atr": 1.25
            }
        }))
        .unwrap();
        let dataset = MarketDataset {
            schema,
            sequences: vec![sequence],
        };

        let tensors = TrainingTensorBuilder::new(&dataset.schema)
            .build(&dataset, &Device::Cpu)
            .unwrap();

        assert_eq!(
            tensors.targets.categorical.to_vec2::<u32>().unwrap(),
            vec![vec![1]]
        );
    }
}
