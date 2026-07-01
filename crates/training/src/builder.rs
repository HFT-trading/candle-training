//! Turn parsed contexts into training tensors (head A targets).
//!
//! Inputs come only from the raw 32-step sequence stream (no summaries — those
//! would trivialize the deterministic labels). Targets are 0-based class ids for
//! the per-block and per-relation label fields.

use candle_core::{Device, Tensor};
use serde_json::{Map, Value};
use structure_core::input::encode_step;
use structure_core::sequence::{BLOCK_LABEL_FIELDS, CATEGORICAL_FEATURES, NUMERIC_FEATURES};
use structure_core::tensors::ModelInputs;
use structure_core::vocab::FeatureVocab;

use crate::dataset::{Context, label_token};

pub struct TrainingTensors {
    pub inputs: ModelInputs,
    /// Block targets, shape `[N, blocks, BLOCK_LABEL_FIELDS]` (0-based class ids).
    pub block_targets: Tensor,
}

pub fn build_tensors(
    contexts: &[Context],
    vocab: &FeatureVocab,
    device: &Device,
) -> Result<TrainingTensors, BuildError> {
    let first = contexts.first().ok_or(BuildError::Empty)?;
    let seq_len = first.metadata.shape.sequence_len;
    let blocks = first.metadata.shape.context_blocks;
    let n = contexts.len();
    let n_cat = CATEGORICAL_FEATURES.len();
    let n_num = NUMERIC_FEATURES.len();

    let mut categorical = Vec::with_capacity(n * seq_len * n_cat);
    let mut numeric = Vec::with_capacity(n * seq_len * n_num);
    let mut block_targets = Vec::with_capacity(n * blocks * BLOCK_LABEL_FIELDS.len());

    for context in contexts {
        if context.training_data.sequences.len() != seq_len
            || context.labels.blocks.len() != blocks
        {
            return Err(BuildError::Shape);
        }
        for sequence in &context.training_data.sequences {
            encode_step(sequence, vocab, &mut categorical, &mut numeric);
        }
        for block in &context.labels.blocks {
            for field in BLOCK_LABEL_FIELDS {
                block_targets.push(target_id(vocab, "blocks", field, block)?);
            }
        }
    }

    Ok(TrainingTensors {
        inputs: ModelInputs {
            categorical: Tensor::from_vec(categorical, (n, seq_len, n_cat), device)?,
            numeric: Tensor::from_vec(numeric, (n, seq_len, n_num), device)?,
        },
        block_targets: Tensor::from_vec(
            block_targets,
            (n, blocks, BLOCK_LABEL_FIELDS.len()),
            device,
        )?,
    })
}

/// 0-based class id for a categorical label, erroring on missing/unknown values.
fn target_id(
    vocab: &FeatureVocab,
    level: &str,
    field: &str,
    values: &Map<String, Value>,
) -> Result<u32, BuildError> {
    let value = values
        .get(field)
        .and_then(label_token)
        .ok_or_else(|| BuildError::MissingLabel {
            level: level.to_owned(),
            field: field.to_owned(),
        })?;
    let key = format!("{level}.{field}");
    let id = vocab.get(&key).map(|entry| entry.id(&value)).unwrap_or(0);
    if id == 0 {
        return Err(BuildError::UnknownLabel { key, value });
    }
    Ok(id - 1)
}

#[derive(Debug)]
pub enum BuildError {
    Empty,
    Shape,
    MissingLabel { level: String, field: String },
    UnknownLabel { key: String, value: String },
    Candle(candle_core::Error),
}

impl From<candle_core::Error> for BuildError {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(formatter, "no contexts to build tensors from"),
            Self::Shape => write!(formatter, "context shape does not match the first row"),
            Self::MissingLabel { level, field } => {
                write!(formatter, "missing label {level}.{field}")
            }
            Self::UnknownLabel { key, value } => {
                write!(formatter, "unknown label value {value:?} for {key}")
            }
            Self::Candle(source) => write!(formatter, "tensor error: {source}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Candle(source) => Some(source),
            _ => None,
        }
    }
}
