//! Serving input contract: a plain struct the consumer fills, plus the single
//! canonical encoder shared by training and serving (so they never disagree).

use candle_core::{Device, Result, Tensor};

use crate::sequence::{CATEGORICAL_FEATURES, MISSING_CATEGORY, NUMERIC_FEATURES, PATTERN_NONE};
use crate::tensors::ModelInputs;
use crate::vocab::FeatureVocab;

/// Anything that can yield per-step feature values by canonical feature name.
/// Implemented by both the training `Sequence` and the serving `StepInput`.
pub trait StepFeatures {
    fn categorical(&self, feature: &str) -> &str;
    fn numeric(&self, feature: &str) -> f32;
}

/// Plain, dependency-light input for one sequence step. The hft-bot service
/// fills this (write your own `From<YourType> for StepInput`).
#[derive(Debug, Clone, Default)]
pub struct StepInput {
    pub micro_trend: String,
    pub direction_hint: String,
    pub vector_hint: String,
    pub bias_hint: String,
    pub quality_hint: String,
    pub behavior_hint: String,
    pub side: String,
    pub pattern_name: Option<String>,
    pub duration_sec: f64,
    pub net_bps: f64,
    pub abs_net_bps: f64,
    pub favorable_bps: f64,
    pub adverse_bps: f64,
    pub opposite_bps: f64,
    pub retention: f64,
    pub confidence: f64,
    pub pattern_confidence: f64,
    pub pattern_length: f64,
}

impl StepFeatures for StepInput {
    fn categorical(&self, feature: &str) -> &str {
        match feature {
            "micro_trend" => &self.micro_trend,
            "direction_hint" => &self.direction_hint,
            "vector_hint" => &self.vector_hint,
            "bias_hint" => &self.bias_hint,
            "quality_hint" => &self.quality_hint,
            "behavior_hint" => &self.behavior_hint,
            "side" => &self.side,
            "pattern.name" => self.pattern_name.as_deref().unwrap_or(PATTERN_NONE),
            _ => MISSING_CATEGORY,
        }
    }

    fn numeric(&self, feature: &str) -> f32 {
        let value = match feature {
            "duration_sec" => self.duration_sec,
            "net_bps" => self.net_bps,
            "abs_net_bps" => self.abs_net_bps,
            "favorable_bps" => self.favorable_bps,
            "adverse_bps" => self.adverse_bps,
            "opposite_bps" => self.opposite_bps,
            "retention" => self.retention,
            "confidence" => self.confidence,
            "pattern.confidence" => self.pattern_confidence,
            "pattern.length" => self.pattern_length,
            _ => 0.0,
        };
        value as f32
    }
}

/// Encode one step into the running id/value buffers (canonical feature order).
pub fn encode_step(
    step: &impl StepFeatures,
    vocab: &FeatureVocab,
    categorical: &mut Vec<u32>,
    numeric: &mut Vec<f32>,
) {
    for feature in CATEGORICAL_FEATURES {
        let id = vocab
            .get(&format!("sequence.{feature}"))
            .map(|entry| entry.id(step.categorical(feature)))
            .unwrap_or(0);
        categorical.push(id);
    }
    for feature in NUMERIC_FEATURES {
        numeric.push(step.numeric(feature));
    }
}

/// Encode an ordered window of steps into `ModelInputs` of shape `[1, steps, *]`.
pub fn encode_steps<S: StepFeatures>(
    steps: &[S],
    vocab: &FeatureVocab,
    device: &Device,
) -> Result<ModelInputs> {
    let n_cat = CATEGORICAL_FEATURES.len();
    let n_num = NUMERIC_FEATURES.len();
    let mut categorical = Vec::with_capacity(steps.len() * n_cat);
    let mut numeric = Vec::with_capacity(steps.len() * n_num);
    for step in steps {
        encode_step(step, vocab, &mut categorical, &mut numeric);
    }
    Ok(ModelInputs {
        categorical: Tensor::from_vec(categorical, (1, steps.len(), n_cat), device)?,
        numeric: Tensor::from_vec(numeric, (1, steps.len(), n_num), device)?,
    })
}
