//! One JSONL row of `market_contexts.jsonl` (schema v2).
//!
//! We only declare the fields we consume; serde ignores the rest. In particular
//! `training_data.{blocks,context}` are intentionally dropped (summaries would
//! trivialize the deterministic labels). Block-to-block relations are no longer
//! a model target — transitions are derived in the report rules layer from the
//! per-block reads instead.

use serde::Deserialize;
use serde_json::{Map, Value};
use structure_core::input::StepFeatures;
use structure_core::sequence::{MISSING_CATEGORY, PATTERN_NONE};

#[derive(Debug, Deserialize)]
pub struct Context {
    pub metadata: Metadata,
    pub training_data: TrainingData,
    pub labels: Labels,
}

#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub source_name: String,
    pub shape: Shape,
}

#[derive(Debug, Deserialize)]
pub struct Shape {
    pub sequence_len: usize,
    pub block_size: usize,
    pub context_blocks: usize,
}

#[derive(Debug, Deserialize)]
pub struct TrainingData {
    pub sequences: Vec<Sequence>,
}

#[derive(Debug, Deserialize)]
pub struct Sequence {
    /// Numeric telemetry (plus identity fields we ignore during encoding).
    pub metadata: Map<String, Value>,
    /// Categorical semantic features.
    pub vector: Map<String, Value>,
    pub pattern: Pattern,
}

impl StepFeatures for Sequence {
    fn categorical(&self, feature: &str) -> &str {
        if feature == "pattern.name" {
            self.pattern.name.as_deref().unwrap_or(PATTERN_NONE)
        } else {
            self.vector
                .get(feature)
                .and_then(|value| value.as_str())
                .unwrap_or(MISSING_CATEGORY)
        }
    }

    fn numeric(&self, feature: &str) -> f32 {
        match feature {
            "pattern.confidence" => self.pattern.confidence as f32,
            "pattern.length" => self.pattern.length as f32,
            _ => self
                .metadata
                .get(feature)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as f32,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Pattern {
    pub name: Option<String>,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub length: i64,
}

#[derive(Debug, Deserialize)]
pub struct Labels {
    pub blocks: Vec<Map<String, Value>>,
}

/// Canonical class token for a categorical label value. Strings pass through;
/// booleans (e.g. `absorption`) map to `"true"`/`"false"` so they ride the same
/// vocabulary/classification path as the string-valued fields.
pub fn label_token(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}
