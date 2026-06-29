//! One JSONL row of `market_contexts.jsonl` (schema v2).
//!
//! We only declare the fields we consume; serde ignores the rest. In particular
//! `training_data.{blocks,context,block_relations}` are intentionally dropped:
//! summaries would trivialize the task and `block_relations` duplicates
//! `labels.relations` (a leak).

use serde::Deserialize;
use serde_json::{Map, Value};
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

impl Sequence {
    /// Categorical value for a feature name (handles the nested `pattern.name`
    /// and missing fields). Shared by vocab building and tensor building.
    pub fn categorical(&self, feature: &str) -> String {
        if feature == "pattern.name" {
            self.pattern
                .name
                .clone()
                .unwrap_or_else(|| PATTERN_NONE.to_owned())
        } else {
            self.vector
                .get(feature)
                .and_then(|value| value.as_str())
                .unwrap_or(MISSING_CATEGORY)
                .to_owned()
        }
    }

    /// Numeric value for a feature name (handles the nested `pattern.*`).
    pub fn numeric(&self, feature: &str) -> f32 {
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
    pub relations: Vec<Map<String, Value>>,
}
