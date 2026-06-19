use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Deserialize)]
pub struct MarketSequence {
    pub sample_id: String,
    pub source_name: String,
    pub seq_len: usize,
    pub state_features: Vec<SequenceStep>,
    pub range_telemetry: Vec<SequenceStep>,
    pub cycle_context: Vec<SequenceStep>,
    pub episode_context: Vec<SequenceStep>,
    pub future_outcomes: Map<String, Value>,
    pub debug_semantics: SequenceDebugSemantics,
}

#[derive(Debug, Deserialize)]
pub struct SequenceDebugSemantics {
    pub per_step: Vec<SequenceStep>,
}

#[derive(Debug, Deserialize)]
pub struct SequenceStep {
    pub snapshot_id: String,
    pub values: Map<String, Value>,
}

impl MarketSequence {
    pub fn value_at(&self, step: usize, feature: &str) -> Option<&Value> {
        [
            &self.state_features,
            &self.range_telemetry,
            &self.cycle_context,
            &self.episode_context,
        ]
        .into_iter()
        .find_map(|group| group.get(step)?.values.get(feature))
    }

    pub fn state_target_at(&self, step: usize, feature: &str) -> Option<&Value> {
        self.debug_semantics.per_step.get(step)?.values.get(feature)
    }
}
