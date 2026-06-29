//! Categorical vocabulary built by scanning the dataset.
//!
//! The schema JSON that ships with the dataset is incomplete and level-specific
//! (e.g. `block.path_quality` has 5 classes, `context.path_quality` has 3), so
//! we build vocab directly from data and key it by `"<level>.<field>"`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Id reserved for values never seen during vocab construction.
pub const UNKNOWN_ID: u32 = 0;

/// One categorical field: value -> dense id (1-based; 0 is unknown) plus the
/// observation counts used later for class weighting.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub value_to_id: BTreeMap<String, u32>,
    pub counts: BTreeMap<String, u64>,
}

impl Vocab {
    /// Record one observation, assigning a fresh id on first sight.
    pub fn observe(&mut self, value: &str) {
        *self.counts.entry(value.to_owned()).or_insert(0) += 1;
        if !self.value_to_id.contains_key(value) {
            let next = self.value_to_id.len() as u32 + 1;
            self.value_to_id.insert(value.to_owned(), next);
        }
    }

    pub fn id(&self, value: &str) -> u32 {
        self.value_to_id.get(value).copied().unwrap_or(UNKNOWN_ID)
    }

    /// Number of known classes (excludes the unknown id) — i.e. head width.
    pub fn class_count(&self) -> usize {
        self.value_to_id.len()
    }

    /// Inverse of training's `id - 1`: the class string for a 0-based head index.
    pub fn label(&self, class_index: usize) -> Option<&str> {
        let id = class_index as u32 + 1;
        self.value_to_id
            .iter()
            .find(|(_, value)| **value == id)
            .map(|(key, _)| key.as_str())
    }
}

/// All categorical fields, keyed by `"<level>.<field>"`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FeatureVocab {
    pub fields: BTreeMap<String, Vocab>,
}

impl FeatureVocab {
    pub fn observe(&mut self, key: &str, value: &str) {
        self.fields.entry(key.to_owned()).or_default().observe(value);
    }

    pub fn get(&self, key: &str) -> Option<&Vocab> {
        self.fields.get(key)
    }

    /// Class string for `"<level>.<field>"` and a 0-based head index.
    pub fn label(&self, key: &str, class_index: usize) -> Option<&str> {
        self.get(key).and_then(|vocab| vocab.label(class_index))
    }
}
