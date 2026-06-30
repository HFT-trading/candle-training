//! Build the categorical vocabulary from the loaded dataset.
//!
//! Input categorical features are keyed `"sequence.<field>"`; supervised labels
//! are keyed `"blocks.<field>"` / `"relations.<field>"`.

use structure_core::input::StepFeatures;
use structure_core::sequence::{BLOCK_LABEL_FIELDS, CATEGORICAL_FEATURES, RELATION_LABEL_FIELDS};
use structure_core::vocab::FeatureVocab;

use super::context::Context;

pub fn build_vocab(contexts: &[Context]) -> FeatureVocab {
    let mut vocab = FeatureVocab::default();
    for context in contexts {
        for sequence in &context.training_data.sequences {
            for feature in CATEGORICAL_FEATURES {
                vocab.observe(
                    &format!("sequence.{feature}"),
                    sequence.categorical(feature),
                );
            }
        }
        for block in &context.labels.blocks {
            for field in BLOCK_LABEL_FIELDS {
                if let Some(value) = block.get(field).and_then(|value| value.as_str()) {
                    vocab.observe(&format!("blocks.{field}"), value);
                }
            }
        }
        for relation in &context.labels.relations {
            for field in RELATION_LABEL_FIELDS {
                if let Some(value) = relation.get(field).and_then(|value| value.as_str()) {
                    vocab.observe(&format!("relations.{field}"), value);
                }
            }
        }
    }
    vocab
}
