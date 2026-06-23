mod loader;
mod schema;
mod sequence;

pub use loader::{DatasetError, load_dataset, load_schema, load_sequence_json};
pub use schema::{FeatureSchema, STATE_TARGET_FEATURES};
pub use sequence::{MarketSequence, SequenceDebugSemantics, SequenceStep};

pub struct MarketDataset {
    pub schema: FeatureSchema,
    pub sequences: Vec<MarketSequence>,
}
