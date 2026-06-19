mod loader;
mod schema;
mod sequence;

pub use loader::{DatasetError, load_dataset};
pub use schema::{FeatureSchema, STATE_TARGET_FEATURES};
pub use sequence::{MarketSequence, SequenceDebugSemantics, SequenceStep};

pub struct MarketDataset {
    pub schema: FeatureSchema,
    pub sequences: Vec<MarketSequence>,
}
