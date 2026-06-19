mod loader;
mod schema;
mod sequence;

pub use loader::{DatasetError, load_dataset};
pub use schema::FeatureSchema;
pub use sequence::{MarketSequence, SequenceStep};

pub struct MarketDataset {
    pub schema: FeatureSchema,
    pub sequences: Vec<MarketSequence>,
}
