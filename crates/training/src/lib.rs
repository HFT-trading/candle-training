//! Training crate: dataset loading, tensor building, trainer and the
//! train/inspect binaries. The model definition itself lives in `structure-core`.

pub mod builder;
pub mod config;
pub mod dataset;
pub mod trainer;
