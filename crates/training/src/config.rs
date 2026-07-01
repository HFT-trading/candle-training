//! App config loaded from `config.yml` (model hyper-params + training knobs).

use std::error::Error;
use std::fs::File;

use serde::Deserialize;
use structure_core::config::ModelConfig;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub validation_fraction: f64,
    pub shuffle_seed: u64,
    /// Apply inverse-frequency class weights to EVERY classification head.
    pub use_class_weights: bool,
    /// Block fields that always get class weights even when `use_class_weights`
    /// is off — for rare flags (e.g. `absorption`) that otherwise collapse to
    /// the majority class while the balanced heads stay sharp unweighted.
    #[serde(default)]
    pub weighted_block_fields: Vec<String>,
    /// Stop after this many epochs without val improvement (0 disables).
    pub early_stop_patience: usize,
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Ok(serde_yaml::from_reader(File::open("config.yml")?)?)
    }
}
