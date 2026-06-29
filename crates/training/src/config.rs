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
    pub block_loss_weight: f64,
    pub relation_loss_weight: f64,
    pub use_class_weights: bool,
    /// Stop after this many epochs without val improvement (0 disables).
    pub early_stop_patience: usize,
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Ok(serde_yaml::from_reader(File::open("config.yml")?)?)
    }
}
