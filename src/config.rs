use std::error::Error;
use std::fs::File;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub max_sequence_steps: usize,
    pub categorical_embedding_dim: usize,
    pub numeric_projection_dim: usize,
    pub step_representation_dim: usize,
    pub gru_hidden_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub validation_fraction: f64,
    pub shuffle_seed: u64,
    pub state_loss_weight: f64,
    pub outlook_loss_weight: f64,
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Ok(serde_yaml::from_reader(File::open("config.yml")?)?)
    }
}
