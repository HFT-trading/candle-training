//! Model hyper-parameters, shared by training and serving (loaded from
//! `config.yml` at train time, saved alongside the model for serving).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub categorical_embedding_dim: usize,
    pub numeric_projection_dim: usize,
    pub step_representation_dim: usize,
    pub gru_hidden_dim: usize,
}
