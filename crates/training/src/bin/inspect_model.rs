//! Build the model with random weights and run one forward pass on a small batch
//! to prove the architecture wires up and the output shapes are right.

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use structure_core::config::ModelConfig;
use structure_core::model::MarketStructureModel;
use structure_core::sequence::BLOCK_LABEL_FIELDS;
use structure_core::tensors::ModelInputs;
use training::builder::build_tensors;
use training::dataset::{build_vocab, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "datasets/market_contexts.jsonl".to_owned());

    let device = Device::Cpu;
    let contexts = load_contexts(&path)?;
    let block_size = contexts[0].metadata.shape.block_size;
    let vocab = build_vocab(&contexts);
    let tensors = build_tensors(&contexts, &vocab, &device)?;

    let config = ModelConfig {
        categorical_embedding_dim: 8,
        numeric_projection_dim: 32,
        step_representation_dim: 64,
        gru_hidden_dim: 96,
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MarketStructureModel::new(&config, &vocab, block_size, vb)?;

    let batch = 16usize;
    let inputs = ModelInputs {
        categorical: tensors.inputs.categorical.narrow(0, 0, batch)?,
        numeric: tensors.inputs.numeric.narrow(0, 0, batch)?,
    };
    let output = model.forward(&inputs)?;

    println!("batch: {batch}");
    println!("--- block heads ({}) ---", output.block_logits.len());
    for (field, logits) in BLOCK_LABEL_FIELDS.iter().zip(&output.block_logits) {
        println!("  {field}: {:?}", logits.dims());
    }
    Ok(())
}
