//! Build training tensors from the dataset and print their shapes/dtypes — the
//! checkpoint proving the builder works end to end on the real file.

use candle_core::Device;
use training::builder::build_tensors;
use training::dataset::{build_vocab, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "datasets/market_contexts.jsonl".to_owned());

    let contexts = load_contexts(&path)?;
    let vocab = build_vocab(&contexts);
    let tensors = build_tensors(&contexts, &vocab, &Device::Cpu)?;

    println!("contexts: {}", contexts.len());
    print_tensor("categorical input", &tensors.inputs.categorical);
    print_tensor("numeric input    ", &tensors.inputs.numeric);
    print_tensor("block targets    ", &tensors.block_targets);

    Ok(())
}

fn print_tensor(label: &str, tensor: &candle_core::Tensor) {
    println!("{label}: {:?} {:?}", tensor.dims(), tensor.dtype());
}
